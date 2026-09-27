"""
Add-Vit: CNN-Transformer Hybrid Architecture for Small Data Paradigm Processing
Chen, Wu, Zhang, Xu, Liang — Neural Processing Letters (2024) 56:198

This is a from-scratch re-implementation based on reading the paper. The paper does not
release official code ("The code of the proposed method available on request"), and a
handful of implementation details are underspecified (exact channel-reduction ratios,
padding choices, and how the 224x224 illustration in Fig. 2 maps onto the 32x32 CIFAR
images actually used in the experiments). Everywhere a choice had to be made, it is
documented in a comment near the relevant code, and the choice follows the paper's stated
design goals (short final token length, overlapping-unfold tokenization, ECA-gated CNN
supplementation, predictive multi-head self-attention chained across blocks, and a
depthwise-separable "add-conv" unit per block).

Modules implemented:
  - ECA / ECA*      : efficient channel attention (Eq. 2) for spatial maps (Conv) and
                       for token sequences (Linear), Eq. 10.
  - OUTE            : Overlapping Unfold Token Embedding (Sec 3.1, Eq. 1).
  - ConvLayer       : downsample + ECA branch that supplies "Feature Map" info (Fig. 3b).
  - AddEmbedding    : the 3-stage tokenizer (Fig. 2, Eq. 3-6).
  - AddAttn         : bottleneck(1x1 down, GELU, 1x1 up) + ECA over attention maps (Fig. 4b).
  - PMSA            : Predictive Multi-head Self-Attention (Fig. 5, Eq. 7-9).
  - AddConv         : PW -> GELU -> DW -> PW -> ECA* depthwise-separable coding unit (Fig. 4c).
  - AddVitBlock     : PMSA + AddConv + MLP transformer block (Fig. 4a).
  - AddViT          : full classifier.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# Stochastic depth (DropPath) — used because Table 4 lists "DropPath 0.1"
# --------------------------------------------------------------------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# --------------------------------------------------------------------------------------
# ECA (Eq. 2): I' = I + sigmoid(Conv(Avg(I)))  — applied to (B, C, H, W) feature maps
# --------------------------------------------------------------------------------------
class ECA(nn.Module):
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        y = self.avg_pool(x)                      # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)        # (B, 1, C)
        y = self.conv(y)                           # (B, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1)       # (B, C, 1, 1)
        y = self.sigmoid(y)
        return x + x * y


# ECA* (Eq. 10): same idea but the Conv is replaced by a Linear, for token sequences
class ECAStar(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.linear = nn.Linear(channels, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, N, C)  -- Avg pools over the token/sequence dimension
        y = x.mean(dim=1, keepdim=True)   # (B, 1, C)
        y = self.sigmoid(self.linear(y))  # (B, 1, C)
        return x + x * y


# --------------------------------------------------------------------------------------
# OUTE: Overlapping Unfold Token Embedding (Sec 3.1, Eq. 1)
# Uses nn.Unfold (im2col) with kernel k, stride, padding, then a linear projection to embed_dim.
# --------------------------------------------------------------------------------------
class OUTE(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding)
        self.proj = nn.Linear(in_chans * kernel_size * kernel_size, embed_dim)

    def out_hw(self, h, w):
        oh = (h + 2 * self.padding - self.kernel_size) // self.stride + 1
        ow = (w + 2 * self.padding - self.kernel_size) // self.stride + 1
        return oh, ow

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        oh, ow = self.out_hw(h, w)
        tokens = self.unfold(x)                 # (B, C*k*k, L)
        tokens = tokens.transpose(1, 2)          # (B, L, C*k*k)
        tokens = self.proj(tokens)               # (B, L, embed_dim)
        return tokens, oh, ow


# --------------------------------------------------------------------------------------
# ConvLayer (Fig. 3b): downsample + ECA. Produces the "Feature Map" fused into each stage.
# --------------------------------------------------------------------------------------
class ConvLayer(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, stride: int):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_chans)
        self.act = nn.GELU()
        self.eca = ECA(out_chans)

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        x = self.eca(x)
        return x


# --------------------------------------------------------------------------------------
# Add-Embedding (Fig. 2, Eq. 3-6)
#
# Design note / assumption: Fig. 2 illustrates 224x224 -> 56x56 -> 28x28 -> fixed tokens,
# but the CIFAR experiments actually use 32x32 images, and the paper never states how the
# 224-scale illustration maps onto 32x32 inputs. The kernel sizes [7, 3, 3] are kept
# identical to the paper's best-performing ablation row (Table 6, 81.26% acc, essentially
# matching the paper's headline 81.25%). Strides are chosen (default 2, 2, 1) so that a
# 32x32 image ends with a 8x8 = 64-token grid: large enough for self-attention to do
# meaningful work, short enough to match the paper's stated goal of "opting for shorter
# tokens ... reduces model redundancy". Using the paper's literal Table 6 strides
# ([4, 2, 2]) on a 32x32 input would collapse the sequence to only ~4 tokens, which is
# almost certainly not what was intended (those strides were presumably tuned for a
# different preprocessing pipeline than what's described in Fig. 2's 224px illustration).
# tokenizer_kernels / tokenizer_strides are exposed as constructor args specifically so
# you can reproduce Table 6's exact settings and compare, if you want to experiment.
# --------------------------------------------------------------------------------------
class AddEmbedding(nn.Module):
    def __init__(self, img_size=32, in_chans=3, embed_dim=384,
                 kernel_sizes=(7, 3, 3), strides=(2, 2, 1)):
        super().__init__()
        k1, k2, k3 = kernel_sizes
        s1, s2, s3 = strides
        p1, p2, p3 = k1 // 2, k2 // 2, k3 // 2

        # Stage 1: raw image -> token map
        self.oute1 = OUTE(in_chans, embed_dim, k1, s1, p1)
        self.convlayer1 = ConvLayer(in_chans, embed_dim, stride=s1)

        # Stage 2: (embed_dim-channel) token map -> token map
        self.oute2 = OUTE(embed_dim, embed_dim, k2, s2, p2)
        self.convlayer2 = ConvLayer(embed_dim, embed_dim, stride=s2)

        # Final OUTE: fixed-length output tokens fed to the backbone
        self.oute3 = OUTE(embed_dim, embed_dim, k3, s3, p3)

        self.img_size = img_size

    @staticmethod
    def _tokens_to_map(tokens, h, w):
        b, l, c = tokens.shape
        return tokens.transpose(1, 2).reshape(b, c, h, w)

    def forward(self, x):
        # x: (B, 3, H, W)
        t0, h1, w1 = self.oute1(x)                      # T'_0  (Eq. 3, i=0)
        f0 = self.convlayer1(x)                          # I'_0  (Eq. 4, i=0)
        f0_tokens = f0.flatten(2).transpose(1, 2)         # (B, L1, C)
        t1 = t0 + f0_tokens                               # T'_1  (Eq. 5, i=0)
        map1 = self._tokens_to_map(t1, h1, w1)            # 3D token map for next stage

        t0b, h2, w2 = self.oute2(map1)                    # T'_1 -> next stage tokens
        f1 = self.convlayer2(map1)                        # I'_1
        f1_tokens = f1.flatten(2).transpose(1, 2)
        t2 = t0b + f1_tokens                               # T'_2 == "T2" in Eq. 6
        map2 = self._tokens_to_map(t2, h2, w2)

        tfinal, h3, w3 = self.oute3(map2)                  # Eq. 6: T_final = OUTE(T2)
        return tfinal, (h3, w3)


# --------------------------------------------------------------------------------------
# Add-Attn (Fig. 4b): bottleneck (1x1 down -> GELU -> 1x1 up) + ECA, applied to an
# attention map treated as a (B, heads, N, N) "image" (heads = channels).
# --------------------------------------------------------------------------------------
class AddAttn(nn.Module):
    def __init__(self, num_heads: int, reduction: int = 4):
        super().__init__()
        hidden = max(num_heads // reduction, 1)
        self.down = nn.Conv2d(num_heads, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.up = nn.Conv2d(hidden, num_heads, kernel_size=1)
        self.eca = ECA(num_heads)

    def forward(self, attn_map):
        # attn_map: (B, heads, N, N)
        x = self.down(attn_map)
        x = self.act(x)
        x = self.up(x)
        x = x + attn_map  # bottleneck residual
        x = self.eca(x)
        return x


# --------------------------------------------------------------------------------------
# Predictive Multi-Head Self-Attention (Fig. 5, Eq. 7-9)
# --------------------------------------------------------------------------------------
class PMSA(nn.Module):
    def __init__(self, dim, num_heads=6, qkv_bias=True, attn_drop=0.0, proj_drop=0.0, alpha=0.5):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.alpha = alpha

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.add_attn = AddAttn(num_heads)

    def forward(self, x, prev_attn=None):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, heads, N, head_dim)

        cur_logits = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N) -- raw logits
        a_cur = cur_logits.softmax(dim=-1)                    # A'_cur (Eq. 9)

        if prev_attn is None:
            # First block: alpha effectively 0 -> plain MSA (paper: "alpha=0 corresponds
            # to a vanilla transformer").
            a_final = a_cur
        else:
            a_pre = self.add_attn(prev_attn)      # A'_pre = add-attn(A_pre)   (Eq. 9)
            a_pre = a_pre.softmax(dim=-1)
            a_final = self.alpha * a_pre + (1 - self.alpha) * a_cur
            a_final = a_final / a_final.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        a_final = self.attn_drop(a_final)
        out = a_final @ v                          # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(b, n, c)
        out = self.proj_drop(self.proj(out))

        # `a_cur` (softmax(QK^T/sqrt(d))) is what feeds the *next* block's add-attn,
        # matching the "To Next Attention Map" arrow in Fig. 5b, which comes off the raw
        # softmax(QK^T) branch rather than the combined/predicted one.
        return out, a_cur


# --------------------------------------------------------------------------------------
# Add-Conv (Fig. 4c): PW -> GELU -> DW -> PW -> ECA*, operating on the spatial (non-cls)
# tokens reshaped back into a 2D grid; the cls token passes through unchanged.
# --------------------------------------------------------------------------------------
class AddConv(nn.Module):
    def __init__(self, dim, grid_hw):
        super().__init__()
        self.h, self.w = grid_hw
        self.pw1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.GELU()
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.pw2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.eca_star = ECAStar(dim)

    def forward(self, x):
        # x: (B, N, C) where N = 1 (cls) + h*w
        cls_tok, patch_tok = x[:, :1, :], x[:, 1:, :]
        b, n, c = patch_tok.shape
        assert n == self.h * self.w, "Add-Conv grid size mismatch"
        grid = patch_tok.transpose(1, 2).reshape(b, c, self.h, self.w)
        grid = self.pw1(grid)
        grid = self.act(grid)
        grid = self.dw(grid)
        grid = self.pw2(grid)
        out_tokens = grid.flatten(2).transpose(1, 2)          # (B, N, C)
        out_tokens = self.eca_star(out_tokens)
        return torch.cat([cls_tok, out_tokens], dim=1)


class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


# --------------------------------------------------------------------------------------
# Add-Vit Block (Fig. 4a): PMSA -> add-conv -> MLP, each with residuals + pre-norm.
# --------------------------------------------------------------------------------------
class AddVitBlock(nn.Module):
    def __init__(self, dim, num_heads, grid_hw, mlp_ratio=4.0, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, alpha=0.5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = PMSA(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop, alpha=alpha)
        self.drop_path1 = DropPath(drop_path)

        self.add_conv = AddConv(dim, grid_hw)
        self.drop_path2 = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)
        self.drop_path3 = DropPath(drop_path)

    def forward(self, x, prev_attn=None):
        attn_out, cur_attn = self.attn(self.norm1(x), prev_attn)
        x = x + self.drop_path1(attn_out)
        x = x + self.drop_path2(self.add_conv(x))
        x = x + self.drop_path3(self.mlp(self.norm2(x)))
        return x, cur_attn


# --------------------------------------------------------------------------------------
# Full model
# --------------------------------------------------------------------------------------
class AddViT(nn.Module):
    def __init__(self, img_size=32, in_chans=3, num_classes=100, embed_dim=384, depth=6,
                 num_heads=6, mlp_ratio=4.0, drop_rate=0.0, attn_drop_rate=0.0,
                 drop_path_rate=0.1, alpha=0.5, tokenizer_kernels=(7, 3, 3), tokenizer_strides=(2, 2, 1)):
        super().__init__()
        self.embed = AddEmbedding(img_size, in_chans, embed_dim,
                                   kernel_sizes=tokenizer_kernels, strides=tokenizer_strides)

        # Work out the token grid size produced by the tokenizer for this img_size.
        with torch.no_grad():
            dummy = torch.zeros(1, in_chans, img_size, img_size)
            _, (gh, gw) = self.embed(dummy)
        self.grid_hw = (gh, gw)
        num_patches = gh * gw

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # linear DropPath schedule
        self.blocks = nn.ModuleList([
            AddVitBlock(embed_dim, num_heads, self.grid_hw, mlp_ratio=mlp_ratio,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], alpha=alpha)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward_features(self, x):
        tokens, _ = self.embed(x)                       # (B, N_patch, C)
        b = tokens.shape[0]
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, tokens], dim=1) + self.pos_embed
        x = self.pos_drop(x)

        prev_attn = None
        for blk in self.blocks:
            x, prev_attn = blk(x, prev_attn)
        x = self.norm(x)
        return x[:, 0]  # cls token representation

    def forward(self, x):
        x = self.forward_features(x)
        return self.head(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def add_vit_cifar(num_classes=100, size="base", drop_path_rate=0.1, alpha=0.5):
    """
    Factory presets. 'base' targets roughly the paper's ~26.8M-parameter, N=6 config.
    Sizes are approximate — print count_parameters(model) after building and adjust
    embed_dim/mlp_ratio if you want to match the paper's parameter count more closely.
    """
    presets = {
        "small": dict(embed_dim=256, depth=6, num_heads=4, mlp_ratio=3.0),
        "base":  dict(embed_dim=384, depth=6, num_heads=6, mlp_ratio=4.0),
        "large": dict(embed_dim=448, depth=8, num_heads=7, mlp_ratio=4.0),
    }
    cfg = presets[size]
    return AddViT(img_size=32, in_chans=3, num_classes=num_classes, drop_path_rate=drop_path_rate,
                  alpha=alpha, tokenizer_kernels=(7, 3, 3), tokenizer_strides=(2, 2, 1), **cfg)


if __name__ == "__main__":
    m = add_vit_cifar(num_classes=100, size="base")
    print("Token grid:", m.grid_hw, " total tokens (incl cls):", m.pos_embed.shape[1])
    print("Params (M):", count_parameters(m) / 1e6)
    x = torch.randn(2, 3, 32, 32)
    y = m(x)
    print("Output shape:", y.shape)
