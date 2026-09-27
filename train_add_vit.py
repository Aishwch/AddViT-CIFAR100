"""
Training script for Add-Vit on CIFAR-10 / CIFAR-100, following Table 4 of the paper as
closely as practical:

    Epochs               300
    Batch size           64   (paper's Table 4; Sec 4.2 mentions 128 for the main run --
                                the code defaults to 128 for speed, override with --batch-size 64
                                to match Table 4 exactly)
    Learning rate        1e-4
    Data augmentation    AutoAugment
    Label smoothing      0.1
    Random erase prob    0.25
    DropPath             0.1
    Cutmix alpha / Mixup alpha   0.8 / 1.0
    EMA decay            0   (i.e. no EMA)
    Optimizer            AdamW, eps=1e-8, betas=(0.9, 0.999), weight_decay=0.05
    Schedule             cosine, 10 warm-up epochs

Notes on reproducibility:
  - The paper reports 81.25% top-1 on CIFAR-100 and 96.67% on CIFAR-10 for their best
    Add-Vit (N=6, ~26.8M params) trained from scratch for 300 epochs. No official code is
    released, several architectural details are under-specified (see add_vit_model.py's
    comments), and exact numbers depend on details (exact channel ratios, weight
    initialization, seed, hardware/AMP behaviour) that cannot be fully recovered from the
    paper text alone. This script reproduces the *described* training recipe faithfully;
    treat the paper's number as a target to get close to, not a guarantee.
  - 300 epochs on CIFAR-100 with a ViT-hybrid model will take several hours even on a
    Colab T4/A100 GPU. Use --epochs to shorten for a quick sanity run first (e.g. 20-30
    epochs), then launch the full run separately (Colab free tier disconnects after
    ~12h of continuous use, and idles out sooner if the tab isn't kept active -- consider
    Colab Pro, checkpointing every epoch, and resuming).
"""
import argparse
import os
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from add_vit_model import add_vit_cifar, count_parameters

try:
    from timm.data import Mixup
    from timm.loss import SoftTargetCrossEntropy
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


def get_dataloaders(dataset, data_dir, batch_size, num_workers=4):
    if dataset == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        ds_cls = torchvision.datasets.CIFAR10
        autoaugment_policy = T.AutoAugmentPolicy.CIFAR10
        num_classes = 10
    elif dataset == "cifar100":
        mean, std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
        ds_cls = torchvision.datasets.CIFAR100
        autoaugment_policy = T.AutoAugmentPolicy.CIFAR10  # torchvision has no CIFAR100 policy
        num_classes = 100
    else:
        raise ValueError(dataset)

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4, padding_mode="reflect"),
        T.RandomHorizontalFlip(),
        T.AutoAugment(policy=autoaugment_policy),
        T.ToTensor(),
        T.Normalize(mean, std),
        T.RandomErasing(p=0.25),  # Table 4: "Random erase prob 0.25"
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    train_set = ds_cls(root=data_dir, train=True, download=True, transform=train_tf)
    test_set = ds_cls(root=data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, num_classes


def cosine_warmup_lr(optimizer, base_lr, warmup_epochs, total_epochs, steps_per_epoch):
    """Returns a LambdaLR-compatible schedule function operating per-step."""
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, mixup_fn,
                     soft_ce, hard_ce):
    model.train()
    running_loss = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if mixup_fn is not None:
            x, y_soft = mixup_fn(x, y)
        else:
            y_soft = None

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            logits = model(x)
            loss = soft_ce(logits, y_soft) if y_soft is not None else hard_ce(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        running_loss += loss.item()
    return running_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar100")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./checkpoints")
    parser.add_argument("--model-size", choices=["small", "base", "large"], default="base")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)  # Sec 4.2 uses 128; Table 4 says 64
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup-alpha", type=float, default=1.0)
    parser.add_argument("--cutmix-alpha", type=float, default=0.8)
    parser.add_argument("--drop-path", type=float, default=0.1)
    parser.add_argument("--alpha-pmsa", type=float, default=0.5, help="PMSA blend ratio (Eq. 9)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    train_loader, test_loader, num_classes = get_dataloaders(
        args.dataset, args.data_dir, args.batch_size, args.num_workers)

    model = add_vit_cifar(num_classes=num_classes, size=args.model_size,
                          drop_path_rate=args.drop_path, alpha=args.alpha_pmsa)
    model.to(device)
    print(f"Model: Add-Vit ({args.model_size}), params: {count_parameters(model)/1e6:.2f}M, "
          f"token grid: {model.grid_hw}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    scheduler = cosine_warmup_lr(optimizer, args.lr, args.warmup_epochs, args.epochs,
                                  steps_per_epoch=len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    mixup_fn = None
    soft_ce = None
    if HAS_TIMM:
        mixup_fn = Mixup(mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha,
                          prob=1.0, switch_prob=0.5, mode="batch",
                          label_smoothing=args.label_smoothing, num_classes=num_classes)
        soft_ce = SoftTargetCrossEntropy()
    else:
        print("timm not found (pip install timm) -- training without Mixup/CutMix.")

    hard_ce = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    start_epoch = 0
    best_acc = 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("best_acc", 0.0)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler,
                                      device, mixup_fn, soft_ce, hard_ce)
        acc = evaluate(model, test_loader, device)
        best_acc = max(best_acc, acc)
        dt = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs} | loss {train_loss:.4f} | "
              f"test acc {acc:.2f}% | best {best_acc:.2f}% | {dt:.1f}s")

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "args": vars(args),
        }, os.path.join(args.output_dir, "last.pt"))
        if acc == best_acc:
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best.pt"))

    print(f"Training done. Best {args.dataset} accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
