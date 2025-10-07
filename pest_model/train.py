import argparse
import os
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from dataset import PestDataset, make_loader, get_default_transforms
from model import build_model
from utils import seed_everything, accuracy, save_checkpoint, load_checkpoint


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    total_samples = 0
    top1_acc = 0.0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        # If targets are one-hot (MixUp), use BCEWithLogitsLoss
        if targets.ndim == 2:
            targets = targets.to(device)
            loss_fn = nn.BCEWithLogitsLoss()
        else:
            targets = targets.to(device, non_blocking=True)
            loss_fn = criterion

        optimizer.zero_grad()
        with autocast(enabled=(scaler is not None)):
            outputs = model(images)
            loss = loss_fn(outputs, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
        # For MixUp, use argmax for accuracy
        if targets.ndim == 2:
            pred = torch.argmax(outputs, dim=1)
            true = torch.argmax(targets, dim=1)
            top1_acc += (pred == true).float().sum().item() / bs
        else:
            top1_acc += accuracy(outputs, targets, topk=(1,))[0] * bs / 100.0

    return total_loss / total_samples, top1_acc / total_samples


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    top1_acc = 0.0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, targets)

            bs = images.size(0)
            total_loss += loss.item() * bs
            total_samples += bs
            top1_acc += accuracy(outputs, targets, topk=(1,))[0] * bs / 100.0

    return total_loss / total_samples, top1_acc / total_samples


def main():
    parser = argparse.ArgumentParser("Pest classifier trainer")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="convnext_tiny")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--out", type=str, default="checkpoints")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing for CrossEntropyLoss")
    parser.add_argument("--find-lr", action="store_true", help="Run learning rate finder and exit")
    args = parser.parse_args()

    seed_everything(args.seed)

    train_ds = PestDataset(args.data_dir, split="train", transform=get_default_transforms("train", args.img_size))
    val_ds = PestDataset(args.data_dir, split="test", transform=get_default_transforms("test", args.img_size))

    num_classes = len(train_ds.classes)
    print(f"Found {num_classes} classes: {train_ds.classes}")

    train_loader = make_loader(args.data_dir, "train", batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, size=args.img_size)
    val_loader = make_loader(args.data_dir, "test", batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, size=args.img_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.backbone, num_classes=num_classes, pretrained=args.pretrained)
    model.to(device)

    # Compute class weights for imbalance
    class_counts = [0] * num_classes
    for _, label in train_ds.samples:
        class_counts[label] += 1
    class_weights = torch.tensor([1.0 / (c if c > 0 else 1) for c in class_counts], dtype=torch.float)
    class_weights = class_weights / class_weights.sum() * num_classes  # normalize
    class_weights = class_weights.to(device)
    print(f"Class counts: {class_counts}")
    print(f"Class weights: {class_weights.tolist()}")
    if args.label_smoothing > 0.0:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    # Learning rate finder utility
    if args.find_lr:
        print("Running learning rate finder...")
        import matplotlib.pyplot as plt
        lrs = []
        losses = []
        init_lr = 1e-7
        final_lr = 1
        num_iter = min(100, len(train_loader))
        optimizer.param_groups[0]["lr"] = init_lr
        for i, (images, targets) in enumerate(train_loader):
            if i >= num_iter:
                break
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            lr = init_lr * (final_lr / init_lr) ** (i / num_iter)
            optimizer.param_groups[0]["lr"] = lr
            lrs.append(lr)
            losses.append(loss.item())
        plt.plot(lrs, losses)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('LR Finder')
        plt.savefig('lr_finder.png')
        print("LR finder plot saved as lr_finder.png")
        return
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = GradScaler() if args.fp16 and device.type == 'cuda' else None

    best_acc = 0.0
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # TensorBoard logging
    from torch.utils.tensorboard import SummaryWriter
    tb_writer = SummaryWriter(log_dir=str(out_dir))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch}/{args.epochs} - train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} val_loss: {val_loss:.4f} val_acc: {val_acc:.4f} time: {time.time()-t0:.1f}s")
        tb_writer.add_scalar('Loss/train', train_loss, epoch)
        tb_writer.add_scalar('Loss/val', val_loss, epoch)
        tb_writer.add_scalar('Acc/train', train_acc, epoch)
        tb_writer.add_scalar('Acc/val', val_acc, epoch)

        # save checkpoint with val_acc in filename
        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'val_acc': val_acc,
        }
        ckpt_name = f"ckpt_epoch{epoch}_valacc{val_acc:.4f}.pth"
        save_checkpoint(ckpt, str(out_dir / ckpt_name))

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(ckpt, str(out_dir / "best.pth"))


if __name__ == '__main__':
    main()
