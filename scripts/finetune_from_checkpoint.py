#!/usr/bin/env python3
"""
Fine-tune a model for a dataset that possibly has different classes than the checkpoint.

Behavior:
- Builds a model for the dataset found under --data-dir (counts classes automatically).
- Loads weights from an existing checkpoint where shapes match (backbone weights are copied; head is left random if shapes differ).
- Optionally freezes the backbone and trains only the head.

Usage example:
  python scripts/finetune_from_checkpoint.py --data-dir pest --checkpoint checkpoints/best.pth --epochs 10 --batch-size 16 --lr 1e-4 --freeze-backbone

"""
import argparse
from pathlib import Path
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from pest_model.dataset import make_loader, get_default_transforms, PestDataset
from pest_model.model import build_model
from pest_model.utils import seed_everything, save_checkpoint, load_checkpoint, accuracy


def load_compatible_weights(model: torch.nn.Module, ckpt_path: Path):
    ckpt = load_checkpoint(str(ckpt_path), device='cpu')
    state = ckpt.get('model_state', ckpt)
    model_state = model.state_dict()
    loaded = 0
    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v
            loaded += 1
    model.load_state_dict(model_state)
    print(f"[INFO] Loaded {loaded} compatible parameters from {ckpt_path}")


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    total_samples = 0
    top1_acc = 0.0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        with autocast(enabled=(scaler is not None)):
            outputs = model(images)
            loss = criterion(outputs, targets)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--checkpoint', required=True, help='Path to existing checkpoint to reuse weights from')
    parser.add_argument('--backbone', default='convnext_tiny')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--freeze-backbone', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--out', type=str, default='checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)

    # discover classes from dataset
    train_ds = PestDataset(args.data_dir, split='train', transform=get_default_transforms('train'))
    val_ds = PestDataset(args.data_dir, split='test', transform=get_default_transforms('test'))
    num_classes = len(train_ds.classes)
    print(f"Found {num_classes} classes: {train_ds.classes}")

    train_loader = make_loader(args.data_dir, 'train', batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_loader = make_loader(args.data_dir, 'test', batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(args.backbone, num_classes=num_classes, pretrained=False)
    # load compatible weights from checkpoint (will copy backbone where shapes match)
    load_compatible_weights(model, Path(args.checkpoint))

    if args.freeze_backbone:
        for name, p in model.named_parameters():
            if name.startswith('backbone'):
                p.requires_grad = False
        print('[INFO] Backbone parameters frozen; training head only')

    model.to(device)

    class_counts = [0] * num_classes
    for _, label in train_ds.samples:
        class_counts[label] += 1
    class_weights = torch.tensor([1.0 / (c if c > 0 else 1) for c in class_counts], dtype=torch.float)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler() if (args.fp16 and device.type == 'cuda') else None

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{args.epochs} - train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} val_loss: {val_loss:.4f} val_acc: {val_acc:.4f} time: {time.time()-t0:.1f}s")

        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'val_acc': val_acc,
        }
        ckpt_name = out_dir / f"finetune_epoch{epoch}_valacc{val_acc:.4f}.pth"
        save_checkpoint(ckpt, str(ckpt_name))
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(ckpt, str(out_dir / 'finetune_best.pth'))


if __name__ == '__main__':
    main()
