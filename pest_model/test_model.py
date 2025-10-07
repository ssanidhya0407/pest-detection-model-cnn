import torch
from dataset import make_loader, PestDataset
from model import build_model

DATA_DIR = 'C:/Users/vinay/OneDrive/Desktop/Pests_CNN/pest'
BACKBONE = 'resnet50'
IMG_SIZE = 224
BATCH_SIZE = 4

def main():
    ds = PestDataset(DATA_DIR, split='train')
    loader = make_loader(DATA_DIR, 'train', batch_size=BATCH_SIZE, num_workers=0, shuffle=False, size=IMG_SIZE)
    num_classes = len(ds.classes)
    model = build_model(BACKBONE, num_classes=num_classes, pretrained=False)
    batch = next(iter(loader))
    imgs, labels = batch
    print(f'Input shape: {imgs.shape}')
    out = model(imgs)
    print(f'Output shape: {out.shape}')
    print(f'Labels: {labels}')

if __name__ == '__main__':
    main()
