import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from dataset import PestDataset, get_default_transforms
from model import build_model
from utils import load_checkpoint

DATA_DIR = 'pest'
CHECKPOINT = 'checkpoints/best.pth'
BACKBONE = 'resnet50'
IMG_SIZE = 224
BATCH_SIZE = 16

def main():
    ds = PestDataset(DATA_DIR, split='test', transform=get_default_transforms('test', IMG_SIZE))
    class_names = ds.classes
    num_classes = len(class_names)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(BACKBONE, num_classes=num_classes, pretrained=False)
    ckpt = load_checkpoint(CHECKPOINT, device)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    for img, label in ds:
        img = img.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(img)
            pred = torch.argmax(logits, dim=1).item()
        y_true.append(label)
        y_pred.append(pred)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    plt.title('Grand Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    print('Confusion matrix saved as confusion_matrix.png')

if __name__ == '__main__':
    main()
