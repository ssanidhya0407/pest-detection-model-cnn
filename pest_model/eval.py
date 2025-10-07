import argparse
import torch
import cv2
import numpy as np
from model import build_model
from dataset import get_default_transforms, PestDataset
from utils import load_checkpoint


def predict_image(model, img_path, class_names, device, img_size=224):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = get_default_transforms("test", img_size)
    img = transform(image=img)["image"]
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    img = torch.from_numpy(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return class_names[pred], probs[0, pred].item()


def main():
    parser = argparse.ArgumentParser("Pest classifier inference")
    parser.add_argument("--img", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth")
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--img-size", type=int, default=224)
    args = parser.parse_args()

    ds = PestDataset(args.data_dir, split="train")
    class_names = ds.classes
    num_classes = len(class_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.backbone, num_classes=num_classes, pretrained=False)
    ckpt = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    pred_class, prob = predict_image(model, args.img, class_names, device, args.img_size)
    print(f"Prediction: {pred_class} (confidence: {prob:.2f})")

if __name__ == "__main__":
    main()
