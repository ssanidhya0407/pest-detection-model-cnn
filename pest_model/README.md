# 🐞 Advanced Pest CNN

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/github/license/ssanidhya0407/pest-detection-model-cnn.svg)](LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ssanidhya0407/pest-detection-model-cnn/blob/main/pest_model/demo.ipynb)

> **Interactive CNN for pest classification using PyTorch, timm, and Albumentations.**

---

## 🚀 Quick Start

#### 1. Setup Environment

<details>
<summary><b>Expand for step-by-step setup</b></summary>

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
</details>

#### 2. Train the Model

```bash
python train.py --data-dir "../pest" --backbone resnet50 --epochs 20 --batch-size 32
```
> ℹ️ Try other backbones: `efficientnet_b0`, `convnext_tiny`, etc.

#### 3. Run Inference (coming soon!)

---

## 🏆 Features

- **PyTorch Dataset**: Flexible, fast ([`dataset.py`](dataset.py))
- **Model Zoo**: Any `timm` backbone + custom head ([`model.py`](model.py))
- **Training Loop**: AMP, scheduler, checkpointing, logging ([`train.py`](train.py))
- **Utilities**: Metrics, random seed, save/load ([`utils.py`](utils.py))

---

## 📁 Dataset Structure

```
pest/
├── train/
│   ├── pest1/
│   ├── pest2/
│   └── ...
└── test/
    ├── pest1/
    ├── pest2/
    └── ...
```
> Each subfolder is a class label.

---

## 🖼️ Visual Example

<details>
<summary><b>See sample pest images</b></summary>

![Sample Pest Image](https://github.com/ssanidhya0407/pest-detection-model-cnn/raw/main/docs/sample_pest.jpg)
</details>

---

## 🧑‍💻 Interactive Demo

Try it in your browser:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ssanidhya0407/pest-detection-model-cnn/blob/main/pest_model/demo.ipynb)

---

## ⚙️ Advanced Usage

- To change backbone:
  ```bash
  python train.py --backbone efficientnet_b0
  ```
- See all options:
  ```bash
  python train.py --help
  ```

---

## ❓ FAQ

<details>
<summary><b>How do I add a new pest class?</b></summary>

Add a new folder to `train/` and `test/` named after the class and fill it with images.
</details>

<details>
<summary><b>Where do I find supported models?</b></summary>

See [timm model list](https://rwightman.github.io/pytorch-image-models/models/).
</details>

---

## 🆘 Troubleshooting

- **Error:** `ModuleNotFoundError: ...`
  - Make sure you installed dependencies: `pip install -r requirements.txt`
- **CUDA issues**
  - Check your PyTorch/CUDA installation.

---

## 🌱 Contribute

Pull requests, issues, and ideas welcome!  

---

## 📄 License

This project is [MIT licensed](../LICENSE).
