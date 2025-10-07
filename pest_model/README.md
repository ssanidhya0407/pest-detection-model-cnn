Advanced Pest CNN
=================

This project trains an advanced CNN to classify pests using the dataset in `../pest` which contains `train/` and `test/` folders with one subfolder per class.

What's included
- `dataset.py` - PyTorch Dataset with Albumentations augmentations
- `model.py` - Model builder using `timm` pretrained backbones and a custom head
- `train.py` - Training loop with AMP, scheduler, checkpointing and logging
- `utils.py` - Helpers for metrics, seed, save/load checkpoints
- `requirements.txt` - Minimal Python dependencies

Quick start
1. Create and activate a Python 3.10+ venv.
2. Install requirements: `pip install -r requirements.txt`
3. Run training (example):
   `python train.py --data-dir "../pest" --backbone resnet50 --epochs 20 --batch-size 32`

Notes
- The script expects `data-dir` to contain `train/` and `test/` folders with class subfolders.
- For experimentation replace `--backbone` with any `timm` model name (e.g., `efficientnet_b0`, `convnext_tiny`).
