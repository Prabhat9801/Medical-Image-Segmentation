# Medical Image Segmentation with Limited Data

## Comparing UNet, UNet++, and TransUNet on ISIC Dataset

This project investigates the performance of Vision Transformer (ViT)-based TransUNet compared to traditional CNN-based architectures (UNet and UNet++) for skin lesion segmentation on the ISIC dataset under limited-label regimes.

## 🎯 Project Objectives

- Implement and compare three state-of-the-art segmentation architectures
- Evaluate performance under different data availability scenarios (10%, 25%, 50%, 100%)
- Demonstrate the effectiveness of ViT-based models in low-data regimes

## 📁 Project Structure

```
medseg-vit-limited-data/
├── data/
│   ├── raw/                    # Raw ISIC dataset
│   │   └── isic/
│   │       ├── images/
│   │       └── masks/
│   └── processed/              # Preprocessed data (256x256)
│       └── isic/
│           ├── images/
│           ├── masks/
│           ├── splits.csv
│           └── splits_small.csv
├── notebooks/
│   ├── 01_isic_preprocessing.ipynb
│   ├── 02_model_testing.ipynb
│   └── 03_eval_colab.ipynb
├── src/
│   ├── datasets/
│   │   └── isic_dataset.py
│   ├── models/
│   │   ├── unet.py
│   │   ├── unetpp.py
│   │   └── transunet.py
│   ├── utils.py
│   ├── train.py
│   └── eval.py
├── reports/
│   ├── figures/
│   └── report.md
└── README.md
```

## 🚀 Quick Start

### Part A: Local Setup

1. **Create Python Environment**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

2. **Install Dependencies (Local)**
```bash
pip install numpy pandas pillow opencv-python matplotlib jupyter
```

3. **Download ISIC Dataset**
- Download ISIC 2018 Task 1 (Boundary Segmentation)
- Place images in `data/raw/isic/images/`
- Place masks in `data/raw/isic/masks/`

4. **Preprocess Data**
```bash
jupyter notebook notebooks/01_isic_preprocessing.ipynb
```

5. **Test Models Locally**
```bash
jupyter notebook notebooks/02_model_testing.ipynb
```

6. **Zip Processed Data**
```bash
# Zip data/processed/isic/ folder
# Upload to Google Drive as isic_processed_256.zip
```

### Part B: Google Colab Training

1. **Open Colab & Clone Repo**
```python
!git clone https://github.com/YOUR_USERNAME/Medical-Image-Segmentation.git
%cd Medical-Image-Segmentation
```

2. **Install Dependencies**
```python
!pip install torch torchvision timm albumentations opencv-python matplotlib pandas numpy
```

3. **Mount Drive & Extract Data**
```python
from google.colab import drive
drive.mount('/content/drive')

import zipfile, os
zip_path = "/content/drive/MyDrive/isic_processed_256.zip"
extract_path = "/content/Medical-Image-Segmentation/data/processed"
os.makedirs(extract_path, exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall(extract_path)
```

4. **Train Models**
```bash
# UNet with 10% data
!python -m src.train --model unet --epochs 30 --batch_size 8 --data_fraction 0.1

# UNet++ with 10% data
!python -m src.train --model unetpp --epochs 30 --batch_size 8 --data_fraction 0.1

# TransUNet with 10% data
!python -m src.train --model transunet --epochs 30 --batch_size 4 --data_fraction 0.1
```

5. **Evaluate Models**
```bash
!python -m src.eval --model unet --checkpoint experiments/unet_10pct.pt
```

## 📊 Models

### 1. UNet
Classic encoder-decoder architecture with skip connections for medical image segmentation.

### 2. UNet++
Enhanced UNet with nested skip pathways and deep supervision.

### 3. TransUNet
Hybrid CNN-Transformer architecture leveraging Vision Transformer for global context.

## 📈 Expected Results

Performance comparison across different data fractions:
- **10% data**: TransUNet expected to outperform CNNs
- **25% data**: TransUNet maintains advantage
- **50% data**: Gap narrows
- **100% data**: All models perform well

## 🔬 Evaluation Metrics

- **Dice Coefficient**: Overlap between prediction and ground truth
- **IoU (Intersection over Union)**: Jaccard index
- **Visual Comparisons**: Side-by-side predictions

## 📝 Citation

If you use this code, please cite:

```bibtex
@misc{medseg-vit-limited-data,
  title={Medical Image Segmentation with Vision Transformers under Limited Data},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/YOUR_USERNAME/Medical-Image-Segmentation}}
}
```

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- ISIC Dataset: International Skin Imaging Collaboration
- UNet: Ronneberger et al., 2015
- UNet++: Zhou et al., 2018
- TransUNet: Chen et al., 2021
