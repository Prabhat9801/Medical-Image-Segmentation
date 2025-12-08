# Medical Image Segmentation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Deep learning-based medical image segmentation using UNet, UNet++, and TransUNet architectures on the ISIC 2018 skin lesion dataset.

## 🌐 Live Demo

**Try the model now!** 👉 [**Skin Lesion Segmentation App**](https://huggingface.co/spaces/Prabhat9801/Skin-Lesion-Segmentation)

Upload a dermoscopic image and get instant AI-powered segmentation results with our best performing UNet++ model (86.08% Dice Score).

[![Open in Spaces](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue)](https://huggingface.co/spaces/Prabhat9801/Skin-Lesion-Segmentation)

## 🎯 Project Overview

This project implements and compares three state-of-the-art deep learning architectures for medical image segmentation:
- **UNet**: Classic encoder-decoder architecture
- **UNet++**: Nested U-Net with dense skip connections
- **TransUNet**: Hybrid CNN-Transformer architecture

### Key Results

🏆 **Best Model: UNet++ with 100% data**
- **Dice Score**: 86.08% ± 16.74%
- **IoU Score**: 78.31% ± 19.06%
- **Pixel Accuracy**: 94.93% ± 7.59%

## 📊 Complete Results

| Model | Data Fraction | Dice Score | IoU Score | Pixel Accuracy |
|-------|---------------|------------|-----------|----------------|
| **UNet++** | **100%** | **0.8608 ± 0.167** | **0.7831 ± 0.191** | **94.93% ± 7.59%** |
| UNet | 100% | 0.8589 ± 0.172 | 0.7818 ± 0.197 | 94.84% ± 8.23% |
| UNet++ | 50% | 0.8415 ± 0.197 | 0.7624 ± 0.216 | 94.51% ± 8.72% |
| UNet | 50% | 0.8377 ± 0.202 | 0.7583 ± 0.220 | 94.43% ± 8.74% |
| UNet++ | 25% | 0.8038 ± 0.212 | 0.7119 ± 0.229 | 93.43% ± 8.66% |
| UNet | 25% | 0.8019 ± 0.219 | 0.7116 ± 0.235 | 93.53% ± 9.11% |
| UNet | 10% | 0.7548 ± 0.227 | 0.6497 ± 0.242 | 91.92% ± 9.70% |
| UNet++ | 10% | 0.7288 ± 0.215 | 0.6119 ± 0.232 | 90.41% ± 10.16% |

### Performance Visualizations

#### Results Comparison
![Results Comparison](https://raw.githubusercontent.com/Prabhat9801/Medical-Image-Segmentation/main/reports/figures/results_comparison.png)

#### Performance Trends
![Performance Trends](https://raw.githubusercontent.com/Prabhat9801/Medical-Image-Segmentation/main/reports/figures/performance_trends.png)

*For detailed analysis, see [PROJECT_ACHIEVEMENTS.md](PROJECT_ACHIEVEMENTS.md)*

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Installation

```bash
# Clone the repository
git clone https://github.com/Prabhat9801/Medical-Image-Segmentation.git
cd Medical-Image-Segmentation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

The project uses the ISIC 2018 skin lesion segmentation dataset.

**Option 1: Use Pre-processed Data**
- Download `isic_processed_256.zip` from [Google Drive](https://drive.google.com/drive/folders/10WLdxr9UY8YZbCNxSFAqB-wo93ftndJf?usp=sharing)
- Extract to `data/processed/isic/`

**Option 2: Download and Process Raw Data**
```bash
python download_isic_dataset.py
python src/preprocess.py
```

### Training

Train models with different data fractions:

```bash
# UNet with 100% data
python -m src.train --model unet --epochs 20 --batch_size 24 --data_fraction 1.0

# UNet++ with 50% data
python -m src.train --model unetpp --epochs 20 --batch_size 20 --data_fraction 0.5

# TransUNet with 25% data
python -m src.train --model transunet --epochs 20 --batch_size 12 --data_fraction 0.25
```

### Evaluation

```bash
# Evaluate a trained model
python -m src.eval \
    --model unet \
    --checkpoint experiments/unet_100pct_XXXXXX/best_model.pt \
    --output_dir results/unet_100pct
```

## 📁 Project Structure

```
Medical-Image-Segmentation/
├── src/
│   ├── models/
│   │   ├── unet.py              # UNet implementation
│   │   ├── unetpp.py            # UNet++ implementation
│   │   └── transunet.py         # TransUNet implementation
│   ├── datasets/
│   │   └── isic_dataset.py      # Dataset loader
│   ├── train.py                 # Training script
│   ├── eval.py                  # Evaluation script
│   └── utils.py                 # Utility functions
├── notebooks/
│   ├── COLAB_TRAIN_UNET.ipynb   # Colab training for UNet
│   ├── COLAB_TRAIN_UNETPP.ipynb # Colab training for UNet++
│   ├── COLAB_TRAIN_TRANSUNET.ipynb
│   └── COLAB_RESULTS.ipynb      # Results evaluation
├── reports/
│   ├── FINAL_REPORT.md          # Comprehensive analysis
│   ├── results_summary.csv      # Results table
│   └── figures/                 # Visualizations
├── requirements.txt
└── README.md
```

## 🔬 Methodology

### Models

1. **UNet**
   - Classic encoder-decoder with skip connections
   - 64 base features
   - ~31M parameters

2. **UNet++**
   - Nested U-Net architecture
   - Dense skip pathways
   - 32 base features
   - ~9M parameters

3. **TransUNet**
   - Hybrid CNN-Transformer
   - ViT encoder + CNN decoder
   - 768 embedding dimensions
   - ~105M parameters

### Training Configuration

- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)
- **Loss Function**: Combined Dice + BCE Loss
- **LR Scheduler**: Cosine Annealing
- **Epochs**: 20 (optimized for fast training)
- **Mixed Precision**: FP16 for 2x speedup
- **Image Size**: 256×256 pixels
- **Data Augmentation**: 
  - Random horizontal/vertical flips
  - Random rotation (±15°)
  - Color jittering
  - Normalization (ImageNet stats)

### Evaluation Metrics

- **Dice Coefficient**: Measures overlap between prediction and ground truth
- **IoU (Jaccard Index)**: Intersection over Union
- **Pixel Accuracy**: Percentage of correctly classified pixels

## 📈 Key Findings

1. **Data Efficiency**: Both UNet and UNet++ show significant improvement from 10% to 100% data
2. **Model Comparison**: UNet++ slightly outperforms UNet at higher data fractions
3. **Convergence**: 20 epochs with mixed precision training provides good results
4. **Stability**: Low standard deviations indicate consistent performance

## 🔗 Resources

### Trained Models & Results
- **Experiments**: [Google Drive - Experiments](https://drive.google.com/drive/folders/14-wNH4hWoinkh1I1blsrmf_f9gXcwjyr?usp=sharing)
- **Training Data**: [Google Drive - Processed Dataset](https://drive.google.com/drive/folders/10WLdxr9UY8YZbCNxSFAqB-wo93ftndJf?usp=sharing)

### Documentation
- [Complete Report](reports/FINAL_REPORT.md)
- [Results CSV](reports/results_summary.csv)
- [Colab Notebooks](notebooks/)

## 🛠️ Advanced Usage

### Google Colab Training

For training on Google Colab with free GPU:

1. Upload `isic_processed_256.zip` to your Google Drive
2. Open the appropriate notebook:
   - `notebooks/COLAB_TRAIN_UNET.ipynb`
   - `notebooks/COLAB_TRAIN_UNETPP.ipynb`
   - `notebooks/COLAB_TRAIN_TRANSUNET.ipynb`
3. Run all cells
4. Results are automatically saved to Google Drive

### Custom Training

```python
from src.models.unet import UNet
from src.datasets.isic_dataset import create_dataloaders
from src.utils import CombinedLoss

# Create model
model = UNet(in_channels=3, out_channels=1, features=64)

# Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    csv_file='data/processed/isic/splits.csv',
    batch_size=16,
    data_fraction=0.5
)

# Train
criterion = CombinedLoss(dice_weight=0.5, bce_weight=0.5)
# ... training loop
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{medical-image-segmentation-2024,
  author = {Prabhat},
  title = {Medical Image Segmentation: UNet, UNet++, and TransUNet},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/Prabhat9801/Medical-Image-Segmentation}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- ISIC 2018 Challenge for the dataset
- Original UNet paper: [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)
- UNet++ paper: [Zhou et al., 2018](https://arxiv.org/abs/1807.10165)
- TransUNet paper: [Chen et al., 2021](https://arxiv.org/abs/2102.04306)

## 📧 Contact

For questions or feedback, please open an issue or contact:
- GitHub: [@Prabhat9801](https://github.com/Prabhat9801)

---

**⭐ Star this repository if you found it helpful!**
