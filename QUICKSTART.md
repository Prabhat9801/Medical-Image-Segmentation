# Quick Start Guide
## Medical Image Segmentation Project

This guide will help you get started with the project quickly.

---

## 📋 Prerequisites

- Python 3.8+
- Git
- Google Account (for Colab and Drive)
- ISIC 2018 Dataset access

---

## 🚀 Part A: Local Setup (Your Computer)

### Step 1: Clone Repository

```bash
git clone https://github.com/Prabhat9801/Medical-Image-Segmentation.git
cd Medical-Image-Segmentation
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download ISIC Dataset

1. Visit: https://challenge.isic-archive.com/data/
2. Download ISIC 2018 Task 1 dataset
3. Extract to:
   - Images → `data/raw/isic/images/`
   - Masks → `data/raw/isic/masks/`

### Step 5: Preprocess Data

```bash
jupyter notebook notebooks/01_isic_preprocessing.ipynb
```

Run all cells to:
- Resize images to 256×256
- Create train/val/test splits
- Save processed data

### Step 6: Test Models (Optional)

```bash
jupyter notebook notebooks/02_model_testing.ipynb
```

This verifies all models work correctly.

### Step 7: Zip Processed Data

**Windows:**
```powershell
Compress-Archive -Path data\processed\isic -DestinationPath isic_processed_256.zip
```

**Linux/Mac:**
```bash
zip -r isic_processed_256.zip data/processed/isic/
```

### Step 8: Upload to Google Drive

1. Upload `isic_processed_256.zip` to your Google Drive
2. Note the location (e.g., `MyDrive/isic_processed_256.zip`)

### Step 9: Push Code to GitHub

```bash
git add .
git commit -m "Initial commit: Project setup complete"
git push origin main
```

---

## ☁️ Part B: Google Colab Training

### Step 1: Open Colab

1. Go to: https://colab.research.google.com/
2. File → New Notebook
3. Runtime → Change Runtime Type → GPU (T4 or better)

### Step 2: Run Training Notebook

1. Upload `notebooks/03_colab_training.ipynb` to Colab
2. Or create new notebook and copy cells
3. Update GitHub username in clone command
4. Run all cells sequentially

### Step 3: Train Models

Train each model with different data fractions:

```bash
# UNet - 10% data
!python -m src.train --model unet --epochs 30 --batch_size 8 --data_fraction 0.1

# UNet++ - 10% data
!python -m src.train --model unetpp --epochs 30 --batch_size 8 --data_fraction 0.1

# TransUNet - 10% data
!python -m src.train --model transunet --epochs 30 --batch_size 4 --data_fraction 0.1
```

Repeat for `--data_fraction 0.25`, `0.5`, and `1.0`.

### Step 4: Evaluate Models

```bash
!python -m src.eval --model unet --checkpoint experiments/unet_10pct_*/best_model.pt
```

### Step 5: Save Results

Copy results back to Google Drive:

```bash
!cp -r experiments /content/drive/MyDrive/medseg_experiments/
!cp -r reports /content/drive/MyDrive/medseg_reports/
```

---

## 📊 Part C: Analysis (Local)

### Step 1: Download Results

Download from Google Drive:
- `medseg_experiments/`
- `medseg_reports/`

### Step 2: Complete Report

Edit `reports/report.md` and fill in:
- Actual performance numbers
- Training curves
- Visualizations

### Step 3: Create Presentation

Use the visualizations from `reports/figures/` to create slides.

---

## 🎯 Expected Timeline

| Phase | Duration | Tasks |
|-------|----------|-------|
| Local Setup | 2-3 hours | Download data, preprocess, test |
| Colab Training | 6-8 hours | Train all models (can run overnight) |
| Analysis | 2-3 hours | Evaluate, create report |
| **Total** | **10-14 hours** | **Complete project** |

---

## 💡 Tips

### For Faster Training
- Use Colab Pro for better GPUs (A100)
- Train multiple models in parallel (separate notebooks)
- Start with small data fraction (10%) to verify everything works

### For Better Results
- Increase epochs to 50 for full data
- Try different learning rates (1e-3, 1e-4, 1e-5)
- Experiment with batch sizes

### Common Issues

**Issue:** Out of memory in Colab  
**Solution:** Reduce batch size (4 or 2 for TransUNet)

**Issue:** Data extraction fails  
**Solution:** Check zip file path in Drive

**Issue:** Import errors  
**Solution:** Ensure you're in project root directory

---

## 📝 Checklist

### Local Setup
- [ ] Repository cloned
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] ISIC dataset downloaded
- [ ] Data preprocessed
- [ ] Models tested
- [ ] Data zipped and uploaded to Drive
- [ ] Code pushed to GitHub

### Colab Training
- [ ] GPU runtime selected
- [ ] Repository cloned in Colab
- [ ] Dependencies installed
- [ ] Data extracted from Drive
- [ ] UNet trained (10%, 25%, 50%, 100%)
- [ ] UNet++ trained (10%, 25%, 50%, 100%)
- [ ] TransUNet trained (10%, 25%, 50%, 100%)
- [ ] All models evaluated
- [ ] Results saved to Drive

### Final Report
- [ ] Results table filled
- [ ] Visualizations added
- [ ] Analysis written
- [ ] Resume bullet created
- [ ] Report reviewed

---

## 🎓 Resume Bullet Template

After completing the project, use this for your resume:

> "Developed and compared three deep learning architectures (UNet, UNet++, TransUNet) for medical image segmentation on the ISIC skin lesion dataset, demonstrating **X% performance improvement** with Vision Transformer-based models under limited-data regimes (10-25% labeled data), validating the effectiveness of global context modeling in medical imaging applications."

Replace **X%** with your actual improvement!

---

## 📚 Additional Resources

- **UNet Paper:** https://arxiv.org/abs/1505.04597
- **UNet++ Paper:** https://arxiv.org/abs/1807.10165
- **TransUNet Paper:** https://arxiv.org/abs/2102.04306
- **ISIC Challenge:** https://challenge.isic-archive.com/

---

## 🆘 Need Help?

If you encounter issues:
1. Check the error message carefully
2. Verify all paths are correct
3. Ensure GPU is available in Colab
4. Check GitHub Issues for similar problems

---

**Good luck with your project! 🚀**
