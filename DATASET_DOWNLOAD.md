# 🚀 FAST DATASET DOWNLOAD GUIDE

## Quick Download with KaggleHub (RECOMMENDED)

### Method 1: Automated Script (Fastest - 5-10 minutes)

```bash
# 1. Install kagglehub
pip install kagglehub

# 2. Run download script
python download_isic_dataset.py
```

**That's it!** The script will:
- ✅ Download ISIC2018 dataset from Kaggle (~2.5 GB)
- ✅ Automatically organize files into correct folders
- ✅ Match images with masks
- ✅ Verify everything is ready

---

## What the Download Script Does

1. **Downloads** ISIC2018 dataset using KaggleHub API
2. **Organizes** files:
   - Training images (2594) → `data/raw/isic/images/`
   - Training masks (2594) → `data/raw/isic/masks/`
   - Validation images (100) → `data/raw/isic/images/`
   - Test images (1000) → `data/raw/isic/images/` (no masks)
3. **Verifies** image-mask pairs are matched
4. **Reports** statistics

---

## Dataset Structure After Download

```
data/raw/isic/
├── images/
│   ├── ISIC_0000000.jpg
│   ├── ISIC_0000001.jpg
│   └── ... (~3,694 images total)
└── masks/
    ├── ISIC_0000000_segmentation.png
    ├── ISIC_0000001_segmentation.png
    └── ... (2,594 masks for training)
```

**Note:** Test images don't have masks (as per ISIC2018 challenge rules)

---

## Alternative Methods

### Method 2: Manual Kaggle Download

1. Go to: https://www.kaggle.com/datasets/tschandl/isic2018-challenge-task1-data-segmentation
2. Click "Download" (requires Kaggle account)
3. Extract zip file
4. Copy folders to `data/raw/isic/`

### Method 3: Direct Python Download

```python
import kagglehub

# Download dataset
path = kagglehub.dataset_download("tschandl/isic2018-challenge-task1-data-segmentation")
print("Path to dataset files:", path)
```

---

## Download Speed Comparison

| Method | Time | Difficulty |
|--------|------|------------|
| **Automated Script** | 5-10 min | ⭐ Easy |
| Manual Kaggle | 10-15 min | ⭐⭐ Medium |
| Direct API | 5-10 min | ⭐⭐⭐ Advanced |

---

## Troubleshooting

### Issue: "kagglehub not found"
**Solution:**
```bash
pip install kagglehub
```

### Issue: "Authentication required"
**Solution:**
1. Create Kaggle account: https://www.kaggle.com/
2. Go to: https://www.kaggle.com/settings/account
3. Scroll to "API" section
4. Click "Create New Token"
5. Save `kaggle.json` to:
   - Windows: `C:\Users\<username>\.kaggle\kaggle.json`
   - Linux/Mac: `~/.kaggle/kaggle.json`

### Issue: "Download is slow"
**Solutions:**
- Use wired connection instead of WiFi
- Download during off-peak hours
- Use download manager (for manual download)
- Consider using university/office high-speed connection

### Issue: "Not enough disk space"
**Solution:**
- Dataset size: ~2.5 GB compressed, ~3.5 GB extracted
- Free up at least 6 GB of space

---

## After Download

### Verify Dataset

```bash
python download_isic_dataset.py
```

Should show:
```
✅ Found 3694 images
✅ Found 2594 masks
✅ Matched pairs: 2594
```

### Next Steps

1. **Preprocess Data:**
   ```bash
   jupyter notebook notebooks/01_isic_preprocessing.ipynb
   ```

2. **Test Models:**
   ```bash
   jupyter notebook notebooks/02_model_testing.ipynb
   ```

3. **Continue with training** as per QUICKSTART.md

---

## Dataset Information

- **Source:** ISIC 2018 Challenge Task 1
- **Total Size:** ~2.5 GB
- **Images:** 3,694 dermoscopic images
- **Masks:** 2,594 binary segmentation masks
- **Format:** 
  - Images: JPG (various sizes)
  - Masks: PNG (binary, same size as images)
- **License:** CC0 Public Domain

---

## Citation

If you use this dataset, please cite:

```
Noel Codella, Veronica Rotemberg, Philipp Tschandl, M. Emre Celebi, 
Stephen Dusza, David Gutman, Brian Helba, Aadi Kalloo, Konstantinos Liopyris, 
Michael Marchetti, Harald Kittler, Allan Halpern: 
"Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by 
the International Skin Imaging Collaboration (ISIC)", 2018; 
https://arxiv.org/abs/1902.03368
```

---

## Quick Command Reference

```bash
# Download dataset
python download_isic_dataset.py

# Verify download
ls data/raw/isic/images/ | wc -l  # Should show ~3694
ls data/raw/isic/masks/ | wc -l   # Should show ~2594

# Start preprocessing
jupyter notebook notebooks/01_isic_preprocessing.ipynb
```

---

**🎉 Ready to download! Run: `python download_isic_dataset.py`**
