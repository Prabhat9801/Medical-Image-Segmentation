# 🏥 Google Colab Training Notebooks

This directory contains separate Colab notebooks for training each model independently to avoid runtime timeouts.

## 📚 Notebooks Overview

### Training Notebooks (Run in Order)

1. **`COLAB_TRAIN_UNET.ipynb`** 
   - Trains UNet on 10%, 25%, 50%, 100% data
   - Runtime: ~2-3 hours
   - Saves results to: `unet_experiments/`

2. **`COLAB_TRAIN_UNETPP.ipynb`**
   - Trains UNet++ on 10%, 25%, 50%, 100% data
   - Runtime: ~2-3 hours
   - Saves results to: `unetpp_experiments/`

3. **`COLAB_TRAIN_TRANSUNET.ipynb`**
   - Trains TransUNet on 10%, 25%, 50%, 100% data
   - Runtime: ~3-4 hours (slower due to transformer)
   - Saves results to: `transunet_experiments/`

### Results Notebook (Run After All Training)

4. **`COLAB_RESULTS.ipynb`**
   - Evaluates all trained models
   - Generates comparison plots
   - Creates results summary CSV
   - Saves everything to Google Drive

## 🚀 Quick Start

### Prerequisites

1. **Upload Data to Google Drive:**
   - Upload `isic_processed_256.zip` to your Google Drive root (`My Drive/`)

2. **GPU Runtime:**
   - Each notebook requires GPU runtime
   - Go to: **Runtime → Change runtime type → GPU**

### Training Workflow

```
Step 1: Run COLAB_TRAIN_UNET.ipynb
   ↓
Step 2: Run COLAB_TRAIN_UNETPP.ipynb
   ↓
Step 3: Run COLAB_TRAIN_TRANSUNET.ipynb
   ↓
Step 4: Run COLAB_RESULTS.ipynb
```

### Running Each Notebook

1. **Open in Colab:**
   - Go to [Google Colab](https://colab.research.google.com/)
   - Upload the notebook or connect to GitHub

2. **Set GPU Runtime:**
   - Runtime → Change runtime type → GPU → Save

3. **Run All Cells:**
   - Runtime → Run all
   - Or run cells sequentially (Shift + Enter)

4. **Monitor Progress:**
   - Training progress bars will show
   - Results saved automatically to Google Drive

## 📊 Expected Runtimes

| Notebook | Data Fractions | Expected Time |
|----------|---------------|---------------|
| UNet | 10%, 25%, 50%, 100% | ~2-3 hours |
| UNet++ | 10%, 25%, 50%, 100% | ~2-3 hours |
| TransUNet | 10%, 25%, 50%, 100% | ~3-4 hours |
| Results | Evaluation only | ~30 minutes |

**Total Training Time: ~8-10 hours**

## 💾 Results Storage

All results are automatically saved to Google Drive:

```
My Drive/
└── medical_segmentation_results/
    ├── unet_experiments/
    ├── unetpp_experiments/
    ├── transunet_experiments/
    └── final_results/
        ├── results_summary.csv
        ├── results_comparison.png
        └── experiments/
```

## ⚠️ Important Notes

1. **Runtime Limits:**
   - Free Colab has ~12 hour runtime limit
   - Run one model per session to avoid timeouts
   - Results are saved to Drive, so you can resume

2. **GPU Availability:**
   - Free Colab GPU may not always be available
   - Consider Colab Pro for guaranteed GPU access

3. **Data Extraction:**
   - Each notebook extracts data independently
   - This ensures clean state for each training run

4. **Memory Management:**
   - Notebooks are optimized for Colab's memory limits
   - TransUNet uses smaller batch size (8 vs 16)

## 🔧 Troubleshooting

### "Runtime disconnected"
- Results are saved to Google Drive
- Re-run the notebook from where it stopped
- Check Drive storage space

### "Out of memory"
- Reduce batch size in training command
- Use smaller data fraction first
- Restart runtime and try again

### "File not found"
- Verify `isic_processed_256.zip` is in Google Drive
- Check Drive is mounted correctly
- Re-run data extraction cell

## 📝 After Training

1. **Download Results:**
   - Download from `My Drive/medical_segmentation_results/`

2. **Local Analysis:**
   - Copy results to local `experiments/` directory
   - Update `reports/report.md` with findings
   - Create visualizations

3. **Documentation:**
   - Update README with final results
   - Add best model configurations
   - Document key findings

## 🎯 Next Steps

After completing all training:

1. ✅ Run `COLAB_RESULTS.ipynb` for comprehensive evaluation
2. ✅ Download results from Google Drive
3. ✅ Analyze performance across models and data fractions
4. ✅ Update project documentation
5. ✅ Create final presentation

---

**Happy Training! 🚀**
