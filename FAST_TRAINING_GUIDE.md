# 🚀 Fast Training Commands for Colab

## Speed Optimizations Applied:
1. ✅ **Mixed Precision (FP16)**: ~2x faster training
2. ✅ **Reduced Workers**: 2 workers (optimal for Colab)
3. ✅ **Reduced Epochs**: 20 epochs (sufficient for convergence)
4. ✅ **Optimized Batch Sizes**: Larger batches for better GPU utilization

---

## Training Commands (Copy-Paste to Colab)

### UNet Training (Fastest - ~1.5 hours total)

```bash
# 10% data (~8 min)
!python -m src.train --model unet --epochs 20 --batch_size 24 --data_fraction 0.1 --lr 1e-4

# 25% data (~15 min)
!python -m src.train --model unet --epochs 20 --batch_size 24 --data_fraction 0.25 --lr 1e-4

# 50% data (~25 min)
!python -m src.train --model unet --epochs 20 --batch_size 24 --data_fraction 0.5 --lr 1e-4

# 100% data (~45 min)
!python -m src.train --model unet --epochs 20 --batch_size 24 --data_fraction 1.0 --lr 1e-4
```

### UNet++ Training (~1.5 hours total)

```bash
# 10% data (~8 min)
!python -m src.train --model unetpp --epochs 20 --batch_size 20 --data_fraction 0.1 --lr 1e-4

# 25% data (~15 min)
!python -m src.train --model unetpp --epochs 20 --batch_size 20 --data_fraction 0.25 --lr 1e-4

# 50% data (~25 min)
!python -m src.train --model unetpp --epochs 20 --batch_size 20 --data_fraction 0.5 --lr 1e-4

# 100% data (~45 min)
!python -m src.train --model unetpp --epochs 20 --batch_size 20 --data_fraction 1.0 --lr 1e-4
```

### TransUNet Training (~2 hours total)

```bash
# 10% data (~12 min)
!python -m src.train --model transunet --epochs 20 --batch_size 12 --data_fraction 0.1 --lr 1e-4

# 25% data (~20 min)
!python -m src.train --model transunet --epochs 20 --batch_size 12 --data_fraction 0.25 --lr 1e-4

# 50% data (~35 min)
!python -m src.train --model transunet --epochs 20 --batch_size 12 --data_fraction 0.5 --lr 1e-4

# 100% data (~60 min)
!python -m src.train --model transunet --epochs 20 --batch_size 12 --data_fraction 1.0 --lr 1e-4
```

---

## Total Training Time Estimate

| Model | Total Time | Per Fraction |
|-------|-----------|--------------|
| UNet | ~1.5 hours | 8/15/25/45 min |
| UNet++ | ~1.5 hours | 8/15/25/45 min |
| TransUNet | ~2 hours | 12/20/35/60 min |
| **TOTAL** | **~5 hours** | (vs. 10+ hours before) |

---

## Speed Improvements

### Before Optimization:
- 50 epochs × 4 fractions × 3 models = ~10-12 hours
- No mixed precision
- 4 workers (overhead in Colab)

### After Optimization:
- 20 epochs (60% reduction)
- Mixed precision FP16 (2x faster)
- 2 workers (optimal for Colab)
- Larger batch sizes (better GPU utilization)
- **Result: ~5 hours total** (50% faster!)

---

## Additional Speed Tips

### 1. Use Colab Pro/Pro+
- Guaranteed GPU access
- Longer runtime limits
- Faster GPUs (V100/A100)

### 2. Run Models Separately
- UNet: Session 1 (~1.5 hours)
- UNet++: Session 2 (~1.5 hours)
- TransUNet: Session 3 (~2 hours)

### 3. Monitor GPU Usage
```python
# Check GPU memory
!nvidia-smi
```

### 4. Early Stopping (Optional)
If validation loss plateaus before 20 epochs, you can stop early.

---

## Verification

After each model training, verify results:

```python
import os
import json

# List experiments
!ls -la experiments/

# Check latest experiment
exp_dir = "experiments/unet_100pct_XXXXXX"  # Replace with actual
with open(f"{exp_dir}/history.json") as f:
    history = json.load(f)
    print(f"Best Dice: {max(history['val_dice']):.4f}")
```

---

## 🎯 Recommended Workflow

1. **Pull latest code** (has mixed precision)
```bash
!cd /content/Medical-Image-Segmentation && git pull
```

2. **Fix splits.csv** (run once)
```python
!python fix_splits_csv.py
```

3. **Train each model** (copy commands above)

4. **Save to Drive after each model**
```bash
!cp -r experiments /content/drive/MyDrive/medical_segmentation_results/
```

This way if runtime disconnects, you don't lose progress!
