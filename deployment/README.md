# 🏥 Skin Lesion Segmentation - Deployment Guide

This folder contains everything needed to deploy the UNet++ model to Hugging Face Spaces.

## 📁 Files Needed

### Required Files:
- ✅ `app.py` - Gradio web application
- ✅ `unetpp.py` - Model architecture
- ✅ `requirements.txt` - Python dependencies
- ⚠️ `best_model.pt` - **YOU NEED TO DOWNLOAD THIS**
- ⚠️ `examples/` folder - **YOU NEED TO ADD EXAMPLE IMAGES**

## 📥 Download the Model

1. Go to Google Drive: https://drive.google.com/drive/folders/14-wNH4hWoinkh1I1blsrmf_f9gXcwjyr
2. Navigate to: `unetpp_experiments/unetpp_100pct_20251206_183240/`
3. Download `best_model.pt` (~35MB)
4. Place it in this `deployment/` folder

## 🖼️ Add Example Images

1. Create `examples/` folder in this directory
2. Add 3-5 sample dermoscopic images:
   - `example1.jpg`
   - `example2.jpg`
   - `example3.jpg`
3. You can get these from the ISIC dataset or use test images from your experiments

## 🧪 Test Locally

Before deploying, test the app locally:

```bash
cd deployment
python app.py
```

This will:
- Load the model
- Start Gradio server
- Open in browser at http://localhost:7860

Test by uploading an image and checking if segmentation works!

## 🚀 Deploy to Hugging Face Spaces

### Step 1: Create Account
1. Go to https://huggingface.co/
2. Sign up for free account

### Step 2: Create New Space
1. Click "New" → "Space"
2. Name: `skin-lesion-segmentation`
3. License: MIT
4. SDK: **Gradio**
5. Hardware: **CPU basic** (free)
6. Click "Create Space"

### Step 3: Upload Files
Upload these files to your Space:
- `app.py`
- `unetpp.py`
- `requirements.txt`
- `best_model.pt`
- `examples/` folder with images

### Step 4: Wait for Build
- Hugging Face will automatically install dependencies
- Build takes ~5-10 minutes
- Watch the logs for any errors

### Step 5: Test & Share!
- Your app will be live at: `https://huggingface.co/spaces/YOUR_USERNAME/skin-lesion-segmentation`
- Test it with different images
- Share the link!

## 🎨 Customization

### Change UI Theme:
In `app.py`, line 132:
```python
with gr.Blocks(theme=gr.themes.Soft(), ...):
```

Try: `gr.themes.Base()`, `gr.themes.Monochrome()`, `gr.themes.Glass()`

### Add More Metrics:
In `calculate_metrics()` function, add:
- Circularity
- Asymmetry index
- Border irregularity

### Change Colors:
In `create_overlay()`, line 67:
```python
colored_mask[mask > 0] = [255, 0, 0]  # Red
```

Try: `[0, 255, 0]` (Green), `[255, 255, 0]` (Yellow)

## ⚡ Performance Tips

### For Faster Inference:
1. **Use GPU** (if available on Hugging Face):
   - Upgrade to GPU hardware in Space settings
   - Free tier has limited GPU hours

2. **Model Quantization**:
   ```python
   model = torch.quantization.quantize_dynamic(
       model, {torch.nn.Linear}, dtype=torch.qint8
   )
   ```

3. **Reduce Image Size**:
   - Change resize from 256x256 to 128x128
   - Faster but slightly less accurate

## 🐛 Troubleshooting

### "Model file not found"
- Make sure `best_model.pt` is in the same folder as `app.py`
- Check file name is exactly `best_model.pt`

### "CUDA out of memory"
- You're on CPU-only tier (this is normal)
- Model will use CPU automatically
- Inference takes 2-3 seconds instead of <1 second

### "Module not found"
- Check `requirements.txt` has all dependencies
- Hugging Face will auto-install on build

### "Image upload fails"
- Check image format (JPG, PNG supported)
- Max size: 10MB
- Try resizing large images

## 📊 Expected Performance

| Hardware | Inference Time | Cost |
|----------|---------------|------|
| CPU (free) | 2-3 seconds | FREE |
| GPU (T4) | <1 second | Limited free hours |

## 🎯 Next Steps

After deployment:
1. ✅ Test with various images
2. ✅ Share link with friends/colleagues
3. ✅ Add to your portfolio/resume
4. ✅ Include in GitHub README
5. ✅ Get feedback and improve!

## 📝 Notes

- **Medical Disclaimer**: Always included in the app
- **Privacy**: Images are not stored, processed in memory only
- **Updates**: Push new files to Space to update
- **Monitoring**: Check Space logs for errors

## 🔗 Useful Links

- **Hugging Face Spaces Docs**: https://huggingface.co/docs/hub/spaces
- **Gradio Docs**: https://www.gradio.app/docs
- **Your GitHub Repo**: https://github.com/Prabhat9801/Medical-Image-Segmentation

---

**Ready to deploy?** Follow the steps above and you'll have a live demo in ~1 hour! 🚀
