"""
Gradio Web App for Skin Lesion Segmentation
Deploys the trained UNet++ model (86.08% Dice Score)
"""

import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
import time

# Import model architecture
from unetpp import UNetPlusPlus

# Global variables
MODEL = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    """Load the trained UNet++ model"""
    global MODEL
    
    if MODEL is None:
        print("Loading model...")
        MODEL = UNetPlusPlus(
            in_channels=3,
            out_channels=1,
            features=32,
            deep_supervision=False
        )
        
        # Load checkpoint
        checkpoint = torch.load('best_model.pt', map_location=DEVICE)
        MODEL.load_state_dict(checkpoint['model_state_dict'])
        MODEL.to(DEVICE)
        MODEL.eval()
        print(f"Model loaded successfully! (Epoch {checkpoint['epoch']})")
    
    return MODEL

def get_transforms():
    """Get preprocessing transforms (same as training)"""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def preprocess_image(image):
    """Preprocess PIL image for model input"""
    transform = get_transforms()
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    img_tensor = transform(image).unsqueeze(0)
    
    return img_tensor

def create_overlay(original_img, mask, alpha=0.4):
    """Create colored overlay of mask on original image"""
    # Resize original to match mask
    original_resized = cv2.resize(
        np.array(original_img), 
        (mask.shape[1], mask.shape[0])
    )
    
    # Create colored mask (red for lesion)
    colored_mask = np.zeros_like(original_resized)
    colored_mask[mask > 0] = [255, 0, 0]  # Red
    
    # Blend
    overlay = cv2.addWeighted(
        original_resized, 
        1 - alpha, 
        colored_mask, 
        alpha, 
        0
    )
    
    return overlay

def calculate_metrics(mask):
    """Calculate lesion metrics from binary mask"""
    # Count lesion pixels
    lesion_pixels = np.sum(mask > 0)
    
    # Calculate area (assuming 1 pixel = 0.0264 mm² for typical dermoscopy)
    area_mm2 = lesion_pixels * 0.0264
    
    # Find contours for perimeter
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    perimeter = cv2.arcLength(contours[0], True) if contours else 0
    
    return {
        'lesion_pixels': int(lesion_pixels),
        'area_mm2': round(area_mm2, 2),
        'perimeter_pixels': round(perimeter, 2),
        'image_coverage': round((lesion_pixels / (256 * 256)) * 100, 2)
    }

def segment_lesion(image):
    """
    Main segmentation function
    
    Args:
        image: PIL Image uploaded by user
        
    Returns:
        original: Original image
        mask_img: Binary segmentation mask
        overlay_img: Colored overlay
        metrics_text: Formatted metrics string
    """
    start_time = time.time()
    
    # Load model
    model = load_model()
    
    # Preprocess
    img_tensor = preprocess_image(image).to(DEVICE)
    
    # Inference
    with torch.no_grad():
        output = model(img_tensor)
        prediction = torch.sigmoid(output)
        mask = (prediction > 0.5).float()
    
    # Convert to numpy
    mask_np = mask.squeeze().cpu().numpy()
    mask_uint8 = (mask_np * 255).astype(np.uint8)
    
    # Create visualizations
    mask_img = Image.fromarray(mask_uint8, mode='L')
    overlay_img = create_overlay(image, mask_np)
    overlay_pil = Image.fromarray(overlay_img)
    
    # Calculate metrics
    metrics = calculate_metrics(mask_np)
    inference_time = time.time() - start_time
    
    # Format metrics text
    metrics_text = f"""
    📊 **Segmentation Results**
    
    🔹 **Lesion Area:** {metrics['lesion_pixels']} pixels ({metrics['area_mm2']} mm²)
    🔹 **Perimeter:** {metrics['perimeter_pixels']} pixels
    🔹 **Image Coverage:** {metrics['image_coverage']}%
    🔹 **Processing Time:** {inference_time:.2f} seconds
    🔹 **Device:** {DEVICE}
    
    ℹ️ **Model Info:**
    - Architecture: UNet++
    - Performance: 86.08% Dice Score
    - Training Data: ISIC 2018 (100%)
    """
    
    return image, mask_img, overlay_pil, metrics_text

# Create Gradio Interface
with gr.Blocks(title="Skin Lesion Segmentation") as demo:
    
    gr.Markdown("""
    # 🏥 AI-Powered Skin Lesion Segmentation
    
    Upload a dermoscopic image to automatically segment skin lesions using deep learning.
    
    **Model:** UNet++ trained on ISIC 2018 dataset  
    **Performance:** 86.08% Dice Score | 78.31% IoU | 94.93% Accuracy
    
    ---
    """)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                type="pil", 
                label="📤 Upload Dermoscopic Image",
                height=400
            )
            
            segment_btn = gr.Button(
                "🔬 Segment Lesion", 
                variant="primary",
                size="lg"
            )
            
            gr.Markdown("""
            ### 📝 Instructions:
            1. Upload a dermoscopic skin image
            2. Click "Segment Lesion"
            3. View results and metrics
            
            ### ⚠️ Medical Disclaimer:
            This is a **research demonstration** only.  
            **NOT for medical diagnosis.**  
            Always consult a qualified dermatologist.
            """)
        
        with gr.Column():
            gr.Markdown("### 🖼️ Results")
            
            with gr.Tabs():
                with gr.Tab("Original"):
                    output_original = gr.Image(label="Original Image")
                
                with gr.Tab("Segmentation"):
                    output_mask = gr.Image(label="Binary Mask")
                
                with gr.Tab("Overlay"):
                    output_overlay = gr.Image(label="Colored Overlay")
            
            metrics_output = gr.Markdown(label="📊 Metrics")
    
    # Examples
    gr.Markdown("### 💡 Try These Examples:")
    gr.Examples(
        examples=[
            ["examples/example1.jpg"],
            ["examples/example2.jpg"],
            ["examples/example3.jpg"],
        ],
        inputs=input_image,
        label="Sample Images"
    )
    
    # Connect button to function
    segment_btn.click(
        fn=segment_lesion,
        inputs=input_image,
        outputs=[output_original, output_mask, output_overlay, metrics_output]
    )
    
    gr.Markdown("""
    ---
    
    ### 🔬 About This Model
    
    This AI model uses **UNet++**, a state-of-the-art deep learning architecture for medical image segmentation.
    
    **Training Details:**
    - Dataset: ISIC 2018 Skin Lesion Challenge
    - Training Samples: 2,594 dermoscopic images
    - Architecture: UNet++ with 9M parameters
    - Training Time: ~45 minutes on NVIDIA T4 GPU
    - Optimization: Mixed Precision (FP16)
    
    **Performance Metrics:**
    - Dice Coefficient: 86.08% ± 16.74%
    - IoU Score: 78.31% ± 19.06%
    - Pixel Accuracy: 94.93% ± 7.59%
    
    **GitHub:** [Medical-Image-Segmentation](https://github.com/Prabhat9801/Medical-Image-Segmentation)
    
    ---
    
    **⚠️ Important:** This tool is for research and educational purposes only. It should not be used for medical diagnosis or treatment decisions. Always consult with qualified healthcare professionals for medical advice.
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch()
