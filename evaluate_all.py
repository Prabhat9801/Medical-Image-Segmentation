"""
Helper script to evaluate all experiments automatically.
Extracts model name and finds checkpoint from experiment directory.
"""

import os
import glob
import subprocess
import json

def evaluate_experiment(exp_dir):
    """Evaluate a single experiment"""
    exp_name = os.path.basename(exp_dir)
    
    # Extract model name from directory name
    if exp_name.startswith('unet_'):
        model = 'unet'
    elif exp_name.startswith('unetpp_'):
        model = 'unetpp'
    elif exp_name.startswith('transunet_'):
        model = 'transunet'
    else:
        print(f"⚠️  Could not determine model from: {exp_name}")
        return False
    
    # Find checkpoint file
    checkpoint_path = os.path.join(exp_dir, 'best_model.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        return False
    
    # Check if already evaluated
    test_metrics_path = os.path.join(exp_dir, 'test_metrics.json')
    if os.path.exists(test_metrics_path):
        print(f"⏭️  Already evaluated: {exp_name}")
        return True
    
    # Run evaluation
    print(f"\n{'='*60}")
    print(f"Evaluating: {exp_name}")
    print(f"Model: {model}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}\n")
    
    cmd = [
        'python', '-m', 'src.eval',
        '--model', model,
        '--checkpoint', checkpoint_path,
        '--output_dir', exp_dir,
        '--num_vis', '8'
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        print(f"\n✅ {exp_name} evaluation complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error evaluating {exp_name}:")
        print(e.stderr)
        return False

def main():
    """Evaluate all experiments"""
    exp_base_dir = "/content/Medical-Image-Segmentation/experiments"
    
    if not os.path.exists(exp_base_dir):
        print(f"❌ Experiments directory not found: {exp_base_dir}")
        return
    
    # Find all experiment directories
    exp_dirs = [
        os.path.join(exp_base_dir, d)
        for d in os.listdir(exp_base_dir)
        if os.path.isdir(os.path.join(exp_base_dir, d))
    ]
    
    if not exp_dirs:
        print("❌ No experiments found!")
        return
    
    print(f"Found {len(exp_dirs)} experiments to evaluate\n")
    
    # Evaluate each experiment
    success_count = 0
    for exp_dir in sorted(exp_dirs):
        if evaluate_experiment(exp_dir):
            success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total experiments: {len(exp_dirs)}")
    print(f"Successfully evaluated: {success_count}")
    print(f"Failed/Skipped: {len(exp_dirs) - success_count}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
