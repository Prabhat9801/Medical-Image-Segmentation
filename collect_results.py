"""
Collect all evaluation results and create summary
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# Find all result files
results_dir = "/content/Medical-Image-Segmentation/experiments"
result_files = glob.glob(f"{results_dir}/*/best_model/results.json")

print(f"Found {len(result_files)} result files\n")

# Collect all results
results = []

for result_file in sorted(result_files):
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    # Extract experiment info from path
    exp_dir = os.path.dirname(os.path.dirname(result_file))
    exp_name = os.path.basename(exp_dir)
    
    # Parse experiment name (e.g., "unet_100pct_20251206_181019")
    parts = exp_name.split('_')
    model = parts[0]
    data_frac = parts[1] if len(parts) > 1 else "unknown"
    
    # Extract metrics
    metrics = data.get('metrics', {})
    
    results.append({
        'Model': model,
        'Data Fraction': data_frac,
        'Dice Score': metrics.get('dice', {}).get('mean', 0),
        'Dice Std': metrics.get('dice', {}).get('std', 0),
        'IoU': metrics.get('iou', {}).get('mean', 0),
        'IoU Std': metrics.get('iou', {}).get('std', 0),
        'Pixel Accuracy': metrics.get('accuracy', {}).get('mean', 0),
        'Accuracy Std': metrics.get('accuracy', {}).get('std', 0),
        'Experiment': exp_name
    })

# Create DataFrame
df_results = pd.DataFrame(results)

# Sort by model and data fraction
df_results = df_results.sort_values(['Model', 'Data Fraction'])

# Display results
print("="*80)
print("📊 COMPLETE RESULTS SUMMARY")
print("="*80)
print(df_results[['Model', 'Data Fraction', 'Dice Score', 'IoU', 'Pixel Accuracy']].to_string(index=False))
print("\n")

# Save to CSV
results_csv = "/content/Medical-Image-Segmentation/results_summary.csv"
df_results.to_csv(results_csv, index=False)
print(f"✅ Results saved to: {results_csv}\n")

# Create comparison plots
print("Creating comparison plots...")

sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Dice Score by Model and Data Fraction
pivot_dice = df_results.pivot(index='Data Fraction', columns='Model', values='Dice Score')
pivot_dice.plot(kind='bar', ax=axes[0, 0], rot=45)
axes[0, 0].set_title('Dice Score Comparison', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Dice Score')
axes[0, 0].set_xlabel('Data Fraction')
axes[0, 0].legend(title='Model')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: IoU by Model and Data Fraction
pivot_iou = df_results.pivot(index='Data Fraction', columns='Model', values='IoU')
pivot_iou.plot(kind='bar', ax=axes[0, 1], rot=45)
axes[0, 1].set_title('IoU Score Comparison', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('IoU Score')
axes[0, 1].set_xlabel('Data Fraction')
axes[0, 1].legend(title='Model')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Pixel Accuracy by Model and Data Fraction
pivot_acc = df_results.pivot(index='Data Fraction', columns='Model', values='Pixel Accuracy')
pivot_acc.plot(kind='bar', ax=axes[1, 0], rot=45)
axes[1, 0].set_title('Pixel Accuracy Comparison', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Pixel Accuracy')
axes[1, 0].set_xlabel('Data Fraction')
axes[1, 0].legend(title='Model')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Overall Performance Heatmap
heatmap_data = df_results.pivot_table(
    index='Model', 
    columns='Data Fraction', 
    values='Dice Score'
)
sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='YlGnBu', ax=axes[1, 1], cbar_kws={'label': 'Dice Score'})
axes[1, 1].set_title('Dice Score Heatmap', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Data Fraction')
axes[1, 1].set_ylabel('Model')

plt.tight_layout()
plt.savefig('/content/Medical-Image-Segmentation/results_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Comparison plots saved to: results_comparison.png\n")

# Find best models
print("="*80)
print("🏆 BEST MODEL OVERALL")
print("="*80)
best_overall = df_results.loc[df_results['Dice Score'].idxmax()]
print(f"Model: {best_overall['Model']}")
print(f"Data Fraction: {best_overall['Data Fraction']}")
print(f"Dice Score: {best_overall['Dice Score']:.4f} ± {best_overall['Dice Std']:.4f}")
print(f"IoU: {best_overall['IoU']:.4f} ± {best_overall['IoU Std']:.4f}")
print(f"Pixel Accuracy: {best_overall['Pixel Accuracy']:.4f} ± {best_overall['Accuracy Std']:.4f}")
print(f"Experiment: {best_overall['Experiment']}")

print("\n" + "="*80)
print("🏆 BEST MODEL PER DATA FRACTION")
print("="*80)

for frac in sorted(df_results['Data Fraction'].unique()):
    frac_data = df_results[df_results['Data Fraction'] == frac]
    best_frac = frac_data.loc[frac_data['Dice Score'].idxmax()]
    print(f"\n{frac}:")
    print(f"  Model: {best_frac['Model']}")
    print(f"  Dice Score: {best_frac['Dice Score']:.4f} ± {best_frac['Dice Std']:.4f}")
    print(f"  IoU: {best_frac['IoU']:.4f}")
    print(f"  Pixel Accuracy: {best_frac['Pixel Accuracy']:.4f}")

print("\n" + "="*80)
print("✅ Results analysis complete!")
print("="*80)
