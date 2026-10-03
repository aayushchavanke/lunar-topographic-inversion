"""
Compare results from baseline, illumination-aware, and full multi-task models

Generates:
- Comparative metrics table
- Confusion matrices
- F1 vs. solar incidence angle plots
- Sample predictions visualization
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from evaluation.metrics import IlluminationStratifiedMetrics
from evaluation.validate_lola import validate_against_lola

def load_experiment_results(exp_dir):
    """Load metrics and predictions from a trained experiment"""
    exp_path = Path(exp_dir)

    # Load config
    with open(exp_path / "config.yaml", 'r') as f:
        import yaml
        config = yaml.safe_load(f)

    # Load best checkpoint metrics
    checkpoint = torch.load(exp_path / "checkpoints" / "best.pt")
    metrics = checkpoint['metrics']

    return {
        'name': config['experiment']['name'],
        'config': config,
        'metrics': metrics,
        'checkpoint': checkpoint
    }

def create_comparison_table(experiments):
    """Create LaTeX table comparing all models"""

    table_data = []

    for exp in experiments:
        name = exp['name']
        metrics = exp['metrics']

        row = {
            'Model': name.replace('_', ' ').title(),
            'mAP50': f"{metrics['map50']:.4f}",
            'mAP50-95': f"{metrics.get('map50_95', 0):.4f}",
            'Precision': f"{metrics.get('precision', 0):.4f}",
            'Recall': f"{metrics.get('recall', 0):.4f}",
            'F1 (Low Sun)': f"{metrics.get('f1_low_sun', 0):.4f}",
            'F1 (Mid Sun)': f"{metrics.get('f1_mid_sun', 0):.4f}",
            'F1 (High Sun)': f"{metrics.get('f1_high_sun', 0):.4f}",
            'Morphology Acc': f"{metrics.get('morphology_accuracy', 'N/A'):.4f}" if 'morphology_accuracy' in metrics else 'N/A'
        }

        table_data.append(row)

    df = pd.DataFrame(table_data)

    # Save as CSV
    df.to_csv('results/tables/model_comparison.csv', index=False)

    # Generate LaTeX table
    latex_table = df.to_latex(index=False, float_format="%.4f")
    with open('results/tables/model_comparison.tex', 'w') as f:
        f.write(latex_table)

    print("\n" + "="*60)
    print("MODEL COMPARISON TABLE")
    print("="*60)
    print(df.to_string(index=False))
    print("\nSaved to:")
    print("  - results/tables/model_comparison.csv")
    print("  - results/tables/model_comparison.tex")

    return df

def plot_f1_vs_solar_angle(experiments):
    """Plot F1-score vs. solar incidence angle for all models"""

    plt.figure(figsize=(12, 8))

    colors = ['red', 'blue', 'green']
    markers = ['o', 's', '^']

    for i, exp in enumerate(experiments):
        name = exp['name']
        metrics = exp['metrics']

        # Extract F1 scores by solar angle bin
        f1_scores = [
            metrics.get('f1_low_sun', 0),
            metrics.get('f1_mid_sun', 0),
            metrics.get('f1_high_sun', 0)
        ]

        solar_angles = ['Low (0-30°)', 'Mid (30-60°)', 'High (60-90°)']

        plt.plot(solar_angles, f1_scores, 
                marker=markers[i], 
                markersize=10, 
                linewidth=2,
                color=colors[i],
                label=name.replace('_', ' ').title())

    plt.xlabel('Solar Incidence Angle', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    plt.title('Model Performance vs. Solar Illumination', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig('results/figures/f1_vs_solar_angle.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved F1 vs. solar angle plot: results/figures/f1_vs_solar_angle.png")

def plot_training_curves(experiments):
    """Plot training loss and mAP curves for all models"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = ['red', 'blue', 'green']

    for i, exp in enumerate(experiments):
        name = exp['name']
        checkpoint = exp['checkpoint']

        # Extract training history (if available)
        if 'training_history' in checkpoint:
            history = checkpoint['training_history']
            epochs = list(range(len(history['train_loss'])))

            # Loss curve
            ax1.plot(epochs, history['train_loss'], 
                    color=colors[i], 
                    linewidth=2,
                    label=f"{name.replace('_', ' ').title()} (Train)")

            # mAP curve
            ax2.plot(epochs, history['val_map50'], 
                    color=colors[i], 
                    linewidth=2,
                    label=f"{name.replace('_', ' ').title()}")

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss Curves', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation mAP50', fontsize=12)
    ax2.set_title('Validation mAP50 Curves', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/figures/training_curves.png', dpi=300, bbox_inches='tight')
    print("✅ Saved training curves: results/figures/training_curves.png")

def create_improvement_summary(experiments):
    """Create summary of improvements from baseline"""

    if len(experiments) < 2:
        print("\n⚠️  Need at least 2 experiments to show improvements")
        return

    baseline = experiments[0]

    print("\n" + "="*60)
    print("IMPROVEMENT SUMMARY (vs. Baseline)")
    print("="*60)

    for i, exp in enumerate(experiments[1:], 1):
        name = exp['name']
        baseline_metrics = baseline['metrics']
        exp_metrics = exp['metrics']

        print(f"\n{name.replace('_', ' ').title()}:")

        # Calculate improvements
        metrics_to_compare = ['map50', 'precision', 'recall', 'f1_low_sun']

        for metric in metrics_to_compare:
            if metric in baseline_metrics and metric in exp_metrics:
                baseline_val = baseline_metrics[metric]
                exp_val = exp_metrics[metric]
                improvement = ((exp_val - baseline_val) / baseline_val) * 100

                print(f"  {metric}: {baseline_val:.4f} → {exp_val:.4f} ({improvement:+.1f}%)")

    print("\n" + "="*60)

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare trained models")
    parser.add_argument("--experiments", type=str, nargs='+', required=True,
                       help="Paths to experiment directories")
    parser.add_argument("--output", type=str, default="results/comparison",
                       help="Output directory for comparison results")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("MODEL COMPARISON ANALYSIS")
    print(f"{'='*60}")
    print(f"Comparing {len(args.experiments)} experiments:")
    for exp_path in args.experiments:
        print(f"  - {exp_path}")
    print(f"{'='*60}\n")

    # Load all experiments
    experiments = []
    for exp_path in args.experiments:
        print(f"Loading {exp_path}...")
        exp_data = load_experiment_results(exp_path)
        experiments.append(exp_data)
        print(f"  ✅ Loaded: {exp_data['name']}")

    # Generate comparison table
    print("\nGenerating comparison table...")
    comparison_df = create_comparison_table(experiments)

    # Generate plots
    print("\nGenerating F1 vs. solar angle plot...")
    plot_f1_vs_solar_angle(experiments)

    print("\nGenerating training curves...")
    plot_training_curves(experiments)

    # Generate improvement summary
    create_improvement_summary(experiments)

    print(f"\n{'='*60}")
    print("COMPARISON COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir.absolute()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
