"""
Visualization module for model performance using matplotlib.
"""

from typing import List
import math
import matplotlib.pyplot as plt


def plot_predictions_vs_actual(y_true: List[float], y_pred: List[float], 
                                title: str = "Predictions vs Actual"):
    """
    Create a simple scatter plot of predictions vs actual values.
    
    Perfect predictions fall on the diagonal line (y = x).
    Points above the line = model overestimated win chance
    Points below the line = model underestimated win chance
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        title: Plot title
    """
    plt.figure(figsize=(10, 8))
    
    # Simple scatter plot
    plt.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors='none', color='steelblue')
    
    # Perfect prediction line (y = x)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
             label='Perfect Prediction', alpha=0.7)
    
    # Labels and formatting
    plt.xlabel('Actual Win Chance (%)', fontsize=12)
    plt.ylabel('Predicted Win Chance (%)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add R² to plot
    mean_y = sum(y_true) / len(y_true)
    ss_total = sum((yt - mean_y) ** 2 for yt in y_true)
    ss_residual = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    r2 = 1.0 - (ss_residual / ss_total) if ss_total != 0 else 0.0
    
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', 
             transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('predictions_vs_actual.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: predictions_vs_actual.png")
    plt.close()


def plot_predictions_vs_labels_colored(y_true: List[float], y_pred: List[float],
                                        title: str = "Predictions vs Labels (Error-Coded)"):
    """
    NEW FUNCTION: Color-coded scatter plot showing prediction quality.
    
    Color coding by error magnitude:
    - Blue = accurate predictions (error < 10%)
    - Orange = moderate error (10-20%)
    - Red = large error (> 20%)
    
    Args:
        y_true: Actual values (labels - ground truth)
        y_pred: Predicted values (model predictions)
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    
    # Calculate absolute errors
    errors = [abs(pred - true) for true, pred in zip(y_true, y_pred)]
    
    # Assign colors based on error magnitude
    colors = []
    for error in errors:
        if error < 10:
            colors.append('blue')
        elif error < 20:
            colors.append('orange')
        else:
            colors.append('red')
    
    # Scatter plot with error-based coloring
    plt.scatter(y_true, y_pred, c=colors, alpha=0.6, s=25, edgecolors='none')
    
    # Perfect prediction line (y = x)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2.5, 
             label='Perfect Prediction', alpha=0.8)
    
    # Labels and formatting
    plt.xlabel('Actual Win Chance (%) — LABELS (Ground Truth)', fontsize=13, fontweight='bold')
    plt.ylabel('Predicted Win Chance (%) — PREDICTIONS (Model Output)', fontsize=13, fontweight='bold')
    plt.title(title, fontsize=15, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    
    # Calculate R² and error statistics
    mean_y = sum(y_true) / len(y_true)
    ss_total = sum((yt - mean_y) ** 2 for yt in y_true)
    ss_residual = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    r2 = 1.0 - (ss_residual / ss_total) if ss_total != 0 else 0.0
    
    # Count points in each error category
    n_accurate = sum(1 for e in errors if e < 10)
    n_moderate = sum(1 for e in errors if 10 <= e < 20)
    n_poor = sum(1 for e in errors if e >= 20)
    total = len(errors)
    
    # Statistics text box
    stats_text = f'R² = {r2:.4f}\n\n'
    stats_text += f'Error Distribution:\n'
    stats_text += f'──────────────────\n'
    stats_text += f'Good (<10%):      {n_accurate:>4} ({100*n_accurate/total:>5.1f}%)\n'
    stats_text += f'Medium (10-20%):  {n_moderate:>4} ({100*n_moderate/total:>5.1f}%)\n'
    stats_text += f'Poor (>20%):      {n_poor:>4} ({100*n_poor/total:>5.1f}%)\n'
    stats_text += f'──────────────────\n'
    stats_text += f'Total:            {total:>4}'
    
    plt.text(0.03, 0.97, stats_text, 
             transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='black'),
             family='monospace')
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        plt.Line2D([0], [0], color='k', linestyle='--', linewidth=2.5, label='Perfect Prediction (y=x)'),
        Patch(facecolor='blue', alpha=0.6, label='🔵 Good: error < 10%'),
        Patch(facecolor='orange', alpha=0.6, label='🟠 Medium: 10-20% error'),
        Patch(facecolor='red', alpha=0.6, label='🔴 Poor: > 20% error')
    ]
    plt.legend(handles=legend_elements, fontsize=11, loc='lower right', framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig('predictions_vs_labels_colored.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: predictions_vs_labels_colored.png")
    print(f"  ├─ Good predictions (< 10% error):  {n_accurate:>4}/{total} ({100*n_accurate/total:.1f}%)")
    print(f"  ├─ Medium predictions (10-20%):     {n_moderate:>4}/{total} ({100*n_moderate/total:.1f}%)")
    print(f"  └─ Poor predictions (> 20% error):  {n_poor:>4}/{total} ({100*n_poor/total:.1f}%)")
    plt.close()


def plot_residuals_histogram(y_true: List[float], y_pred: List[float],
                              title: str = "Prediction Errors Distribution"):
    """
    Create a histogram of prediction errors (residuals).
    
    Residual = predicted - actual
    Positive residual = overestimated
    Negative residual = underestimated
    
    A good model should have errors centered around 0.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        title: Plot title
    """
    # Calculate residuals
    residuals = [pred - true for true, pred in zip(y_true, y_pred)]
    
    # Calculate statistics
    mean_error = sum(residuals) / len(residuals)
    variance = sum((r - mean_error) ** 2 for r in residuals) / len(residuals)
    std_error = math.sqrt(variance)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Histogram
    plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    
    # Mean line
    plt.axvline(mean_error, color='red', linestyle='--', linewidth=2, 
                label=f'Mean Error = {mean_error:.2f}%')
    
    # Zero line
    plt.axvline(0, color='green', linestyle='-', linewidth=2, alpha=0.5,
                label='Zero Error')
    
    # Labels
    plt.xlabel('Prediction Error (Predicted - Actual) %', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Statistics box
    stats_text = f'Mean: {mean_error:.2f}%\nStd: {std_error:.2f}%\nMin: {min(residuals):.2f}%\nMax: {max(residuals):.2f}%'
    plt.text(0.02, 0.98, stats_text, 
             transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('residuals_histogram.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: residuals_histogram.png")
    plt.close()


def plot_feature_importance(feature_names: List[str], weights: List[float],
                             top_n: int = 15,
                             title: str = "Feature Importance"):
    """
    Create a bar chart of feature importance.
    
    Args:
        feature_names: Names of features
        weights: Model weights for each feature
        top_n: Number of top features to show
        title: Plot title
    """
    # Sort by absolute weight
    importance = list(zip(feature_names, weights))
    importance.sort(key=lambda x: abs(x[1]), reverse=True)
    importance = importance[:top_n]
    
    # Unzip
    names, vals = zip(*importance)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Color bars based on sign
    colors = ['green' if w > 0 else 'red' for w in vals]
    
    # Horizontal bar chart
    y_pos = range(len(names))
    plt.barh(y_pos, vals, color=colors, alpha=0.7, edgecolor='black')
    
    # Customize
    plt.yticks(y_pos, names, fontsize=10)
    plt.xlabel('Weight (Impact on Win Chance)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axvline(0, color='black', linewidth=0.8)
    plt.grid(True, alpha=0.3, axis='x')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Positive (helps win)'),
        Patch(facecolor='red', alpha=0.7, label='Negative (hurts win)')
    ]
    plt.legend(handles=legend_elements, fontsize=10, loc='best')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: feature_importance.png")
    plt.close()


def create_all_visualizations(y_train: List[float], y_pred_train: List[float],
                               y_test: List[float], y_pred_test: List[float],
                               feature_names: List[str], weights: List[float]):
    """
    Create all visualizations and save them as PNG files.
    
    Args:
        y_train: Training actual values
        y_pred_train: Training predictions
        y_test: Test actual values
        y_pred_test: Test predictions
        feature_names: List of feature names
        weights: Model weights
    """
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    # 1. Simple Predictions vs Actual
    plot_predictions_vs_actual(
        y_test, 
        y_pred_test,
        title="Test Set: Predictions vs Actual"
    )
    
    # 2. NEW: Color-coded Predictions vs Labels
    plot_predictions_vs_labels_colored(
        y_test,
        y_pred_test,
        title="Test Set: Predictions vs Labels (Error-Coded)"
    )
    
    # 3. Residuals
    plot_residuals_histogram(
        y_test,
        y_pred_test,
        title="Test Set: Prediction Error Distribution"
    )
    
    # 4. Feature Importance
    plot_feature_importance(
        feature_names,
        weights,
        top_n=15,
        title="Top 15 Most Important Features"
    )
    
    print("\n" + "=" * 70)
    print("All visualizations saved!")
    print("  • predictions_vs_actual.png (simple blue dots)")
    print("  • predictions_vs_labels_colored.png (NEW: color-coded by error)")
    print("  • residuals_histogram.png")
    print("  • feature_importance.png")
    print("=" * 70)