"""
Neural Network Architecture Comparison

Trains two different neural network architectures and compares their performance:
- Standard: 34 → 64 → 32 → 1  (~4,353 parameters)
- Conservative: 34 → 48 → 24 → 1  (~2,881 parameters)
"""

from CSVParser import CSVParser
from BattlelPreprocessor import BattlePreprocessor
from data_utils import train_test_split, StandardScaler
from neural_network import (
    NeuralNetwork,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r_squared
)
from visualizations import create_all_visualizations
from battlePredictor import BattlePredictor

def train_and_evaluate_model(model_name: str, hidden_sizes: list[int],
                              X_train_scaled, y_train, X_test_scaled, y_test):
    """Train a single model and return its performance metrics."""
    print(f"\n{'=' * 70}")
    print(f"TRAINING: {model_name}")
    print(f"{'=' * 70}")
    
    # Create model
    model = NeuralNetwork(
        input_size=len(X_train_scaled[0]),
        hidden_sizes=hidden_sizes,
        learning_rate=0.001,
        epochs=300,
        batch_size=64,
        regularization=0.001
    )
    
    print(f"\nArchitecture: {model.get_architecture_summary()}")
    print(f"\nTraining for {model.epochs} epochs...")
    
    # Train with validation monitoring
    model.fit(X_train_scaled, y_train, 
              X_val=X_test_scaled, y_val=y_test,
              verbose=True)
    
    # Make predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Calculate metrics
    metrics = {
        'train_mse': mean_squared_error(y_train, y_pred_train),
        'test_mse': mean_squared_error(y_test, y_pred_test),
        'train_rmse': root_mean_squared_error(y_train, y_pred_train),
        'test_rmse': root_mean_squared_error(y_test, y_pred_test),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'train_r2': r_squared(y_train, y_pred_train),
        'test_r2': r_squared(y_test, y_pred_test)
    }
    
    # Print results
    print(f"\n{'-' * 70}")
    print(f"RESULTS: {model_name}")
    print(f"{'-' * 70}")
    print(f"{'Metric':<10} {'Train':>12} {'Test':>12}")
    print(f"{'-' * 70}")
    print(f"{'MSE':<10} {metrics['train_mse']:>12.2f} {metrics['test_mse']:>12.2f}")
    print(f"{'RMSE':<10} {metrics['train_rmse']:>12.2f} {metrics['test_rmse']:>12.2f}")
    print(f"{'MAE':<10} {metrics['train_mae']:>12.2f} {metrics['test_mae']:>12.2f}")
    print(f"{'R²':<10} {metrics['train_r2']:>12.4f} {metrics['test_r2']:>12.4f}")
    print(f"{'-' * 70}")
    
    return model, metrics, y_pred_train, y_pred_test


def main():
    from typing import List
    
    print("=" * 70)
    print("NEURAL NETWORK ARCHITECTURE COMPARISON")
    print("=" * 70)
    
    # ═════════════════════════════════════════════════════════════════
    # Load and preprocess data (same as before)
    # ═════════════════════════════════════════════════════════════════
    print("\n[1/5] Loading data...")
    parser = CSVParser("battle_dataset_10000.csv")
    
    print("[2/5] Preprocessing...")
    preprocessor = BattlePreprocessor(parser.fetch_all())
    X, y = preprocessor.preprocess()
    
    print("[3/5] Splitting (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    print("[4/5] Scaling...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n✓ Data ready: {len(X_train)} train, {len(X_test)} test, {len(X[0])} features")
    
    # ═════════════════════════════════════════════════════════════════
    # Train both models
    # ═════════════════════════════════════════════════════════════════
    print(f"\n[5/5] Training both architectures...")
    
    # Model 1: Standard architecture
    model1, metrics1, pred_train1, pred_test1 = train_and_evaluate_model(
        "STANDARD (64 → 32)",
        hidden_sizes=[64, 32],
        X_train_scaled=X_train_scaled,
        y_train=y_train,
        X_test_scaled=X_test_scaled,
        y_test=y_test
    )
    
    # Model 2: Conservative architecture
    model2, metrics2, pred_train2, pred_test2 = train_and_evaluate_model(
        "CONSERVATIVE (48 → 24)",
        hidden_sizes=[48, 24],
        X_train_scaled=X_train_scaled,
        y_train=y_train,
        X_test_scaled=X_test_scaled,
        y_test=y_test
    )
    
    # ═════════════════════════════════════════════════════════════════
    # Side-by-side comparison
    # ═════════════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 70}")
    print("FINAL COMPARISON")
    print(f"{'=' * 70}")
    
    print(f"\n{'Architecture':<30} {'Standard (64→32)':<20} {'Conservative (48→24)':<20}")
    print(f"{'-' * 70}")
    print(f"{'Parameters':<30} {model1.count_parameters():<20,} {model2.count_parameters():<20,}")
    print(f"{'Samples/Parameter':<30} {len(X_train)/model1.count_parameters():<20.2f} {len(X_train)/model2.count_parameters():<20.2f}")
    print(f"{'-' * 70}")
    print(f"{'Test RMSE':<30} {metrics1['test_rmse']:<20.2f} {metrics2['test_rmse']:<20.2f}")
    print(f"{'Test MAE':<30} {metrics1['test_mae']:<20.2f} {metrics2['test_mae']:<20.2f}")
    print(f"{'Test R²':<30} {metrics1['test_r2']:<20.4f} {metrics2['test_r2']:<20.4f}")
    print(f"{'-' * 70}")
    
    # Determine winner
    if metrics1['test_r2'] > metrics2['test_r2']:
        winner = "Standard (64→32)"
        winner_margin = (metrics1['test_r2'] - metrics2['test_r2']) * 100
    else:
        winner = "Conservative (48→24)"
        winner_margin = (metrics2['test_r2'] - metrics1['test_r2']) * 100
    
    print(f"\n🏆 WINNER: {winner}")
    print(f"   Margin: {winner_margin:.2f} percentage points higher R²")
    
    # Determine which model to use for visualizations
    if metrics1['test_r2'] >= metrics2['test_r2']:
        best_model = model1
        best_preds_train = pred_train1
        best_preds_test = pred_test1
        best_name = "Standard (64→32)"
    else:
        best_model = model2
        best_preds_train = pred_train2
        best_preds_test = pred_test2
        best_name = "Conservative (48→24)"
    
    print(f"\n✓ Using {best_name} for visualizations")
    
    # ═════════════════════════════════════════════════════════════════
    # Generate visualizations for best model
    # ═════════════════════════════════════════════════════════════════
    # Neural networks don't have simple "weights" like linear models
    # So we'll skip feature importance for now
    print("\nGenerating visualizations for best model...")
    
    from visualizations import (
        plot_predictions_vs_actual,
        plot_predictions_vs_labels_colored,
        plot_residuals_histogram
    )
    
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    plot_predictions_vs_actual(
        y_test,
        best_preds_test,
        title=f"Neural Network ({best_name}): Predictions vs Actual"
    )
    
    plot_predictions_vs_labels_colored(
        y_test,
        best_preds_test,
        title=f"Neural Network ({best_name}): Error-Coded"
    )
    
    plot_residuals_histogram(
        y_test,
        best_preds_test,
        title=f"Neural Network ({best_name}): Error Distribution"
    )
    
    print("\n" + "=" * 70)
    print("All visualizations saved!")
    print("  • predictions_vs_actual.png")
    print("  • predictions_vs_labels_colored.png")
    print("  • residuals_histogram.png")
    print("=" * 70)
    
    # ═════════════════════════════════════════════════════════════════
    # Summary
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"✓ Trained 2 neural network architectures")
    print(f"✓ Best model: {best_name}")
    print(f"✓ Test R²: {max(metrics1['test_r2'], metrics2['test_r2']):.4f}")
    print(f"✓ Test RMSE: {min(metrics1['test_rmse'], metrics2['test_rmse']):.2f}%")
    print(f"✓ Predictions guaranteed in 0-100% range (sigmoid output)")
    print(f"{'=' * 70}")
    
    # ═════════════════════════════════════════════════════════════════
    # Save the best model
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("SAVING BEST MODEL")
    print(f"{'=' * 70}")
    
    predictor = BattlePredictor()
    predictor.model = best_model
    predictor.scaler = scaler
    predictor.feature_names = BattlePreprocessor.feature_names()
    predictor.model_metadata = {
        'architecture': best_name,
        'test_r2': max(metrics1['test_r2'], metrics2['test_r2']),
        'test_rmse': min(metrics1['test_rmse'], metrics2['test_rmse']),
        'test_mae': min(metrics1['test_mae'], metrics2['test_mae']),
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    predictor.save("battle_predictor_best.json")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()