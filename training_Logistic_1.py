"""
Battle Win Chance Predictor - Training Script

This script trains a Ridge Regression model to predict battle outcomes.

Flow:
1. Load CSV data
2. Preprocess into numerical features
3. Split into train/test sets
4. Scale features
5. Train model
6. Evaluate performance
7. Show feature importance
"""

from CSVParser import CSVParser
from BattlelPreprocessor import BattlePreprocessor
from data_utils import train_test_split, StandardScaler
from logisticRegressionModel import (
    LogisticRegression,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r_squared
)
from visualizations import create_all_visualizations

def main():
    print("=" * 70)
    print("BATTLE WIN CHANCE PREDICTOR - TRAINING")
    print("=" * 70)
    
    # ═════════════════════════════════════════════════════════════════
    # STEP 1: Load Data
    # ═════════════════════════════════════════════════════════════════
    print("\n[1/7] Loading CSV data...")
    parser = CSVParser("battle_dataset_10000.csv")
    print(f"      ✓ Loaded {parser.count() - 1} battles (+ 1 header row)")
    
    # ═════════════════════════════════════════════════════════════════
    # STEP 2: Preprocess
    # ═════════════════════════════════════════════════════════════════
    print("\n[2/7] Preprocessing...")
    preprocessor = BattlePreprocessor(parser.fetch_all())
    X, y = preprocessor.preprocess()
    print(f"      ✓ Created {len(X)} feature vectors")
    print(f"      ✓ Each vector has {len(X[0])} features")
    print(f"      ✓ Target range: {min(y):.1f}% to {max(y):.1f}%")
    
    # ═════════════════════════════════════════════════════════════════
    # STEP 3: Train/Test Split
    # ═════════════════════════════════════════════════════════════════
    print("\n[3/7] Splitting into train/test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_seed=42
    )
    print(f"      ✓ Training set: {len(X_train)} battles")
    print(f"      ✓ Test set:     {len(X_test)} battles")
    
    # ═════════════════════════════════════════════════════════════════
    # STEP 4: Feature Scaling
    # ═════════════════════════════════════════════════════════════════
    print("\n[4/7] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"      ✓ Fitted scaler on training data")
    print(f"      ✓ Applied to both train and test sets")
    
    # ═════════════════════════════════════════════════════════════════
    # STEP 5: Train Model
    # ═════════════════════════════════════════════════════════════════
    print("\n[5/7] Training Logistic Regression model...")
    print(f"      (learning_rate=0.01, epochs=1000, regularization=0.01)")
    model = LogisticRegression(
        learning_rate=0.01,
        epochs=1000,
        batch_size=64,
        regularization=0.01
    )
    model.fit(X_train_scaled, y_train, verbose=True)
    print(f"      ✓ Model trained successfully")
    print(f"      ✓ Learned {len(model.weights)} weights + 1 bias")
    
    # ═════════════════════════════════════════════════════════════════
    # STEP 6: Make Predictions
    # ═════════════════════════════════════════════════════════════════
    print("\n[6/7] Making predictions...")
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    print(f"      ✓ Generated predictions for all battles")
    
    # ═════════════════════════════════════════════════════════════════
    # STEP 7: Evaluate Performance
    # ═════════════════════════════════════════════════════════════════
    print("\n[7/7] Evaluating model...")
    
    # Calculate metrics
    metrics = {
        "MSE": (mean_squared_error, "Lower is better"),
        "RMSE": (root_mean_squared_error, "Avg error in percentage points"),
        "MAE": (mean_absolute_error, "Avg absolute error"),
        "R²": (r_squared, "1.0 = perfect, 0.0 = no better than mean"),
    }
    
    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE")
    print("=" * 70)
    print(f"{'Metric':<10} {'Train':>12} {'Test':>12}  {'Meaning':<30}")
    print("-" * 70)
    
    for name, (metric_fn, meaning) in metrics.items():
        train_score = metric_fn(y_train, y_pred_train)
        test_score = metric_fn(y_test, y_pred_test)
        print(f"{name:<10} {train_score:>12.4f} {test_score:>12.4f}  {meaning}")
    
    print("=" * 70)
    
    # ═════════════════════════════════════════════════════════════════
    # Feature Importance
    # ═════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TOP 15 MOST IMPORTANT FEATURES")
    print("=" * 70)
    print("(Ranked by absolute weight — larger magnitude = more impact)\n")
    
    feature_names = BattlePreprocessor.feature_names()
    importance = model.get_feature_importance(feature_names)
    
    print(f"{'Feature':<50} {'Weight':>10}  {'Impact'}")
    print("-" * 70)
    
    for i, (name, weight) in enumerate(importance[:15]):
        sign = "↑" if weight > 0 else "↓"
        impact = "helps you win" if weight > 0 else "hurts you"
        
        # Visual bar
        bar_length = int(abs(weight) * 2)
        bar = "█" * min(bar_length, 20)
        
        print(f"{name:<50} {weight:>+9.3f}  {sign} {impact}")
        if i == 14:
            print("-" * 70)
    
    # ═════════════════════════════════════════════════════════════════
    # Sample Predictions
    # ═════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS (10 random test battles)")
    print("=" * 70)
    print(f"{'Actual':>8}  {'Predicted':>10}  {'Error':>8}")
    print("-" * 70)
    
    for i in range(min(10, len(y_test))):
        actual = y_test[i]
        predicted = y_pred_test[i]
        error = predicted - actual
        print(f"{actual:>7.1f}%  {predicted:>9.1f}%  {error:>+7.1f}%")
    
    print("=" * 70)
    
    # ═════════════════════════════════════════════════════════════════
    # Summary
    # ═════════════════════════════════════════════════════════════════
    test_r2 = r_squared(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✓ Model explains {test_r2*100:.1f}% of variance in test data")
    print(f"✓ Average prediction error: ±{test_mae:.1f} percentage points")
    print(f"✓ Predictions guaranteed to be in 0-100% range (sigmoid output)")
    print(f"✓ Model successfully trained on {len(X_train)} battles")
    print("=" * 70)
    print("\nNext steps:")
    print("  • Adjust learning_rate (0.001-0.1) for better convergence")
    print("  • Try more epochs (2000-5000) for better fit")
    print("  • Experiment with neural networks for even better performance")
    print("  • Use the model to predict outcomes of new battles!")
    print("=" * 70)
    
    # ═════════════════════════════════════════════════════════════════
    # VISUALIZATIONS
    # ═════════════════════════════════════════════════════════════════
    feature_names = BattlePreprocessor.feature_names()
    create_all_visualizations(
        y_train, y_pred_train,
        y_test, y_pred_test,
        feature_names, model.weights
    )


if __name__ == "__main__":
    main()