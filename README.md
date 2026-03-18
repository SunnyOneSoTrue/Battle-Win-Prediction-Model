# Battle Win Predictor - Usage Guide

A neural network model that predicts battle outcomes with 99.6% accuracy.

## 📊 Model Performance

- **R² Score**: 0.9962 (explains 99.62% of variance)
- **RMSE**: 2.37% (average error)
- **Accuracy**: 99.95% of predictions within 10% error
- **Architecture**: 34 → 48 → 24 → 1 (Conservative Neural Network)

---

## 🚀 Quick Start

### 1. Load the Model

```python
from model_saver import BattlePredictor

# Load the trained model
predictor = BattlePredictor.load("battle_predictor_best.json")
```

### 2. Make a Prediction

```python
# Create feature vector (see examples below)
features = create_battle_features(
    enemy_army_size=5000, own_army_size=6000,
    enemy_training=6.0, own_training=8.0,
    # ... all 18 parameters ...
)

# Get prediction
win_chance = predictor.predict_battle(features)
print(f"Win chance: {win_chance:.1f}%")

# Get prediction with confidence
win_chance, confidence = predictor.predict_battle(features, return_confidence=True)
print(f"Win chance: {win_chance:.1f}% (Confidence: {confidence})")
```

---

## 📝 Input Parameters

The model requires **18 parameters** (9 ratios + 9 categorical features):

### Ratio Parameters (enemy/own):
1. **Army Size**: Total troop count
2. **Training Level**: 1-10 scale
3. **Equipment Prestige**: 1-10 scale
4. **Supplies**: Amount of supplies
5. **Morale**: 1-10 scale
6. **Fatigue**: 1-10 scale (higher = more tired)
7. **Commander Skill**: 1-10 scale
8. **Command Efficiency**: 1-10 scale
9. **Technology Level**: 1-10 scale

### Categorical Parameters:
10. **Weather**: "Clear", "Cloudy", "Rain", "Fog", "Storm"
11. **Visibility**: "Very Low", "Low", "Moderate", "Good", "Excellent"
12. **Time of Day**: "Dawn", "Morning", "Midday", "Afternoon", "Dusk", "Night"
13. **Own Posture**: "Attack", "Defend"
14. **Enemy Posture**: "Attack", "Defend"
15. **Surprise Factor**: "None", "Partial", "Full Ambush"
16. **Home Territory**: "No", "Yes"

---

## 💡 Examples

### Example 1: Evenly Matched Battle

```python
features = create_battle_features(
    enemy_army_size=5000, own_army_size=5000,
    enemy_training=7.0, own_training=7.0,
    enemy_equipment=8.0, own_equipment=8.0,
    enemy_supplies=3000, own_supplies=3000,
    enemy_morale=7.5, own_morale=7.5,
    enemy_fatigue=5.0, own_fatigue=5.0,
    enemy_commander_skill=6.0, own_commander_skill=6.0,
    enemy_command_efficiency=7.0, own_command_efficiency=7.0,
    enemy_tech_level=7.0, own_tech_level=7.0,
    weather="Clear",
    visibility="Good",
    time_of_day="Midday",
    own_posture="Attack",
    enemy_posture="Defend",
    surprise="None",
    home_territory="No"
)

prediction = predictor.predict_battle(features)
# Expected: ~50-55% (slight advantage for attacker)
```

### Example 2: You Have the Advantage

```python
features = create_battle_features(
    enemy_army_size=3000, own_army_size=6000,  # 2:1 advantage
    enemy_training=5.0, own_training=8.0,
    enemy_equipment=6.0, own_equipment=9.0,
    enemy_supplies=1500, own_supplies=4000,
    enemy_morale=5.0, own_morale=8.5,
    enemy_fatigue=7.0, own_fatigue=3.0,  # Enemy is tired
    enemy_commander_skill=5.0, own_commander_skill=8.0,
    enemy_command_efficiency=5.0, own_command_efficiency=8.0,
    enemy_tech_level=6.0, own_tech_level=9.0,
    weather="Clear",
    visibility="Excellent",
    time_of_day="Morning",
    own_posture="Attack",
    enemy_posture="Defend",
    surprise="Full Ambush",
    home_territory="Yes"
)

prediction = predictor.predict_battle(features)
# Expected: ~95-99% (overwhelming advantage)
```

### Example 3: Batch Predictions

```python
# Predict multiple battles at once
battle1_features = create_battle_features(...)
battle2_features = create_battle_features(...)
battle3_features = create_battle_features(...)

predictions = predictor.predict_batch([
    battle1_features,
    battle2_features,
    battle3_features
])

# predictions = [85.3, 23.7, 51.2]  (example output)
```

---

## 📁 Files

- `battle_predictor_best.json` - Trained model (load this)
- `model_saver.py` - Save/load functionality
- `use_model.py` - Complete usage examples
- `neural_network.py` - Model architecture
- `battle_preprocessor.py` - Feature preprocessing
- `data_utils.py` - Scaling utilities

---

## 🔧 Re-training

To retrain the model with new data:

```bash
python train_neural_networks.py
```

This will:
1. Train both architectures (64→32 and 48→24)
2. Compare performance
3. Save the best model as `battle_predictor_best.json`
4. Generate visualizations

---

## ⚠️ Important Notes

1. **Feature Order Matters**: The 34 features must be in the exact order:
   - 9 log-transformed ratios
   - 25 one-hot encoded categorical values

2. **Scaling**: Features are automatically scaled by the loaded model

3. **Output Range**: Predictions are guaranteed to be 0-100%

4. **Confidence**: Higher confidence for predictions near 0% or 100%, lower for ~50%

---

## 📊 Model Architecture

```
Input Layer (34 features)
    ↓
Hidden Layer 1 (48 neurons, ReLU activation)
    ↓
Hidden Layer 2 (24 neurons, ReLU activation)
    ↓
Output Layer (1 neuron, Sigmoid × 100)
    ↓
Win Probability (0-100%)
```

**Total Parameters**: 2,881
**Training Samples**: 8,000
**Test Samples**: 2,000

---

## 🎯 Interpreting Results

| Prediction | Meaning |
|-----------|---------|
| 90-100% | Very likely to win - overwhelming advantage |
| 70-90% | Likely to win - significant advantage |
| 55-70% | Moderate advantage |
| 45-55% | Toss-up - even match |
| 30-45% | Moderate disadvantage |
| 10-30% | Likely to lose - significant disadvantage |
| 0-10% | Very likely to lose - overwhelming disadvantage |

---

## 📞 Support

For issues or questions, check `use_model.py` for complete working examples.
