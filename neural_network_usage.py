"""
Battle Win Predictor - Usage Examples

This file shows how to use the trained model to predict battle outcomes.
"""

from battlePredictor import BattlePredictor
from BattlelPreprocessor import BattlePreprocessor
import math

# ═════════════════════════════════════════════════════════════════════
# LOAD THE TRAINED MODEL
# ═════════════════════════════════════════════════════════════════════

print("Loading trained model...")
predictor = BattlePredictor.load("battle_predictor_best.json")
print()

# ═════════════════════════════════════════════════════════════════════
# HELPER FUNCTION: Convert battle parameters to features
# ═════════════════════════════════════════════════════════════════════

def create_battle_features(
    # Ratios (enemy / own)
    enemy_army_size: float,
    own_army_size: float,
    enemy_training: float,
    own_training: float,
    enemy_equipment: float,
    own_equipment: float,
    enemy_supplies: float,
    own_supplies: float,
    enemy_morale: float,
    own_morale: float,
    enemy_fatigue: float,
    own_fatigue: float,
    enemy_commander_skill: float,
    own_commander_skill: float,
    enemy_command_efficiency: float,
    own_command_efficiency: float,
    enemy_tech_level: float,
    own_tech_level: float,
    # Categorical features
    weather: str,  # "Clear", "Cloudy", "Rain", "Fog", "Storm"
    visibility: str,  # "Very Low", "Low", "Moderate", "Good", "Excellent"
    time_of_day: str,  # "Dawn", "Morning", "Midday", "Afternoon", "Dusk", "Night"
    own_posture: str,  # "Attack", "Defend"
    enemy_posture: str,  # "Attack", "Defend"
    surprise: str,  # "None", "Partial", "Full Ambush"
    home_territory: str  # "No", "Yes"
) -> list:
    """
    Convert battle parameters into the 34-feature vector needed by the model.
    
    Returns:
        List of 34 features ready for prediction
    """
    features = []
    
    # Step 1: Calculate and log-transform ratios
    ratios = [
        enemy_army_size / own_army_size,
        enemy_training / own_training,
        enemy_equipment / own_equipment,
        enemy_supplies / own_supplies,
        enemy_morale / own_morale,
        enemy_fatigue / own_fatigue,
        enemy_commander_skill / own_commander_skill,
        enemy_command_efficiency / own_command_efficiency,
        enemy_tech_level / own_tech_level
    ]
    
    for ratio in ratios:
        features.append(math.log(max(ratio, 1e-9)))
    
    # Step 2: One-hot encode categoricals (in the exact same order as training)
    # Weather
    for val in ["Clear", "Cloudy", "Rain", "Fog", "Storm"]:
        features.append(1.0 if weather == val else 0.0)
    
    # Visibility
    for val in ["Very Low", "Low", "Moderate", "Good", "Excellent"]:
        features.append(1.0 if visibility == val else 0.0)
    
    # Time of day
    for val in ["Dawn", "Morning", "Midday", "Afternoon", "Dusk", "Night"]:
        features.append(1.0 if time_of_day == val else 0.0)
    
    # Own posture
    for val in ["Defend", "Attack"]:
        features.append(1.0 if own_posture == val else 0.0)
    
    # Enemy posture
    for val in ["Defend", "Attack"]:
        features.append(1.0 if enemy_posture == val else 0.0)
    
    # Surprise
    for val in ["None", "Partial", "Full Ambush"]:
        features.append(1.0 if surprise == val else 0.0)
    
    # Home territory
    for val in ["No", "Yes"]:
        features.append(1.0 if home_territory == val else 0.0)
    
    return features


# ═════════════════════════════════════════════════════════════════════
# EXAMPLE 1: Evenly matched battle
# ═════════════════════════════════════════════════════════════════════

print("=" * 70)
print("EXAMPLE 1: Evenly Matched Battle")
print("=" * 70)

features = create_battle_features(
    # Both sides have equal forces
    enemy_army_size=5000, own_army_size=5000,
    enemy_training=7.0, own_training=7.0,
    enemy_equipment=8.0, own_equipment=8.0,
    enemy_supplies=3000, own_supplies=3000,
    enemy_morale=7.5, own_morale=7.5,
    enemy_fatigue=5.0, own_fatigue=5.0,
    enemy_commander_skill=6.0, own_commander_skill=6.0,
    enemy_command_efficiency=7.0, own_command_efficiency=7.0,
    enemy_tech_level=7.0, own_tech_level=7.0,
    # Neutral conditions
    weather="Clear",
    visibility="Good",
    time_of_day="Midday",
    own_posture="Attack",
    enemy_posture="Defend",
    surprise="None",
    home_territory="No"
)

prediction, confidence = predictor.predict_battle(features, return_confidence=True)
print(f"\nPredicted win chance: {prediction:.1f}%")
print(f"Confidence: {confidence}")
print()

# ═════════════════════════════════════════════════════════════════════
# EXAMPLE 2: You have the advantage
# ═════════════════════════════════════════════════════════════════════

print("=" * 70)
print("EXAMPLE 2: You Have Significant Advantages")
print("=" * 70)

features = create_battle_features(
    # You outnumber them 2:1
    enemy_army_size=3000, own_army_size=6000,
    enemy_training=5.0, own_training=8.0,
    enemy_equipment=6.0, own_equipment=9.0,
    enemy_supplies=1500, own_supplies=4000,
    enemy_morale=5.0, own_morale=8.5,
    enemy_fatigue=7.0, own_fatigue=3.0,  # They're tired, you're fresh
    enemy_commander_skill=5.0, own_commander_skill=8.0,
    enemy_command_efficiency=5.0, own_command_efficiency=8.0,
    enemy_tech_level=6.0, own_tech_level=9.0,
    # Good conditions + you have surprise
    weather="Clear",
    visibility="Excellent",
    time_of_day="Morning",
    own_posture="Attack",
    enemy_posture="Defend",
    surprise="Full Ambush",
    home_territory="Yes"
)

prediction, confidence = predictor.predict_battle(features, return_confidence=True)
print(f"\nPredicted win chance: {prediction:.1f}%")
print(f"Confidence: {confidence}")
print()

# ═════════════════════════════════════════════════════════════════════
# EXAMPLE 3: You're outnumbered but have better position
# ═════════════════════════════════════════════════════════════════════

print("=" * 70)
print("EXAMPLE 3: Outnumbered But Defending Home Territory")
print("=" * 70)

features = create_battle_features(
    # They outnumber you
    enemy_army_size=8000, own_army_size=4000,
    # But you have better training and morale
    enemy_training=5.0, own_training=9.0,
    enemy_equipment=6.0, own_equipment=8.0,
    enemy_supplies=3000, own_supplies=3000,
    enemy_morale=6.0, own_morale=9.0,
    enemy_fatigue=4.0, own_fatigue=3.0,
    enemy_commander_skill=6.0, own_commander_skill=9.0,
    enemy_command_efficiency=6.0, own_command_efficiency=9.0,
    enemy_tech_level=6.0, own_tech_level=8.0,
    # Defending at home
    weather="Clear",
    visibility="Good",
    time_of_day="Afternoon",
    own_posture="Defend",
    enemy_posture="Attack",
    surprise="None",
    home_territory="Yes"
)

prediction, confidence = predictor.predict_battle(features, return_confidence=True)
print(f"\nPredicted win chance: {prediction:.1f}%")
print(f"Confidence: {confidence}")
print()

# ═════════════════════════════════════════════════════════════════════
# EXAMPLE 4: Bad weather hurts you
# ═════════════════════════════════════════════════════════════════════

print("=" * 70)
print("EXAMPLE 4: Equal Forces, But Storm Reduces Visibility")
print("=" * 70)

features = create_battle_features(
    # Equal forces
    enemy_army_size=5000, own_army_size=5000,
    enemy_training=7.0, own_training=7.0,
    enemy_equipment=7.0, own_equipment=7.0,
    enemy_supplies=2500, own_supplies=2500,
    enemy_morale=7.0, own_morale=7.0,
    enemy_fatigue=5.0, own_fatigue=5.0,
    enemy_commander_skill=6.0, own_commander_skill=6.0,
    enemy_command_efficiency=6.0, own_command_efficiency=6.0,
    enemy_tech_level=7.0, own_tech_level=7.0,
    # But terrible conditions
    weather="Storm",
    visibility="Very Low",
    time_of_day="Night",
    own_posture="Attack",
    enemy_posture="Defend",
    surprise="None",
    home_territory="No"
)

prediction, confidence = predictor.predict_battle(features, return_confidence=True)
print(f"\nPredicted win chance: {prediction:.1f}%")
print(f"Confidence: {confidence}")
print()

# ═════════════════════════════════════════════════════════════════════
# EXAMPLE 5: Batch prediction (multiple battles at once)
# ═════════════════════════════════════════════════════════════════════

print("=" * 70)
print("EXAMPLE 5: Batch Prediction (3 battles)")
print("=" * 70)

battle1 = create_battle_features(
    enemy_army_size=3000, own_army_size=5000,
    enemy_training=6.0, own_training=8.0,
    enemy_equipment=6.0, own_equipment=8.0,
    enemy_supplies=2000, own_supplies=3000,
    enemy_morale=6.0, own_morale=8.0,
    enemy_fatigue=6.0, own_fatigue=4.0,
    enemy_commander_skill=6.0, own_commander_skill=8.0,
    enemy_command_efficiency=6.0, own_command_efficiency=8.0,
    enemy_tech_level=6.0, own_tech_level=8.0,
    weather="Clear", visibility="Good", time_of_day="Morning",
    own_posture="Attack", enemy_posture="Defend",
    surprise="Partial", home_territory="Yes"
)

battle2 = create_battle_features(
    enemy_army_size=7000, own_army_size=4000,
    enemy_training=8.0, own_training=6.0,
    enemy_equipment=8.0, own_equipment=6.0,
    enemy_supplies=4000, own_supplies=2000,
    enemy_morale=8.0, own_morale=6.0,
    enemy_fatigue=4.0, own_fatigue=6.0,
    enemy_commander_skill=8.0, own_commander_skill=6.0,
    enemy_command_efficiency=8.0, own_command_efficiency=6.0,
    enemy_tech_level=8.0, own_tech_level=6.0,
    weather="Rain", visibility="Low", time_of_day="Dusk",
    own_posture="Defend", enemy_posture="Attack",
    surprise="None", home_territory="No"
)

battle3 = create_battle_features(
    enemy_army_size=5000, own_army_size=5000,
    enemy_training=7.0, own_training=7.0,
    enemy_equipment=7.0, own_equipment=7.0,
    enemy_supplies=2500, own_supplies=2500,
    enemy_morale=7.0, own_morale=7.0,
    enemy_fatigue=5.0, own_fatigue=5.0,
    enemy_commander_skill=7.0, own_commander_skill=7.0,
    enemy_command_efficiency=7.0, own_command_efficiency=7.0,
    enemy_tech_level=7.0, own_tech_level=7.0,
    weather="Cloudy", visibility="Moderate", time_of_day="Midday",
    own_posture="Attack", enemy_posture="Defend",
    surprise="None", home_territory="No"
)

predictions = predictor.predict_batch([battle1, battle2, battle3])

print("\nBattle 1 (You have advantages):    {:.1f}%".format(predictions[0]))
print("Battle 2 (Enemy has advantages):   {:.1f}%".format(predictions[1]))
print("Battle 3 (Even match):             {:.1f}%".format(predictions[2]))
print()

print("=" * 70)
print("Done! The model is ready to use.")
print("=" * 70)