import json
import os
from typing import Dict, Any
from neural_network import NeuralNetwork
from data_utils import StandardScaler

class BattlePredictor:
    """
    Wrapper class for the trained battle prediction model.
    
    Handles:
    - Saving/loading trained models
    - Making predictions on new battles
    - Providing confidence scores
    """
    
    def __init__(self):
        self.model: NeuralNetwork = None
        self.scaler: StandardScaler = None
        self.feature_names = None
        self.model_metadata = {}
    
    # ═════════════════════════════════════════════════════════════════
    # SAVE / LOAD
    # ═════════════════════════════════════════════════════════════════
    
    def save(self, filepath: str = "battle_predictor.json"):
        """
        Save the trained model to a JSON file.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("No model to save. Train a model first.")
        
        model_data = {
            # Model architecture
            "input_size": self.model.input_size,
            "hidden_sizes": self.model.hidden_sizes,
            
            # Model weights and biases
            "weights": self.model.weights,
            "biases": self.model.biases,
            
            # Scaler parameters
            "scaler_means": self.scaler.means,
            "scaler_stds": self.scaler.stds,
            
            # Metadata
            "feature_names": self.feature_names,
            "metadata": self.model_metadata
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"✓ Model saved to: {filepath}")
        print(f"  Architecture: {self.model.get_architecture_summary()}")
        print(f"  Test R²: {self.model_metadata.get('test_r2', 'N/A')}")
    
    @classmethod
    def load(cls, filepath: str = "battle_predictor.json") -> "BattlePredictor":
        """
        Load a trained model from a JSON file.
        
        Args:
            filepath: Path to the saved model
        
        Returns:
            BattlePredictor instance with loaded model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        # Create predictor instance
        predictor = cls()
        
        # Reconstruct model
        predictor.model = NeuralNetwork(
            input_size=model_data["input_size"],
            hidden_sizes=model_data["hidden_sizes"]
        )
        predictor.model.weights = model_data["weights"]
        predictor.model.biases = model_data["biases"]
        predictor.model._is_fitted = True
        
        # Reconstruct scaler
        predictor.scaler = StandardScaler()
        predictor.scaler.means = model_data["scaler_means"]
        predictor.scaler.stds = model_data["scaler_stds"]
        predictor.scaler._is_fitted = True
        
        # Metadata
        predictor.feature_names = model_data.get("feature_names")
        predictor.model_metadata = model_data.get("metadata", {})
        
        print(f"✓ Model loaded from: {filepath}")
        print(f"  Architecture: {predictor.model.get_architecture_summary()}")
        print(f"  Test R²: {predictor.model_metadata.get('test_r2', 'N/A')}")
        
        return predictor
    
    # ═════════════════════════════════════════════════════════════════
    # PREDICTION
    # ═════════════════════════════════════════════════════════════════
    
    def predict_battle(self, features: list, return_confidence: bool = False) -> float:
        """
        Predict win probability for a single battle.
        
        Args:
            features: List of 34 feature values (log-ratios + one-hot encoded)
            return_confidence: If True, return (prediction, confidence_level)
        
        Returns:
            Win probability (0-100%) or tuple (probability, confidence)
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Load a model first.")
        
        # Scale features
        scaled = self.scaler.transform([features])[0]
        
        # Predict
        prediction = self.model.predict([scaled])[0]
        
        if return_confidence:
            # Confidence based on distance from 50%
            # Predictions near 0% or 100% are high confidence
            # Predictions near 50% are low confidence
            distance_from_middle = abs(prediction - 50.0)
            confidence = min(100, distance_from_middle * 2)  # 0-100 scale
            
            if confidence > 80:
                confidence_level = "Very High"
            elif confidence > 60:
                confidence_level = "High"
            elif confidence > 40:
                confidence_level = "Medium"
            elif confidence > 20:
                confidence_level = "Low"
            else:
                confidence_level = "Very Low"
            
            return prediction, confidence_level
        
        return prediction
    
    def predict_batch(self, features_list: list) -> list:
        """
        Predict win probabilities for multiple battles.
        
        Args:
            features_list: List of feature vectors
        
        Returns:
            List of predictions (0-100%)
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Load a model first.")
        
        # Scale all features
        scaled = self.scaler.transform(features_list)
        
        # Predict
        return self.model.predict(scaled)