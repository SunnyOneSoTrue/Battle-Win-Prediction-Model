import math
from typing import List
import random

class LogisticRegression:
    """
    Logistic Regression for predicting win probability (0-100%).
    
    Unlike Linear Regression which can output any value, Logistic Regression
    uses a sigmoid function to guarantee outputs between 0% and 100%.
    
    Model:
        z = bias + w1×f1 + w2×f2 + ... + w34×f34
        probability = 100 × sigmoid(z)
        where sigmoid(z) = 1 / (1 + e^(-z))
    
    Training uses gradient descent with cross-entropy loss.
    """
    
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000, 
                 batch_size: int = 32, regularization: float = 0.01):
        """
        Initialize the model.
        
        Args:
            learning_rate: Step size for gradient descent (default: 0.01)
            epochs: Number of training iterations (default: 1000)
            batch_size: Mini-batch size (default: 32)
            regularization: L2 penalty strength (default: 0.01)
        """
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.reg = regularization
        
        self.weights: List[float] = []
        self.bias: float = 0.0
        self._is_fitted = False
    
    # ═════════════════════════════════════════════════════════════════
    # CORE FUNCTIONS
    # ═════════════════════════════════════════════════════════════════
    
    @staticmethod
    def _sigmoid(z: float) -> float:
        """
        Sigmoid activation function.
        
        Maps any real number to (0, 1).
        sigmoid(0) = 0.5
        sigmoid(large positive) → 1
        sigmoid(large negative) → 0
        """
        # Clip to prevent overflow
        z = max(min(z, 500), -500)
        return 1.0 / (1.0 + math.exp(-z))
    
    def _compute_z(self, features: List[float]) -> float:
        """Compute the linear combination z = bias + w·x"""
        return self.bias + sum(w * f for w, f in zip(self.weights, features))
    
    def _predict_proba_single(self, features: List[float]) -> float:
        """Predict probability for a single sample (0-100%)."""
        z = self._compute_z(features)
        prob_0_to_1 = self._sigmoid(z)
        return prob_0_to_1 * 100.0  # Scale to 0-100%
    
    # ═════════════════════════════════════════════════════════════════
    # TRAINING
    # ═════════════════════════════════════════════════════════════════
    
    def fit(self, X: List[List[float]], y: List[float], verbose: bool = True) -> "LogisticRegression":
        """
        Train the model using gradient descent with MSE loss.
        
        We use MSE (mean squared error) instead of cross-entropy because
        our target is a continuous probability (0-100%), not a binary class.
        
        Args:
            X: Training features
            y: Training targets (win_chance_% from 0-100)
            verbose: Print progress during training
        
        Returns:
            self (for method chaining)
        """
        if not X or not y:
            raise ValueError("X and y cannot be empty")
        
        n_samples = len(X)
        n_features = len(X[0])
        
        # Initialize weights randomly (small values)
        random.seed(42)
        self.weights = [random.gauss(0, 0.01) for _ in range(n_features)]
        self.bias = 0.0
        
        # Training loop
        for epoch in range(self.epochs):
            # Shuffle data
            indices = list(range(n_samples))
            random.shuffle(indices)
            
            epoch_loss = 0.0
            
            # Mini-batch gradient descent
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch_indices = indices[start:end]
                batch_size = len(batch_indices)
                
                # Accumulate gradients for this batch
                grad_weights = [0.0] * n_features
                grad_bias = 0.0
                batch_loss = 0.0
                
                for idx in batch_indices:
                    features = X[idx]
                    target = y[idx]  # 0-100%
                    
                    # Forward pass
                    z = self._compute_z(features)
                    pred = self._sigmoid(z) * 100.0  # 0-100%
                    
                    # Loss (MSE)
                    error = pred - target
                    batch_loss += error ** 2
                    
                    # Backward pass (gradient of MSE w.r.t. weights)
                    # dLoss/dw = 2 * error * d(sigmoid(z)*100)/dw
                    #          = 2 * error * 100 * sigmoid'(z) * features
                    # where sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))
                    
                    sig = self._sigmoid(z)
                    d_sigmoid = sig * (1 - sig)
                    
                    # gradient = 2 * error * 100 * d_sigmoid
                    gradient = 2.0 * error * 100.0 * d_sigmoid
                    
                    # Accumulate gradients
                    for j in range(n_features):
                        grad_weights[j] += gradient * features[j]
                    grad_bias += gradient
                
                epoch_loss += batch_loss
                
                # Average gradients over batch
                for j in range(n_features):
                    grad_weights[j] /= batch_size
                grad_bias /= batch_size
                
                # Add L2 regularization gradient
                for j in range(n_features):
                    grad_weights[j] += 2 * self.reg * self.weights[j]
                
                # Update weights (gradient descent)
                for j in range(n_features):
                    self.weights[j] -= self.lr * grad_weights[j]
                self.bias -= self.lr * grad_bias
            
            # Print progress
            if verbose and (epoch % 100 == 0 or epoch == self.epochs - 1):
                avg_loss = epoch_loss / n_samples
                rmse = math.sqrt(avg_loss)
                print(f"  Epoch {epoch+1:>4}/{self.epochs}   MSE={avg_loss:>8.2f}   RMSE={rmse:>6.2f}%")
        
        self._is_fitted = True
        return self
    
    # ═════════════════════════════════════════════════════════════════
    # PREDICTION
    # ═════════════════════════════════════════════════════════════════
    
    def predict(self, X: List[List[float]]) -> List[float]:
        """
        Predict win probabilities for new data.
        
        Args:
            X: Feature vectors
        
        Returns:
            List of predictions (0-100%)
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before predict()")
        
        return [self._predict_proba_single(features) for features in X]
    
    # ═════════════════════════════════════════════════════════════════
    # UTILITY
    # ═════════════════════════════════════════════════════════════════
    
    def get_feature_importance(self, feature_names: List[str]) -> List[tuple]:
        """
        Get features ranked by absolute weight magnitude.
        
        Args:
            feature_names: Names of features
        
        Returns:
            List of (feature_name, weight) tuples sorted by |weight|
        """
        if not self._is_fitted:
            raise RuntimeError("Must fit model before checking importance")
        
        importance = list(zip(feature_names, self.weights))
        importance.sort(key=lambda x: abs(x[1]), reverse=True)
        return importance


# ═════════════════════════════════════════════════════════════════════
# METRICS (same as before)
# ═════════════════════════════════════════════════════════════════════

def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    """MSE - lower is better."""
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)


def root_mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    """RMSE - average error in percentage points."""
    return math.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true: List[float], y_pred: List[float]) -> float:
    """MAE - average absolute error."""
    return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / len(y_true)


def r_squared(y_true: List[float], y_pred: List[float]) -> float:
    """R² - proportion of variance explained (1.0 = perfect)."""
    mean_y = sum(y_true) / len(y_true)
    ss_total = sum((yt - mean_y) ** 2 for yt in y_true)
    ss_residual = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    
    if ss_total == 0:
        return 0.0
    
    return 1.0 - (ss_residual / ss_total)