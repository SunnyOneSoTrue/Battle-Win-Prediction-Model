import math
from typing import List

class RidgeRegression:
    """
    Ridge Regression model (Linear Regression with L2 regularization).
    
    What it does:
    - Learns a weight for each feature
    - Prediction = bias + (weight1 × feature1) + (weight2 × feature2) + ...
    - The "Ridge" part adds a penalty for large weights (prevents overfitting)
    
    Why Ridge instead of plain Linear Regression?
    - With 34 features and only 8000 training samples, plain linear regression
      might overfit (memorize training data instead of learning patterns)
    - Ridge adds a small penalty: "prefer smaller weights unless the data
      really demands large ones"
    - The alpha parameter controls this: higher alpha = stronger penalty
    
    Math:
    - Plain linear regression minimizes: sum((y - prediction)^2)
    - Ridge regression minimizes: sum((y - prediction)^2) + alpha × sum(weights^2)
                                   ^^^^ fit the data ^^^^   ^^^^ keep weights small ^^^^
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Initialize the model.
        
        Args:
            alpha: Regularization strength
                   - 0 = no regularization (plain linear regression)
                   - Small (0.1) = weak regularization
                   - Medium (1.0) = balanced (good default)
                   - Large (10.0) = strong regularization
        """
        self.alpha = alpha
        self.weights: List[float] = []
        self.bias: float = 0.0
        self._is_fitted = False
    
    def fit(self, X: List[List[float]], y: List[float]) -> "RidgeRegression":
        """
        Train the model on data.
        
        This uses the "closed-form solution" - a mathematical formula that
        directly calculates the optimal weights. No iteration needed!
        
        Formula: weights = (X^T X + alpha*I)^(-1) X^T y
        
        Where:
        - X^T = X transposed
        - I = identity matrix
        - ^(-1) = matrix inverse
        
        Args:
            X: Training features (each row is one battle)
            y: Training targets (win_chance_% for each battle)
        
        Returns:
            self (for method chaining)
        """
        if not X or not y:
            raise ValueError("X and y cannot be empty")
        
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length. Got {len(X)} and {len(y)}")
        
        n_samples = len(X)
        n_features = len(X[0])
        
        # Add a column of 1s to X for the bias term
        # This is a trick: instead of tracking bias separately, we treat it
        # as just another weight with a constant feature value of 1
        X_with_bias = [[1.0] + row for row in X]
        
        # Now we solve: (X^T X + alpha*I)^(-1) X^T y
        
        # Step 1: X^T (transpose)
        X_t = self._transpose(X_with_bias)
        
        # Step 2: X^T X
        XtX = self._matrix_multiply(X_t, X_with_bias)
        
        # Step 3: Add alpha to diagonal (except bias term at [0,0])
        # This is the regularization penalty
        for i in range(len(XtX)):
            if i > 0:  # Skip the bias term (first row/col)
                XtX[i][i] += self.alpha
        
        # Step 4: X^T y
        Xty = [sum(X_t[i][j] * y[j] for j in range(n_samples)) 
               for i in range(len(X_t))]
        
        # Step 5: Solve (XtX)^(-1) Xty using Gaussian elimination
        solution = self._solve_linear_system(XtX, Xty)
        
        # Extract bias and weights
        self.bias = solution[0]
        self.weights = solution[1:]
        self._is_fitted = True
        
        return self
    
    def predict(self, X: List[List[float]]) -> List[float]:
        """
        Make predictions on new data.
        
        For each battle:
            prediction = bias + w1*f1 + w2*f2 + ... + w34*f34
        
        Args:
            X: Feature vectors to predict on
        
        Returns:
            List of predictions (win_chance_% for each battle)
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before predict()")
        
        predictions = []
        for row in X:
            # Dot product of weights and features, plus bias
            pred = self.bias + sum(w * f for w, f in zip(self.weights, row))
            predictions.append(pred)
        
        return predictions
    
    # ═════════════════════════════════════════════════════════════════
    # INTERNAL HELPER METHODS (Linear Algebra)
    # ═════════════════════════════════════════════════════════════════
    
    @staticmethod
    def _transpose(matrix: List[List[float]]) -> List[List[float]]:
        """Transpose a matrix (flip rows and columns)."""
        return [list(col) for col in zip(*matrix)]
    
    @staticmethod
    def _matrix_multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        """Multiply two matrices: A × B."""
        B_t = [list(col) for col in zip(*B)]  # Transpose B for easier access
        return [
            [sum(a * b for a, b in zip(row_a, col_b)) for col_b in B_t]
            for row_a in A
        ]
    
    @staticmethod
    def _solve_linear_system(A: List[List[float]], b: List[float]) -> List[float]:
        """
        Solve Ax = b using Gaussian elimination with partial pivoting.
        
        This finds x such that when you multiply A by x, you get b.
        In our case: A = (X^T X + alpha*I) and b = X^T y
        """
        n = len(A)
        
        # Create augmented matrix [A | b]
        aug = [row[:] + [b[i]] for i, row in enumerate(A)]
        
        # Forward elimination (turn A into upper triangular form)
        for col in range(n):
            # Find the row with the largest value in this column (pivoting)
            # This improves numerical stability
            max_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
            aug[col], aug[max_row] = aug[max_row], aug[col]
            
            # Get the pivot value
            pivot = aug[col][col]
            if abs(pivot) < 1e-10:
                raise ValueError("Matrix is singular (cannot be inverted)")
            
            # Scale the pivot row so pivot becomes 1
            aug[col] = [v / pivot for v in aug[col]]
            
            # Eliminate this column in all rows below
            for row in range(col + 1, n):
                factor = aug[row][col]
                aug[row] = [aug[row][i] - factor * aug[col][i] for i in range(n + 1)]
        
        # Back substitution (solve from bottom to top)
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            x[i] = aug[i][n] - sum(aug[i][j] * x[j] for j in range(i + 1, n))
        
        return x
    
    # ═════════════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ═════════════════════════════════════════════════════════════════
    
    def get_feature_importance(self, feature_names: List[str]) -> List[tuple[str, float]]:
        """
        Get features ranked by absolute weight magnitude.
        
        Larger absolute weight = more important feature
        Positive weight = increases win chance
        Negative weight = decreases win chance
        
        Args:
            feature_names: Names of features (from BattlePreprocessor.feature_names())
        
        Returns:
            List of (feature_name, weight) tuples, sorted by |weight|
        """
        if not self._is_fitted:
            raise RuntimeError("Must fit model before checking importance")
        
        importance = list(zip(feature_names, self.weights))
        importance.sort(key=lambda x: abs(x[1]), reverse=True)
        return importance


# ═════════════════════════════════════════════════════════════════════
# METRICS: Functions to evaluate model performance
# ═════════════════════════════════════════════════════════════════════

def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    """
    Average squared difference between predictions and actual values.
    
    Lower is better. Perfect prediction = 0.
    """
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)


def root_mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    """
    Square root of MSE. Same units as the target (percentage points).
    
    Interpretation: "On average, predictions are off by X percentage points"
    """
    return math.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true: List[float], y_pred: List[float]) -> float:
    """
    Average absolute difference between predictions and actual values.
    
    More intuitive than MSE: "predictions are off by X% on average"
    """
    return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / len(y_true)


def r_squared(y_true: List[float], y_pred: List[float]) -> float:
    """
    R² (coefficient of determination): proportion of variance explained.
    
    Range: -∞ to 1.0
    - 1.0 = perfect predictions
    - 0.0 = model is no better than predicting the mean
    - < 0 = model is worse than just predicting the mean
    
    Interpretation: R² = 0.85 means the model explains 85% of the variance
    """
    mean_y = sum(y_true) / len(y_true)
    ss_total = sum((yt - mean_y) ** 2 for yt in y_true)
    ss_residual = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    
    if ss_total == 0:
        return 0.0
    
    return 1.0 - (ss_residual / ss_total)