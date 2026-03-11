import random
import math
from typing import List, Tuple

# ═════════════════════════════════════════════════════════════════════
# 1. TRAIN/TEST SPLIT
# ═════════════════════════════════════════════════════════════════════

def train_test_split(
    X: List[List[float]], 
    y: List[float],
    test_size: float = 0.2,
    random_seed: int = 42
) -> Tuple[List[List[float]], List[List[float]], List[float], List[float]]:
    """
    Split data into training and test sets.
    
    Process:
    1. Combine X and y into pairs
    2. Shuffle randomly (so we don't get all easy battles in train, hard in test)
    3. Cut at the test_size percentage
    
    Args:
        X: Feature vectors (list of lists)
        y: Target values (list of floats)
        test_size: Fraction to use for testing (0.2 = 20%)
        random_seed: For reproducibility (same seed = same split every time)
    
    Returns:
        (X_train, X_test, y_train, y_test)
    
    Example:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        print(f"Training on {len(X_train)} battles")
        print(f"Testing on {len(X_test)} battles")
    """
    # Validate input
    if len(X) != len(y):
        raise ValueError(f"X and y must have same length. Got {len(X)} and {len(y)}")
    
    if not (0 < test_size < 1):
        raise ValueError(f"test_size must be between 0 and 1. Got {test_size}")
    
    # Combine X and y so they stay paired during shuffle
    combined = list(zip(X, y))
    
    # Shuffle with a fixed seed (so we get the same split every time we run)
    random.seed(random_seed)
    random.shuffle(combined)
    
    # Calculate where to cut
    n_total = len(combined)
    n_train = int(n_total * (1 - test_size))
    
    # Split
    train_pairs = combined[:n_train]
    test_pairs = combined[n_train:]
    
    # Unzip back into separate X and y
    X_train, y_train = zip(*train_pairs) if train_pairs else ([], [])
    X_test, y_test = zip(*test_pairs) if test_pairs else ([], [])
    
    # Convert back to lists
    return list(X_train), list(X_test), list(y_train), list(y_test)


# ═════════════════════════════════════════════════════════════════════
# 2. STANDARD SCALER (Z-Score Normalization)
# ═════════════════════════════════════════════════════════════════════

class StandardScaler:
    """
    Standardizes features by removing mean and scaling to unit variance.
    
    Formula: z = (x - mean) / std
    
    After scaling:
    - Each feature has mean ≈ 0
    - Each feature has standard deviation ≈ 1
    
    CRITICAL: Always fit on training data only, then transform both train and test!
    
    Usage:
        scaler = StandardScaler()
        
        # Fit on training data
        scaler.fit(X_train)
        
        # Transform both sets
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Or do both at once for training data
        X_train_scaled = scaler.fit_transform(X_train)
    """
    
    def __init__(self):
        self.means: List[float] = []
        self.stds: List[float] = []
        self._is_fitted = False
    
    def fit(self, X: List[List[float]]) -> "StandardScaler":
        """
        Calculate mean and std for each feature from the training data.
        
        Args:
            X: Training data (list of feature vectors)
        
        Returns:
            self (for method chaining)
        """
        if not X:
            raise ValueError("Cannot fit on empty data")
        
        n_samples = len(X)
        n_features = len(X[0])
        
        # Calculate mean for each feature
        self.means = []
        for j in range(n_features):
            feature_sum = sum(X[i][j] for i in range(n_samples))
            mean = feature_sum / n_samples
            self.means.append(mean)
        
        # Calculate standard deviation for each feature
        self.stds = []
        for j in range(n_features):
            variance = sum((X[i][j] - self.means[j]) ** 2 for i in range(n_samples)) / n_samples
            std = math.sqrt(variance)
            
            # Prevent division by zero for constant features
            # (if a feature has the same value everywhere, std=0, so we set it to 1)
            if std < 1e-9:
                std = 1.0
            
            self.stds.append(std)
        
        self._is_fitted = True
        return self
    
    def transform(self, X: List[List[float]]) -> List[List[float]]:
        """
        Apply the standardization to data using the stored mean and std.
        
        Args:
            X: Data to transform (can be train or test)
        
        Returns:
            Scaled version of X
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before transform()")
        
        if not X:
            return []
        
        n_features = len(X[0])
        if n_features != len(self.means):
            raise ValueError(f"X has {n_features} features but scaler was fitted on {len(self.means)}")
        
        # Apply z-score: (x - mean) / std
        X_scaled = []
        for row in X:
            scaled_row = [
                (row[j] - self.means[j]) / self.stds[j]
                for j in range(n_features)
            ]
            X_scaled.append(scaled_row)
        
        return X_scaled
    
    def fit_transform(self, X: List[List[float]]) -> List[List[float]]:
        """
        Convenience method: fit and transform in one call.
        
        Use this ONLY for training data:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)  # just transform, don't fit!
        
        Args:
            X: Training data
        
        Returns:
            Scaled version of X
        """
        self.fit(X)
        return self.transform(X)


# ═════════════════════════════════════════════════════════════════════
# 3. DEMO
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("DEMO: Train/Test Split and Scaling")
    print("=" * 70)
    
    # Create some dummy data
    X = [[i, i*2, i*3] for i in range(100)]  # 100 samples, 3 features each
    y = [i * 10 for i in range(100)]         # targets
    
    print(f"\nOriginal data: {len(X)} samples, {len(X[0])} features")
    print(f"First 3 samples: {X[:3]}")
    
    # Split
    print("\n" + "-" * 70)
    print("SPLITTING (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(f"Train: {len(X_train)} samples")
    print(f"Test:  {len(X_test)} samples")
    
    # Scale
    print("\n" + "-" * 70)
    print("SCALING...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nMeans (from training data):  {[round(m, 2) for m in scaler.means]}")
    print(f"Stds (from training data):   {[round(s, 2) for s in scaler.stds]}")
    
    print(f"\nFirst training sample BEFORE scaling: {X_train[0]}")
    print(f"First training sample AFTER scaling:  {[round(v, 4) for v in X_train_scaled[0]]}")
    
    print(f"\nFirst test sample BEFORE scaling: {X_test[0]}")
    print(f"First test sample AFTER scaling:  {[round(v, 4) for v in X_test_scaled[0]]}")
    
    print("\n" + "=" * 70)
    print("Notice: After scaling, values are centered around 0")
    print("=" * 70)