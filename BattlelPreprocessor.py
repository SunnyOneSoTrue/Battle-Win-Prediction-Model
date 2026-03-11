import math
from typing import List, Tuple, Dict

class BattlePreprocessor:
    """
    Converts raw CSV data from CSVParser into model-ready numerical features.
    
    Responsibilities:
    - Extract the header and separate it from data rows
    - Convert string values to floats
    - Select only the columns the model needs (ratios + categoricals)
    - Log-transform ratio features (because the underlying model uses log-space)
    - One-hot encode categorical features
    
    Usage:
        from csv_parser import CSVParser
        from battle_preprocessor import BattlePreprocessor
        
        parser = CSVParser("battle_dataset_10000.csv")
        preprocessor = BattlePreprocessor(parser.fetch_all())
        X, y = preprocessor.preprocess()
        
        # X is now a list of 10000 feature vectors (each with 34 floats)
        # y is a list of 10000 target values (win_chance_% as floats)
    """
    
    # ═════════════════════════════════════════════════════════════════
    # CLASS CONSTANTS: Define which columns to use and how
    # ═════════════════════════════════════════════════════════════════
    
    # The 9 ratio columns we'll feed into the model (after log transform)
    RATIO_COLS = [
        "army_size_ratio (enemy/own)",
        "training_level_ratio (enemy/own)",
        "equipment_prestige_ratio (enemy/own)",
        "supplies_ratio (enemy/own)",
        "morale_ratio (enemy/own)",
        "fatigue_ratio (enemy/own)",
        "commander_skill_ratio (enemy/own)",
        "command_efficiency_ratio (enemy/own)",
        "technology_level_ratio (enemy/own)",
    ]
    
    # Categorical columns and their possible values (order matters for one-hot!)
    # Each column will expand into len(values) binary features
    CATEGORICALS: Dict[str, List[str]] = {
        "weather_condition": [
            "Clear", "Cloudy", "Rain", "Fog", "Storm"
        ],
        "visibility": [
            "Very Low", "Low", "Moderate", "Good", "Excellent"
        ],
        "time_of_day": [
            "Dawn", "Morning", "Midday", "Afternoon", "Dusk", "Night"
        ],
        "own_strategic_posture": [
            "Defend", "Attack"
        ],
        "enemy_strategic_posture": [
            "Defend", "Attack"
        ],
        "surprise_ambush_factor": [
            "None", "Partial", "Full Ambush"
        ],
        "home_territory_advantage": [
            "No", "Yes"
        ],
    }
    
    # The target column (what we're trying to predict)
    TARGET_COL = "win_chance_%"
    
    # ═════════════════════════════════════════════════════════════════
    # CONSTRUCTOR
    # ═════════════════════════════════════════════════════════════════
    
    def __init__(self, raw_data: List[Tuple]):
        """
        Initialize the preprocessor with raw CSV data.
        
        Args:
            raw_data: Output from CSVParser.fetch_all()
                     Row 0 must be the header
                     Rows 1+ are the actual data
        """
        if len(raw_data) < 2:
            raise ValueError("raw_data must have at least a header and one data row")
        
        # Separate header from data
        self.headers = list(raw_data[0])
        self.rows = raw_data[1:]
        
        # Build a lookup dict: column_name → column_index
        # This lets us do: row[self._col_idx["weather_condition"]]
        self._col_idx = {name: i for i, name in enumerate(self.headers)}
        
        # Validate that all required columns exist
        self._validate_columns()
    
    def _validate_columns(self):
        """Check that all required columns are present in the CSV."""
        missing = []
        
        # Check ratio columns
        for col in self.RATIO_COLS:
            if col not in self._col_idx:
                missing.append(col)
        
        # Check categorical columns
        for col in self.CATEGORICALS.keys():
            if col not in self._col_idx:
                missing.append(col)
        
        # Check target column
        if self.TARGET_COL not in self._col_idx:
            missing.append(self.TARGET_COL)
        
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    # ═════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═════════════════════════════════════════════════════════════════
    
    def preprocess(self) -> Tuple[List[List[float]], List[float]]:
        """
        Convert all rows into (X, y) for training/testing.
        
        Returns:
            X: List of feature vectors (each is a list of 34 floats)
            y: List of target values (win_chance_% as floats)
        
        Example:
            X, y = preprocessor.preprocess()
            print(f"Got {len(X)} samples with {len(X[0])} features each")
        """
        X = []
        y = []
        
        for row in self.rows:
            # Extract features for this battle
            features = self._extract_features(row)
            X.append(features)
            
            # Extract target
            target = float(row[self._col_idx[self.TARGET_COL]])
            y.append(target)
        
        return X, y
    
    # ═════════════════════════════════════════════════════════════════
    # INTERNAL HELPER: Feature Extraction
    # ═════════════════════════════════════════════════════════════════
    
    def _extract_features(self, row: Tuple) -> List[float]:
        """
        Convert one row into a feature vector.
        
        Process:
        1. Extract 9 ratio columns and take log() of each
        2. One-hot encode 7 categorical columns (→ 24 binary features)
        
        Total: 9 + 24 = 33 features
        
        Args:
            row: A single data row (tuple of strings from CSV)
        
        Returns:
            A list of 34 floats representing this battle
        """
        feature_vector = []
        
        # ─────────────────────────────────────────────────────────────
        # STEP 1: Log-transform ratio features
        # 
        # Why log?
        # - The data was generated with a sigmoid model using log-ratios
        # - log(1.0) = 0 → balanced forces
        # - log(2.0) = 0.69 → enemy is 2x stronger
        # - log(0.5) = -0.69 → we are 2x stronger
        # - It's symmetric around zero and captures multiplicative effects
        # ─────────────────────────────────────────────────────────────
        for col_name in self.RATIO_COLS:
            raw_value = float(row[self._col_idx[col_name]])
            
            # Guard against log(0) by using max with a tiny epsilon
            # (ratios should never be 0 in practice, but safety first)
            log_value = math.log(max(raw_value, 1e-9))
            
            feature_vector.append(log_value)
        
        # ─────────────────────────────────────────────────────────────
        # STEP 2: One-hot encode categorical features
        # 
        # For each categorical column:
        #   - Get the actual value in this row (e.g. "Clear")
        #   - Create one binary feature for each possible value
        #   - Set the matching one to 1.0, all others to 0.0
        #
        # Example for weather_condition = "Rain":
        #   weather_Clear  = 0.0
        #   weather_Cloudy = 0.0
        #   weather_Rain   = 1.0  ← only this one is "hot"
        #   weather_Fog    = 0.0
        #   weather_Storm  = 0.0
        # ─────────────────────────────────────────────────────────────
        for col_name, possible_values in self.CATEGORICALS.items():
            # Get the actual value from this row
            actual_value = row[self._col_idx[col_name]]
            
            # Create one binary feature for each possible value
            for possible_value in possible_values:
                if actual_value == possible_value:
                    feature_vector.append(1.0)  # "hot" (on)
                else:
                    feature_vector.append(0.0)  # "cold" (off)
        
        return feature_vector
    
    # ═════════════════════════════════════════════════════════════════
    # UTILITY: Human-readable feature names
    # ═════════════════════════════════════════════════════════════════
    
    @staticmethod
    def feature_names() -> List[str]:
        """
        Generate human-readable names for all features in order.
        
        Useful for:
        - Debugging: seeing which feature is which
        - Interpretation: understanding model weights
        - Visualization: labeling plots
        
        Returns:
            List of 34 feature names matching the order in feature vectors
        
        Example:
            names = BattlePreprocessor.feature_names()
            for i, name in enumerate(names):
                print(f"Feature {i}: {name}")
        """
        names = []
        
        # Log-transformed ratio features (9 total)
        for col in BattlePreprocessor.RATIO_COLS:
            names.append(f"log({col})")
        
        # One-hot encoded categorical features (24 total)
        for col, values in BattlePreprocessor.CATEGORICALS.items():
            for value in values:
                names.append(f"{col}={value}")
        
        return names
    
    # ═════════════════════════════════════════════════════════════════
    # UTILITY: Get number of features
    # ═════════════════════════════════════════════════════════════════
    
    @staticmethod
    def num_features() -> int:
        """
        Returns the total number of features in the output.
        
        This is useful when initializing models that need to know
        the input dimension ahead of time.
        
        Returns:
            34 (9 log-ratios + 24 one-hot encoded + 1 for balance)
        """
        num_ratios = len(BattlePreprocessor.RATIO_COLS)
        num_categorical = sum(len(values) for values in BattlePreprocessor.CATEGORICALS.values())
        return num_ratios + num_categorical