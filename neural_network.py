import math
import random
from typing import List, Tuple

class NeuralNetwork:
    """
    Flexible Multi-Layer Neural Network for regression.
    
    Architecture:
        Input → Hidden Layer 1 (ReLU) → Hidden Layer 2 (ReLU) → Output (Sigmoid×100)
    
    The network size is configurable - you specify the hidden layer sizes.
    
    Training uses:
    - Gradient descent with Adam optimizer
    - MSE loss (mean squared error)
    - Mini-batch training
    - L2 regularization to prevent overfitting
    """
    
    def __init__(self, input_size: int, hidden_sizes: List[int],
                 learning_rate: float = 0.001, 
                 epochs: int = 300,
                 batch_size: int = 64,
                 regularization: float = 0.001):
        """
        Initialize the neural network.
        
        Args:
            input_size: Number of input features (34 for our problem)
            hidden_sizes: List of hidden layer sizes, e.g. [64, 32] or [48, 24]
            learning_rate: Adam optimizer learning rate
            epochs: Number of training epochs
            batch_size: Mini-batch size
            regularization: L2 regularization strength
        
        Example:
            # Standard architecture: 34 → 64 → 32 → 1
            model = NeuralNetwork(34, [64, 32])
            
            # Conservative architecture: 34 → 48 → 24 → 1
            model = NeuralNetwork(34, [48, 24])
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.reg = regularization
        
        # Build the network architecture
        self._build_network()
        
        # Track training history
        self.train_losses = []
        self.val_losses = []
        
        self._is_fitted = False
    
    # ═════════════════════════════════════════════════════════════════
    # NETWORK INITIALIZATION
    # ═════════════════════════════════════════════════════════════════
    
    def _build_network(self):
        """Initialize weights and biases for all layers using He initialization."""
        random.seed(42)
        
        # Build layer sizes: [input_size, hidden1, hidden2, ..., 1]
        layer_sizes = [self.input_size] + self.hidden_sizes + [1]
        
        # Initialize weights and biases for each layer
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            
            # He initialization: scale = sqrt(2 / fan_in)
            # Works well with ReLU activations
            scale = math.sqrt(2.0 / fan_in)
            
            # Weight matrix: (fan_in × fan_out)
            W = [[random.gauss(0, scale) for _ in range(fan_out)] 
                 for _ in range(fan_in)]
            
            # Bias vector: (fan_out,)
            b = [0.0] * fan_out
            
            self.weights.append(W)
            self.biases.append(b)
        
        # Initialize Adam optimizer state (momentum and velocity for each parameter)
        self._init_adam()
    
    def _init_adam(self):
        """Initialize Adam optimizer momentum and velocity buffers."""
        # First moment (momentum) and second moment (velocity) for weights
        self.m_weights = []
        self.v_weights = []
        self.m_biases = []
        self.v_biases = []
        
        for W, b in zip(self.weights, self.biases):
            # Zero-initialize momentum and velocity
            m_W = [[0.0] * len(W[0]) for _ in range(len(W))]
            v_W = [[0.0] * len(W[0]) for _ in range(len(W))]
            m_b = [0.0] * len(b)
            v_b = [0.0] * len(b)
            
            self.m_weights.append(m_W)
            self.v_weights.append(v_W)
            self.m_biases.append(m_b)
            self.v_biases.append(v_b)
        
        self.adam_t = 0  # Adam timestep
    
    # ═════════════════════════════════════════════════════════════════
    # ACTIVATION FUNCTIONS
    # ═════════════════════════════════════════════════════════════════
    
    @staticmethod
    def _relu(x: float) -> float:
        """ReLU: max(0, x)"""
        return max(0.0, x)
    
    @staticmethod
    def _relu_derivative(x: float) -> float:
        """Derivative of ReLU: 1 if x > 0, else 0"""
        return 1.0 if x > 0 else 0.0
    
    @staticmethod
    def _sigmoid(x: float) -> float:
        """Sigmoid: 1 / (1 + e^(-x))"""
        x = max(min(x, 500), -500)  # Clip to prevent overflow
        return 1.0 / (1.0 + math.exp(-x))
    
    # ═════════════════════════════════════════════════════════════════
    # FORWARD PASS
    # ═════════════════════════════════════════════════════════════════
    
    def _forward(self, x: List[float]) -> Tuple[float, dict]:
        """
        Forward pass through the network.
        
        Args:
            x: Input feature vector
        
        Returns:
            (prediction, cache) where cache stores intermediate values for backprop
        """
        cache = {'activations': [x], 'z_values': []}
        
        current = x
        
        # Pass through all hidden layers (with ReLU)
        for i in range(len(self.weights) - 1):
            # Linear: z = W^T @ x + b
            z = [sum(self.weights[i][j][k] * current[j] for j in range(len(current)))
                 + self.biases[i][k]
                 for k in range(len(self.biases[i]))]
            
            # Activation: a = ReLU(z)
            a = [self._relu(zi) for zi in z]
            
            cache['z_values'].append(z)
            cache['activations'].append(a)
            current = a
        
        # Final layer: sigmoid output (scaled to 0-100%)
        # Linear
        z_final = sum(self.weights[-1][j][0] * current[j] for j in range(len(current)))
        z_final += self.biases[-1][0]
        
        # Sigmoid × 100
        output = self._sigmoid(z_final) * 100.0
        
        cache['z_values'].append([z_final])
        cache['output'] = output
        
        return output, cache
    
    # ═════════════════════════════════════════════════════════════════
    # BACKWARD PASS (BACKPROPAGATION)
    # ═════════════════════════════════════════════════════════════════
    
    def _backward(self, cache: dict, y_true: float) -> Tuple[List, List]:
        """
        Backward pass to compute gradients.
        
        Args:
            cache: Cached values from forward pass
            y_true: True target value (0-100%)
        
        Returns:
            (weight_gradients, bias_gradients)
        """
        n_layers = len(self.weights)
        weight_grads = [None] * n_layers
        bias_grads = [None] * n_layers
        
        # Output layer gradient
        # Loss = (prediction - target)^2
        # dL/d(output) = 2 * (output - target)
        output = cache['output']
        d_output = 2.0 * (output - y_true)
        
        # output = sigmoid(z) * 100
        # d(output)/dz = 100 * sigmoid'(z) = 100 * sigmoid(z) * (1 - sigmoid(z))
        z_final = cache['z_values'][-1][0]
        sig = self._sigmoid(z_final)
        d_z = d_output * 100.0 * sig * (1.0 - sig)
        
        # Gradient w.r.t. final layer weights and biases
        prev_activation = cache['activations'][-2]
        weight_grads[-1] = [[d_z * prev_activation[j]] for j in range(len(prev_activation))]
        bias_grads[-1] = [d_z]
        
        # Backpropagate through hidden layers
        d_a = [self.weights[-1][j][0] * d_z for j in range(len(self.weights[-1]))]
        
        for layer in range(n_layers - 2, -1, -1):
            # ReLU derivative
            z = cache['z_values'][layer]
            d_z = [d_a[i] * self._relu_derivative(z[i]) for i in range(len(z))]
            
            # Gradients for this layer
            prev_activation = cache['activations'][layer]
            W_grad = [[d_z[k] * prev_activation[j] 
                      for k in range(len(d_z))]
                     for j in range(len(prev_activation))]
            b_grad = list(d_z)
            
            weight_grads[layer] = W_grad
            bias_grads[layer] = b_grad
            
            # Propagate to previous layer
            if layer > 0:
                d_a = [sum(self.weights[layer][j][k] * d_z[k] 
                          for k in range(len(d_z)))
                      for j in range(len(self.weights[layer]))]
        
        return weight_grads, bias_grads
    
    # ═════════════════════════════════════════════════════════════════
    # ADAM OPTIMIZER
    # ═════════════════════════════════════════════════════════════════
    
    def _adam_update(self, weight_grads: List, bias_grads: List, batch_size: int):
        """
        Update weights using Adam optimizer.
        
        Adam combines:
        - Momentum (moving average of gradients)
        - RMSprop (moving average of squared gradients)
        - Bias correction for the moving averages
        """
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        self.adam_t += 1
        
        for layer in range(len(self.weights)):
            # Average gradients over batch
            W_grad = weight_grads[layer]
            b_grad = bias_grads[layer]
            
            for i in range(len(W_grad)):
                for j in range(len(W_grad[i])):
                    W_grad[i][j] /= batch_size
            for i in range(len(b_grad)):
                b_grad[i] /= batch_size
            
            # Add L2 regularization gradient
            for i in range(len(self.weights[layer])):
                for j in range(len(self.weights[layer][i])):
                    W_grad[i][j] += 2 * self.reg * self.weights[layer][i][j]
            
            # Update weights with Adam
            for i in range(len(self.weights[layer])):
                for j in range(len(self.weights[layer][i])):
                    g = W_grad[i][j]
                    
                    # Update momentum and velocity
                    self.m_weights[layer][i][j] = beta1 * self.m_weights[layer][i][j] + (1 - beta1) * g
                    self.v_weights[layer][i][j] = beta2 * self.v_weights[layer][i][j] + (1 - beta2) * g**2
                    
                    # Bias correction
                    m_hat = self.m_weights[layer][i][j] / (1 - beta1**self.adam_t)
                    v_hat = self.v_weights[layer][i][j] / (1 - beta2**self.adam_t)
                    
                    # Update weight
                    self.weights[layer][i][j] -= self.lr * m_hat / (math.sqrt(v_hat) + eps)
            
            # Update biases with Adam
            for i in range(len(self.biases[layer])):
                g = b_grad[i]
                
                self.m_biases[layer][i] = beta1 * self.m_biases[layer][i] + (1 - beta1) * g
                self.v_biases[layer][i] = beta2 * self.v_biases[layer][i] + (1 - beta2) * g**2
                
                m_hat = self.m_biases[layer][i] / (1 - beta1**self.adam_t)
                v_hat = self.v_biases[layer][i] / (1 - beta2**self.adam_t)
                
                self.biases[layer][i] -= self.lr * m_hat / (math.sqrt(v_hat) + eps)
    
    # ═════════════════════════════════════════════════════════════════
    # TRAINING
    # ═════════════════════════════════════════════════════════════════
    
    def fit(self, X: List[List[float]], y: List[float], 
            X_val: List[List[float]] = None, y_val: List[float] = None,
            verbose: bool = True) -> "NeuralNetwork":
        """
        Train the network.
        
        Args:
            X: Training features
            y: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            verbose: Print progress
        
        Returns:
            self
        """
        n_samples = len(X)
        
        # Mark as fitted immediately so we can do validation predictions
        self._is_fitted = True
        
        for epoch in range(self.epochs):
            # Shuffle data
            indices = list(range(n_samples))
            random.shuffle(indices)
            
            epoch_loss = 0.0
            
            # Mini-batch training
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch_indices = indices[start:end]
                actual_batch_size = len(batch_indices)
                
                # Accumulate gradients over batch
                batch_weight_grads = None
                batch_bias_grads = None
                batch_loss = 0.0
                
                for idx in batch_indices:
                    # Forward pass
                    pred, cache = self._forward(X[idx])
                    
                    # Loss
                    error = pred - y[idx]
                    batch_loss += error ** 2
                    
                    # Backward pass
                    w_grads, b_grads = self._backward(cache, y[idx])
                    
                    # Accumulate
                    if batch_weight_grads is None:
                        batch_weight_grads = [[row[:] for row in layer] for layer in w_grads]
                        batch_bias_grads = [b[:] for b in b_grads]
                    else:
                        for l in range(len(w_grads)):
                            for i in range(len(w_grads[l])):
                                for j in range(len(w_grads[l][i])):
                                    batch_weight_grads[l][i][j] += w_grads[l][i][j]
                            for i in range(len(b_grads[l])):
                                batch_bias_grads[l][i] += b_grads[l][i]
                
                epoch_loss += batch_loss
                
                # Update weights
                self._adam_update(batch_weight_grads, batch_bias_grads, actual_batch_size)
            
            # Track losses
            avg_loss = epoch_loss / n_samples
            self.train_losses.append(avg_loss)
            
            # Validation loss
            if X_val is not None and y_val is not None:
                val_preds = self.predict(X_val)
                val_loss = sum((p - t)**2 for p, t in zip(val_preds, y_val)) / len(y_val)
                self.val_losses.append(val_loss)
            
            # Print progress
            if verbose and (epoch % 50 == 0 or epoch == self.epochs - 1):
                rmse = math.sqrt(avg_loss)
                if X_val is not None:
                    val_rmse = math.sqrt(val_loss)
                    print(f"  Epoch {epoch+1:>3}/{self.epochs}   Train RMSE={rmse:>6.2f}%   Val RMSE={val_rmse:>6.2f}%")
                else:
                    print(f"  Epoch {epoch+1:>3}/{self.epochs}   Train RMSE={rmse:>6.2f}%")
        
        return self
    
    # ═════════════════════════════════════════════════════════════════
    # PREDICTION
    # ═════════════════════════════════════════════════════════════════
    
    def predict(self, X: List[List[float]]) -> List[float]:
        """Make predictions on new data."""
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before predict()")
        
        return [self._forward(x)[0] for x in X]
    
    # ═════════════════════════════════════════════════════════════════
    # UTILITY
    # ═════════════════════════════════════════════════════════════════
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        total = 0
        for W, b in zip(self.weights, self.biases):
            total += len(W) * len(W[0])  # weights
            total += len(b)               # biases
        return total
    
    def get_architecture_summary(self) -> str:
        """Get a string summary of the architecture."""
        layers = [self.input_size] + self.hidden_sizes + [1]
        arch_str = " → ".join(str(l) for l in layers)
        params = self.count_parameters()
        return f"{arch_str}  ({params:,} parameters)"


# ═════════════════════════════════════════════════════════════════════
# METRICS (same as before)
# ═════════════════════════════════════════════════════════════════════

def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)

def root_mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))

def mean_absolute_error(y_true: List[float], y_pred: List[float]) -> float:
    return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / len(y_true)

def r_squared(y_true: List[float], y_pred: List[float]) -> float:
    mean_y = sum(y_true) / len(y_true)
    ss_total = sum((yt - mean_y) ** 2 for yt in y_true)
    ss_residual = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    return 1.0 - (ss_residual / ss_total) if ss_total != 0 else 0.0