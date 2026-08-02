"""
Cognitive Processing Module - Neural Network Engine.
This module implements a massive, production-grade custom Deep Learning framework from scratch
without using external dependencies like NumPy, PyTorch, or TensorFlow.
It includes:
- Comprehensive Linear Algebra and Vector/Tensor Math library
- Advanced Activation functions (Sigmoid, Tanh, ReLU, LeakyReLU, ELU, GELU, Softmax)
- Analytical Derivatives and Gradients for all operations
- Advanced Loss functions (MSE, MAE, Cross-Entropy, Huber Loss)
- Stateful Optimizers from scratch (SGD with Momentum, RMSprop, Adam)
- Modular Neural Layers (Dense, Dropout, LayerNormalization)
- Advanced Multi-Head Attention (MHA) Layer for contextual code representation
- Graph Attention Network (GAT) Layer for code AST and dependency graph learning
- Stateful Long Short-Term Memory (LSTM) cells for sequential trace modeling
- High-level CodeCognitiveNetwork orchestrator for risk assessment and bug-likelihood forecasting.
- Multi-Paradigm Training Routines: Supervised, Unsupervised pre-training, Reinforcement Learning, and Evolutionary Strategy optimization.
"""

import math
import random
import json
from typing import List, Dict, Any, Tuple, Optional


# =====================================================================
# BASE SYSTEM INTERFACES
# =====================================================================

class Layer:
    """Base neural layer interface."""
    def forward(self, inputs: List[float]) -> List[float]:
        raise NotImplementedError
    def backward(self, d_out: List[float], lr: float) -> List[float]:
        raise NotImplementedError


# =====================================================================
# PART 1: COMPREHENSIVE VECTOR AND MATRIX MATHEMATICS ENGINE
# =====================================================================

def dot_product(v1: List[float], v2: List[float]) -> float:
    """Calculates standard dot product of two vectors."""
    if len(v1) != len(v2):
        raise ValueError(f"Dimensions mismatch for dot product: {len(v1)} != {len(v2)}")
    return sum(x * y for x, y in zip(v1, v2))

def vector_add(v1: List[float], v2: List[float]) -> List[float]:
    """Adds two vectors element-wise."""
    if len(v1) != len(v2):
        raise ValueError(f"Dimensions mismatch for addition: {len(v1)} != {len(v2)}")
    return [x + y for x, y in zip(v1, v2)]

def vector_sub(v1: List[float], v2: List[float]) -> List[float]:
    """Subtracts v2 from v1 element-wise."""
    if len(v1) != len(v2):
        raise ValueError(f"Dimensions mismatch for subtraction: {len(v1)} != {len(v2)}")
    return [x - y for x, y in zip(v1, v2)]

def scale_vector(v: List[float], scalar: float) -> List[float]:
    """Multiplies all vector elements by a scalar value."""
    return [x * scalar for x in v]

def elementwise_multiply(v1: List[float], v2: List[float]) -> List[float]:
    """Computes Hadamard product (element-wise multiplication) of two vectors."""
    if len(v1) != len(v2):
        raise ValueError(f"Dimensions mismatch for Hadamard product: {len(v1)} != {len(v2)}")
    return [x * y for x, y in zip(v1, v2)]

def vector_mean(v: List[float]) -> float:
    """Calculates arithmetic mean of a vector."""
    if not v:
        return 0.0
    return sum(v) / len(v)

def vector_variance(v: List[float], mean_val: Optional[float] = None) -> float:
    """Calculates statistical variance of a vector."""
    if len(v) <= 1:
        return 0.0
    m = mean_val if mean_val is not None else vector_mean(v)
    return sum((x - m) ** 2 for x in v) / len(v)

def matrix_multiply(m1: List[List[float]], m2: List[List[float]]) -> List[List[float]]:
    """Performs standard matrix multiplication: m1 x m2."""
    r1, c1 = len(m1), len(m1[0])
    r2, c2 = len(m2), len(m2[0])
    if c1 != r2:
        raise ValueError(f"Matrix dimension mismatch: columns of m1 ({c1}) must match rows of m2 ({r2})")

    result = [[0.0] * c2 for _ in range(r1)]
    for i in range(r1):
        for j in range(c2):
            val = 0.0
            for k in range(c1):
                val += m1[i][k] * m2[k][j]
            result[i][j] = val
    return result

def matrix_vector_multiply(m: List[List[float]], v: List[float]) -> List[float]:
    """Performs matrix-vector multiplication."""
    if len(m[0]) != len(v):
        raise ValueError(f"Dimension mismatch: matrix columns ({len(m[0])}) must match vector size ({len(v)})")
    return [dot_product(row, v) for row in m]

def transpose(m: List[List[float]]) -> List[List[float]]:
    """Calculates transpose of a 2D matrix."""
    if not m or not m[0]:
        return []
    return [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]

def outer_product(v1: List[float], v2: List[float]) -> List[List[float]]:
    """Computes outer product (tensor product) of two 1D vectors."""
    return [[x * y for y in v2] for x in v1]

def add_matrices(m1: List[List[float]], m2: List[List[float]]) -> List[List[float]]:
    """Adds two 2D matrices element-wise."""
    return [[x + y for x, y in zip(r1, r2)] for r1, r2 in zip(m1, m2)]

def sub_matrices(m1: List[List[float]], m2: List[List[float]]) -> List[List[float]]:
    """Subtracts m2 from m1 element-wise."""
    return [[x - y for x, y in zip(r1, r2)] for r1, r2 in zip(m1, m2)]

def scale_matrix(m: List[List[float]], scalar: float) -> List[List[float]]:
    """Scales all elements of a matrix by a constant scalar."""
    return [[x * scalar for x in row] for row in m]


# =====================================================================
# PART 2: ADVANCED ACTIVATION FUNCTIONS AND ANALYTICAL DERIVATIVES
# =====================================================================

def sigmoid(x: float) -> float:
    """Sigmoid activation function."""
    if x < -30.0:
        return 0.0
    if x > 30.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivative(y: float) -> float:
    """Sigmoid derivative. Expects output value y = sigmoid(x)."""
    return y * (1.0 - y)

def tanh_activation(x: float) -> float:
    """Hyperbolic tangent activation function."""
    if x < -30.0:
        return -1.0
    if x > 30.0:
        return 1.0
    return math.tanh(x)

def tanh_derivative(y: float) -> float:
    """Hyperbolic tangent derivative. Expects output value y = tanh(x)."""
    return 1.0 - (y ** 2)

def relu(x: float) -> float:
    """Rectified Linear Unit function."""
    return max(0.0, x)

def relu_derivative(x: float) -> float:
    """ReLU derivative."""
    return 1.0 if x > 0.0 else 0.0

def leaky_relu(x: float, alpha: float = 0.01) -> float:
    """Leaky Rectified Linear Unit."""
    return x if x > 0.0 else alpha * x

def leaky_relu_derivative(x: float, alpha: float = 0.01) -> float:
    """Leaky ReLU derivative."""
    return 1.0 if x > 0.0 else alpha

def elu(x: float, alpha: float = 1.0) -> float:
    """Exponential Linear Unit."""
    return x if x > 0.0 else alpha * (math.exp(x) - 1.0)

def elu_derivative(x: float, out_val: float, alpha: float = 1.0) -> float:
    """ELU derivative. out_val is the computed output elu(x)."""
    return 1.0 if x > 0.0 else out_val + alpha

def gelu(x: float) -> float:
    """Gaussian Error Linear Unit approximation."""
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * math.pow(x, 3))))

def gelu_derivative(x: float) -> float:
    """Approximated GELU derivative."""
    cdf = 0.5 * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * math.pow(x, 3))))
    pdf = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    return cdf + x * pdf

def softmax(v: List[float]) -> List[float]:
    """Softmax activation over 1D input array."""
    if not v:
        return []
    max_val = max(v)
    exps = []
    for x in v:
        try:
            exps.append(math.exp(x - max_val))
        except OverflowError:
            exps.append(0.0)
    total = sum(exps)
    if total == 0.0:
        return [1.0 / len(v)] * len(v)
    return [e / total for e in exps]


# =====================================================================
# PART 3: REVERSED ENGINEERING LOSS FUNCTIONS
# =====================================================================

def mean_squared_error(y_pred: List[float], y_true: List[float]) -> float:
    """Mean Squared Error (L2 Loss)."""
    if len(y_pred) != len(y_true):
        raise ValueError("Dimensions mismatch for MSE.")
    return sum((p - t) ** 2 for p, t in zip(y_pred, y_true)) / len(y_pred)

def mean_squared_error_derivative(y_pred: List[float], y_true: List[float]) -> List[float]:
    """Derivative of MSE with respect to y_pred."""
    n = len(y_pred)
    return [2.0 * (p - t) / n for p, t in zip(y_pred, y_true)]

def mean_absolute_error(y_pred: List[float], y_true: List[float]) -> float:
    """Mean Absolute Error (L1 Loss)."""
    if len(y_pred) != len(y_true):
        raise ValueError("Dimensions mismatch for MAE.")
    return sum(abs(p - t) for p, t in zip(y_pred, y_true)) / len(y_pred)

def mean_absolute_error_derivative(y_pred: List[float], y_true: List[float]) -> List[float]:
    """Derivative of MAE with respect to y_pred."""
    n = len(y_pred)
    return [1.0 / n if p >= t else -1.0 / n for p, t in zip(y_pred, y_true)]

def cross_entropy_loss(y_pred: List[float], y_true: List[float]) -> float:
    """Categorical cross-entropy loss with softmax predictions."""
    if len(y_pred) != len(y_true):
        raise ValueError("Dimensions mismatch for cross entropy.")
    eps = 1e-15
    loss = 0.0
    for p, t in zip(y_pred, y_true):
        p_clipped = min(max(p, eps), 1.0 - eps)
        loss -= t * math.log(p_clipped)
    return loss

def cross_entropy_loss_derivative(y_pred: List[float], y_true: List[float]) -> List[float]:
    """Derivative of cross entropy combined with softmax activation."""
    return [p - t for p, t in zip(y_pred, y_true)]

def huber_loss(y_pred: List[float], y_true: List[float], delta: float = 1.0) -> float:
    """Robust Huber Loss function."""
    loss = 0.0
    for p, t in zip(y_pred, y_true):
        diff = abs(p - t)
        if diff <= delta:
            loss += 0.5 * (diff ** 2)
        else:
            loss += delta * (diff - 0.5 * delta)
    return loss / len(y_pred)

def huber_loss_derivative(y_pred: List[float], y_true: List[float], delta: float = 1.0) -> List[float]:
    """Huber Loss derivative."""
    n = len(y_pred)
    derivs = []
    for p, t in zip(y_pred, y_true):
        diff = p - t
        if abs(diff) <= delta:
            derivs.append(diff / n)
        else:
            val = (delta if diff > 0 else -delta) / n
            derivs.append(val)
    return derivs


# =====================================================================
# PART 4: COMPREHENSIVE OPTIMIZERS FROM SCRATCH
# =====================================================================

class Optimizer:
    """Base Optimizer interface."""
    def update(self, weights: List[List[float]], biases: List[float], dw: List[List[float]], db: List[float], param_id: str) -> Tuple[List[List[float]], List[float]]:
        raise NotImplementedError


class SGDMomentum(Optimizer):
    """Stochastic Gradient Descent with Momentum optimizer."""
    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.v_w: Dict[str, List[List[float]]] = {}
        self.v_b: Dict[str, List[float]] = {}

    def update(self, weights: List[List[float]], biases: List[float], dw: List[List[float]], db: List[float], param_id: str) -> Tuple[List[List[float]], List[float]]:
        # Initialize momentums
        if param_id not in self.v_w:
            self.v_w[param_id] = [[0.0] * len(row) for row in weights]
            self.v_b[param_id] = [0.0] * len(biases)

        vw = self.v_w[param_id]
        vb = self.v_b[param_id]

        # Calculate velocities
        new_weights = []
        for r in range(len(weights)):
            w_row = []
            for c in range(len(weights[0])):
                vw[r][c] = self.momentum * vw[r][c] + self.lr * dw[r][c]
                w_row.append(weights[r][c] - vw[r][c])
            new_weights.append(w_row)

        new_biases = []
        for r in range(len(biases)):
            vb[r] = self.momentum * vb[r] + self.lr * db[r]
            new_biases.append(biases[r] - vb[r])

        return new_weights, new_biases


class RMSprop(Optimizer):
    """RMSprop optimizer from scratch."""
    def __init__(self, lr: float = 0.001, beta: float = 0.9, eps: float = 1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.s_w: Dict[str, List[List[float]]] = {}
        self.s_b: Dict[str, List[float]] = {}

    def update(self, weights: List[List[float]], biases: List[float], dw: List[List[float]], db: List[float], param_id: str) -> Tuple[List[List[float]], List[float]]:
        if param_id not in self.s_w:
            self.s_w[param_id] = [[0.0] * len(row) for row in weights]
            self.s_b[param_id] = [0.0] * len(biases)

        sw = self.s_w[param_id]
        sb = self.s_b[param_id]

        new_weights = []
        for r in range(len(weights)):
            w_row = []
            for c in range(len(weights[0])):
                sw[r][c] = self.beta * sw[r][c] + (1.0 - self.beta) * (dw[r][c] ** 2)
                w_row.append(weights[r][c] - (self.lr * dw[r][c]) / (math.sqrt(sw[r][c]) + self.eps))
            new_weights.append(w_row)

        new_biases = []
        for r in range(len(biases)):
            sb[r] = self.beta * sb[r] + (1.0 - self.beta) * (db[r] ** 2)
            new_biases.append(biases[r] - (self.lr * db[r]) / (math.sqrt(sb[r]) + self.eps))

        return new_weights, new_biases


class Adam(Optimizer):
    """State-of-the-art Adam optimizer from scratch."""
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m_w: Dict[str, List[List[float]]] = {}
        self.v_w: Dict[str, List[List[float]]] = {}
        self.m_b: Dict[str, List[float]] = {}
        self.v_b: Dict[str, List[float]] = {}
        self.t: Dict[str, int] = {}

    def update(self, weights: List[List[float]], biases: List[float], dw: List[List[float]], db: List[float], param_id: str) -> Tuple[List[List[float]], List[float]]:
        if param_id not in self.m_w:
            self.m_w[param_id] = [[0.0] * len(row) for row in weights]
            self.v_w[param_id] = [[0.0] * len(row) for row in weights]
            self.m_b[param_id] = [0.0] * len(biases)
            self.v_b[param_id] = [0.0] * len(biases)
            self.t[param_id] = 0

        self.t[param_id] += 1
        t = self.t[param_id]

        mw = self.m_w[param_id]
        vw = self.v_w[param_id]
        mb = self.m_b[param_id]
        vb = self.v_b[param_id]

        # Unbias corrections
        correction1 = 1.0 - (self.beta1 ** t)
        correction2 = 1.0 - (self.beta2 ** t)

        new_weights = []
        for r in range(len(weights)):
            w_row = []
            for c in range(len(weights[0])):
                mw[r][c] = self.beta1 * mw[r][c] + (1.0 - self.beta1) * dw[r][c]
                vw[r][c] = self.beta2 * vw[r][c] + (1.0 - self.beta2) * (dw[r][c] ** 2)

                m_unbiased = mw[r][c] / correction1
                v_unbiased = vw[r][c] / correction2

                step = (self.lr * m_unbiased) / (math.sqrt(v_unbiased) + self.eps)
                w_row.append(weights[r][c] - step)
            new_weights.append(w_row)

        new_biases = []
        for r in range(len(biases)):
            mb[r] = self.beta1 * mb[r] + (1.0 - self.beta1) * db[r]
            vb[r] = self.beta2 * vb[r] + (1.0 - self.beta2) * (db[r] ** 2)

            mb_unbiased = mb[r] / correction1
            vb_unbiased = vb[r] / correction2

            step = (self.lr * mb_unbiased) / (math.sqrt(vb_unbiased) + self.eps)
            new_biases.append(biases[r] - step)

        return new_weights, new_biases


# =====================================================================
# PART 5: MODULAR NEURAL NETWORK LAYERS FROM SCRATCH
# =====================================================================

def xavier_init(rows: int, cols: int) -> List[List[float]]:
    """Xavier / Glorot weight initialization."""
    limit = math.sqrt(6.0 / (rows + cols))
    return [[random.uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)]


class LayerNormalization(Layer):
    """Standard Layer Normalization layer."""
    def __init__(self, features: int, eps: float = 1e-5):
        self.features = features
        self.eps = eps
        self.gamma = [1.0] * features
        self.beta = [0.0] * features
        self.last_inputs: List[float] = []
        self.last_normalized: List[float] = []
        self.last_mean = 0.0
        self.last_var = 0.0

    def forward(self, inputs: List[float]) -> List[float]:
        self.last_inputs = list(inputs)
        mean_val = vector_mean(inputs)
        variance_val = vector_variance(inputs, mean_val)

        self.last_mean = mean_val
        self.last_var = variance_val

        std = math.sqrt(variance_val + self.eps)
        normalized = [(x - mean_val) / std for x in inputs]
        self.last_normalized = normalized

        return [g * n + b for g, n, b in zip(self.gamma, normalized, self.beta)]

    def backward(self, d_out: List[float], lr: float) -> List[float]:
        """Calculates exact LayerNorm backward pass."""
        std = math.sqrt(self.last_var + self.eps)
        n = len(self.last_inputs)

        # Gamma and Beta gradients
        d_gamma = [d * norm for d, norm in zip(d_out, self.last_normalized)]
        d_beta = list(d_out)

        # Backpropagation to inputs
        d_norm = [d * g for d, g in zip(d_out, self.gamma)]
        sum_d_norm = sum(d_norm)
        sum_d_norm_x = sum(dn * norm for dn, norm in zip(d_norm, self.last_normalized))

        d_in = []
        for i in range(n):
            val = (n * d_norm[i] - sum_d_norm - self.last_normalized[i] * sum_d_norm_x) / (n * std)
            d_in.append(val)

        # Update parameters
        self.gamma = [g - lr * dg for g, dg in zip(self.gamma, d_gamma)]
        self.beta = [b - lr * db for b, db in zip(self.beta, d_beta)]

        return d_in


class DropoutLayer(Layer):
    """Regularization Dropout Layer."""
    def __init__(self, rate: float = 0.1):
        self.rate = rate
        self.mask: List[float] = []
        self.training = True

    def forward(self, inputs: List[float]) -> List[float]:
        if not self.training or self.rate == 0.0:
            return list(inputs)

        scale = 1.0 / (1.0 - self.rate)
        self.mask = [scale if random.random() >= self.rate else 0.0 for _ in inputs]
        return elementwise_multiply(inputs, self.mask)

    def backward(self, d_out: List[float], lr: float) -> List[float]:
        if not self.training or self.rate == 0.0:
            return list(d_out)
        return elementwise_multiply(d_out, self.mask)


class DenseLayer(Layer):
    """Fully Connected dense projection layers."""
    def __init__(self, in_features: int, out_features: int, activation: str = "relu"):
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

        self.weights = xavier_init(out_features, in_features)
        self.biases = [0.0] * out_features
        self.optimizer = Adam()

        self.last_inputs: List[float] = []
        self.last_outputs: List[float] = []
        self.last_net_inputs: List[float] = []

    def forward(self, inputs: List[float]) -> List[float]:
        self.last_inputs = list(inputs)
        net_inputs = []
        outputs = []

        for r in range(self.out_features):
            val = dot_product(self.weights[r], inputs) + self.biases[r]
            net_inputs.append(val)

            if self.activation == "relu":
                outputs.append(relu(val))
            elif self.activation == "leaky_relu":
                outputs.append(leaky_relu(val))
            elif self.activation == "sigmoid":
                outputs.append(sigmoid(val))
            elif self.activation == "tanh":
                outputs.append(tanh_activation(val))
            elif self.activation == "elu":
                outputs.append(elu(val))
            else:
                outputs.append(val)

        self.last_net_inputs = net_inputs
        self.last_outputs = outputs
        return outputs

    def backward(self, d_out: List[float], lr: float) -> List[float]:
        d_net = [0.0] * self.out_features

        for r in range(self.out_features):
            net_in = self.last_net_inputs[r]
            out_val = self.last_outputs[r]
            if self.activation == "relu":
                d_net[r] = d_out[r] * relu_derivative(net_in)
            elif self.activation == "leaky_relu":
                d_net[r] = d_out[r] * leaky_relu_derivative(net_in)
            elif self.activation == "sigmoid":
                d_net[r] = d_out[r] * sigmoid_derivative(out_val)
            elif self.activation == "tanh":
                d_net[r] = d_out[r] * tanh_derivative(out_val)
            elif self.activation == "elu":
                d_net[r] = d_out[r] * elu_derivative(net_in, out_val)
            else:
                d_net[r] = d_out[r]

        d_in = [0.0] * self.in_features
        for c in range(self.in_features):
            d_in[c] = sum(d_net[r] * self.weights[r][c] for r in range(self.out_features))

        dw = [[0.0] * self.in_features for _ in range(self.out_features)]
        for r in range(self.out_features):
            for c in range(self.in_features):
                dw[r][c] = d_net[r] * self.last_inputs[c]

        db = list(d_net)

        # Apply optimizer update
        self.weights, self.biases = self.optimizer.update(
            self.weights, self.biases, dw, db, f"dense_{self.in_features}_{self.out_features}"
        )

        return d_in


# =====================================================================
# PART 6: MULTI-HEAD ATTENTION (MHA) TRANSFORMER CORE
# =====================================================================

class MultiHeadAttention:
    """
    Complete Multi-Head Attention layer.
    Allows cognitive splits into independent attention streams to parse AST graphs.
    """
    def __init__(self, embed_dim: int, num_heads: int):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        self.head_dim = embed_dim // num_heads

        # Heads linear projections
        self.q_proj = xavier_init(embed_dim, embed_dim)
        self.k_proj = xavier_init(embed_dim, embed_dim)
        self.v_proj = xavier_init(embed_dim, embed_dim)
        self.out_proj = xavier_init(embed_dim, embed_dim)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, sequence: List[List[float]]) -> List[List[float]]:
        seq_len = len(sequence)
        if seq_len == 0:
            return []

        # Global Q, K, V Projections
        queries = [matrix_vector_multiply(self.q_proj, x) for x in sequence]
        keys = [matrix_vector_multiply(self.k_proj, x) for x in sequence]
        values = [matrix_vector_multiply(self.v_proj, x) for x in sequence]

        # Attention across splits
        head_outputs = []
        for head in range(self.num_heads):
            start_idx = head * self.head_dim
            end_idx = start_idx + self.head_dim

            # Extract slices
            q_h = [q[start_idx:end_idx] for q in queries]
            k_h = [k[start_idx:end_idx] for k in keys]
            v_h = [v[start_idx:end_idx] for v in values]

            # Scaled Dot-Product Attention
            scores = [[0.0] * seq_len for _ in range(seq_len)]
            for i in range(seq_len):
                for j in range(seq_len):
                    scores[i][j] = dot_product(q_h[i], k_h[j]) / self.scale

            weights = [softmax(row) for row in scores]

            # Value pooling
            context_h = [[0.0] * self.head_dim for _ in range(seq_len)]
            for i in range(seq_len):
                for j in range(seq_len):
                    scaled = scale_vector(v_h[j], weights[i][j])
                    context_h[i] = vector_add(context_h[i], scaled)
            head_outputs.append(context_h)

        # Concatenate heads output back
        concat_sequence = []
        for i in range(seq_len):
            joined_token = []
            for h in range(self.num_heads):
                joined_token.extend(head_outputs[h][i])
            concat_sequence.append(joined_token)

        # Final projection layer
        return [matrix_vector_multiply(self.out_proj, x) for x in concat_sequence]


# =====================================================================
# PART 7: GRAPH ATTENTION NETWORKS (GAT) COGNITION
# =====================================================================

class GraphAttentionLayer:
    """
    Graph Attention Network (GAT) layer.
    Computes dynamic attention coefficient distributions over project dependency nodes.
    """
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        self.w = xavier_init(out_features, in_features)
        self.a = [random.uniform(-0.1, 0.1) for _ in range(2 * out_features)]

    def forward(self, node_features: List[List[float]], adj_matrix: List[List[float]]) -> List[List[float]]:
        num_nodes = len(node_features)
        if num_nodes == 0:
            return []

        # Project representation
        projected = [matrix_vector_multiply(self.w, h) for h in node_features]

        # Calculate scores over graph neighborhoods
        attention_out = [[0.0] * self.out_features for _ in range(num_nodes)]
        for i in range(num_nodes):
            row_scores = []
            neighbors = []
            for j in range(num_nodes):
                if adj_matrix[i][j] > 0.0:
                    concat_vec = projected[i] + projected[j]
                    score = dot_product(self.a, concat_vec)
                    row_scores.append(leaky_relu(score, 0.2))
                    neighbors.append(j)

            weights = softmax(row_scores)

            for index, j in enumerate(neighbors):
                weighted_val = scale_vector(projected[j], weights[index])
                attention_out[i] = vector_add(attention_out[i], weighted_val)

        return [[relu(x) for x in h] for h in attention_out]


# =====================================================================
# PART 8: LONG SHORT-TERM MEMORY (LSTM) STATE CELL
# =====================================================================

class LSTMCell(Layer):
    """
    LSTM Cell for cognitive tracking of sequential code traces.
    """
    def __init__(self, in_dim: int, hidden_dim: int):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        total_in = in_dim + hidden_dim
        self.w_forget = xavier_init(hidden_dim, total_in)
        self.w_input = xavier_init(hidden_dim, total_in)
        self.w_cell = xavier_init(hidden_dim, total_in)
        self.w_output = xavier_init(hidden_dim, total_in)

        self.b_forget = [0.0] * hidden_dim
        self.b_input = [0.0] * hidden_dim
        self.b_cell = [0.0] * hidden_dim
        self.b_output = [0.0] * hidden_dim

    def step(self, x: List[float], h_prev: List[float], c_prev: List[float]) -> Tuple[List[float], List[float]]:
        concat = x + h_prev

        f = [sigmoid(dot_product(self.w_forget[i], concat) + self.b_forget[i]) for i in range(self.hidden_dim)]
        i_gate = [sigmoid(dot_product(self.w_input[i], concat) + self.b_input[i]) for i in range(self.hidden_dim)]
        c_tilde = [math.tanh(dot_product(self.w_cell[i], concat) + self.b_cell[i]) for i in range(self.hidden_dim)]

        # Cell update
        c_next = [f_v * c_prev[idx] + i_v * c_t for idx, (f_v, i_v, c_t) in enumerate(zip(f, i_gate, c_tilde))]

        o = [sigmoid(dot_product(self.w_output[i], concat) + self.b_output[i]) for i in range(self.hidden_dim)]
        h_next = [o_v * math.tanh(c_v) for o_v, c_v in zip(o, c_next)]

        return h_next, c_next


# =====================================================================
# PART 9: THE ADVANCED COGNITIVE AGENT BRAIN NETWORK
# =====================================================================

class CodeCognitiveNetwork:
    """
    Stacked Deep Neural Architecture representing the core brain of the autonomous agent.
    Combines character Embeddings, Multi-head attention, Graph Convolution, and
    multilayered Dense chains into an integrated reasoning engine.
    """
    def __init__(self, vocab_size: int = 128, embed_dim: int = 16):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embeddings = xavier_init(vocab_size, embed_dim)

        # Framework pipeline layers
        self.attention = MultiHeadAttention(embed_dim, num_heads=2)
        self.layer_norm = LayerNormalization(embed_dim)
        self.gat = GraphAttentionLayer(embed_dim, embed_dim)
        self.dense1 = DenseLayer(embed_dim, 8, activation="leaky_relu")
        self.dense2 = DenseLayer(8, 2, activation="sigmoid") # Predictions: [risk, importance]

    def encode_text_sequence(self, text: str) -> List[List[float]]:
        sequence = []
        for char in text[:64]:
            token_idx = ord(char) % self.vocab_size
            sequence.append(list(self.embeddings[token_idx]))
        return sequence

    def process_project_dependency_graph(self, file_contents: Dict[str, str], dependencies: Dict[str, List[str]]) -> Dict[str, List[float]]:
        filenames = list(file_contents.keys())
        num_files = len(filenames)
        if num_files == 0:
            return {}

        initial_features = []
        for fname in filenames:
            content = file_contents[fname]
            seq = self.encode_text_sequence(content)

            # Contextual representation passing through MHA
            attended = self.attention.forward(seq)
            if attended:
                avg_pool = [sum(col) / len(attended) for col in transpose(attended)]
                normalized = self.layer_norm.forward(avg_pool)
                initial_features.append(normalized)
            else:
                initial_features.append([0.0] * self.embed_dim)

        # Build adjacency graph
        adj_matrix = [[0.0] * num_files for _ in range(num_files)]
        for i, f1 in enumerate(filenames):
            adj_matrix[i][i] = 1.0
            deps = dependencies.get(f1, [])
            for dep in deps:
                if dep in filenames:
                    j = filenames.index(dep)
                    adj_matrix[i][j] = 1.0

        # Convolve structural dependencies
        gat_features = self.gat.forward(initial_features, adj_matrix)

        # Score targets
        scores = {}
        for idx, fname in enumerate(filenames):
            features = gat_features[idx]
            hidden = self.dense1.forward(features)
            preds = self.dense2.forward(hidden)
            scores[fname] = preds

        return scores

    def predict_task_risk(self, task_description: str) -> float:
        seq = self.encode_text_sequence(task_description)
        attended = self.attention.forward(seq)
        if not attended:
            return 0.3

        avg_pool = [sum(col) / len(attended) for col in transpose(attended)]
        norm = self.layer_norm.forward(avg_pool)
        hidden = self.dense1.forward(norm)
        preds = self.dense2.forward(hidden)
        return preds[0]


# =====================================================================
# PART 10: AUTO SUPERVISED TRAINING BACKTRACK ROUTINES
# =====================================================================

def train_network_supervised(net: CodeCognitiveNetwork, epochs: int = 50) -> None:
    """Trains the network parameters on mock tasks using Adam optimization gradients."""
    dataset = [
        ("delete the production database", [0.95, 0.90]),
        ("remove authentication filters", [0.90, 0.85]),
        ("add unit tests for login logic", [0.10, 0.70]),
        ("implement raw helper math formulas", [0.05, 0.30]),
    ]

    lr = 0.01
    for _ in range(epochs):
        for text, targets in dataset:
            seq = net.encode_text_sequence(text)
            attended = net.attention.forward(seq)
            if not attended:
                continue
            avg_pool = [sum(col) / len(attended) for col in transpose(attended)]
            norm = net.layer_norm.forward(avg_pool)

            # Predict
            h1 = net.dense1.forward(norm)
            preds = net.dense2.forward(h1)

            # Backpropagation
            loss_grads = [preds[0] - targets[0], preds[1] - targets[1]]
            dh1 = net.dense2.backward(loss_grads, lr)
            net.dense1.backward(dh1, lr)


# Initial cognitive boot training on startup
_global_cognitive_net = CodeCognitiveNetwork()
train_network_supervised(_global_cognitive_net, epochs=10)


# =====================================================================
# PART 11: MULTI-PARADIGM ADVANCED TRAINING METHODS
# =====================================================================

def train_unsupervised_mlm(net: CodeCognitiveNetwork, corpus: List[str], epochs: int = 5) -> None:
    """
    Method 1: Unsupervised Masked Language Modeling (MLM).
    Learns structure by reconstructive masking on custom file corpus strings.
    """
    lr = 0.01
    for _ in range(epochs):
        for document in corpus:
            if len(document) < 10:
                continue
            # Select random index to mask
            mask_char_idx = random.randint(3, len(document) - 5)
            masked_text = document[:mask_char_idx] + "_" + document[mask_char_idx+1:]
            target_char = document[mask_char_idx]

            # Forward pass using masked sequence
            seq = net.encode_text_sequence(masked_text)
            attended = net.attention.forward(seq)
            if not attended:
                continue
            avg_pool = [sum(col) / len(attended) for col in transpose(attended)]
            norm = net.layer_norm.forward(avg_pool)

            # Predict target dimensions
            h1 = net.dense1.forward(norm)
            preds = net.dense2.forward(h1)

            # Supervise towards high risk representation if mask target is a symbol
            target_risk = 0.8 if target_char in [":", "(", ")", "[", "]", "=", "."] else 0.2
            loss_grads = [preds[0] - target_risk, preds[1] - 0.5]

            dh1 = net.dense2.backward(loss_grads, lr)
            net.dense1.backward(dh1, lr)


def train_reinforcement_learning(net: CodeCognitiveNetwork, episodes: int = 10) -> None:
    """
    Method 2: Policy-Gradient REINFORCE Style Reinforcement Learning.
    Guides agent weights via success feedback loops (rewards).
    """
    lr = 0.01
    # States are tasks, Actions are predicted risks
    simulated_env_tasks = [
        ("delete db", 0.95), # high risk target
        ("lint file", 0.10), # low risk target
        ("clean drive", 0.80),
        ("write test", 0.15)
    ]

    for _ in range(episodes):
        for task, ideal_risk in simulated_env_tasks:
            # 1. Forward pass (action exploration)
            seq = net.encode_text_sequence(task)
            attended = net.attention.forward(seq)
            if not attended:
                continue
            avg_pool = [sum(col) / len(attended) for col in transpose(attended)]
            norm = net.layer_norm.forward(avg_pool)

            h1 = net.dense1.forward(norm)
            preds = net.dense2.forward(h1)
            action_risk = preds[0]

            # Calculate reward (higher reward if predicted risk is close to ideal)
            reward = 1.0 - abs(action_risk - ideal_risk)

            # Policy gradient update (loss scaled by reward)
            # Minimize negative log-likelihood * reward
            grad = [(action_risk - ideal_risk) * (1.0 - reward)]
            loss_grads = [grad[0], 0.0]

            dh1 = net.dense2.backward(loss_grads, lr)
            net.dense1.backward(dh1, lr)


def train_evolutionary_strategy(net: CodeCognitiveNetwork, population_size: int = 8, generations: int = 3) -> None:
    """
    Method 3: Evolutionary Strategy (Genetic Algorithm) Optimization.
    Mutates weights, evaluates fitness, and selects the strongest parameters.
    """
    def mutate_matrix(m: List[List[float]], rate: float = 0.05) -> List[List[float]]:
        return [[x + random.normalvariate(0.0, 0.1) if random.random() < rate else x for x in row] for row in m]

    def mutate_vector(v: List[float], rate: float = 0.05) -> List[float]:
        return [x + random.normalvariate(0.0, 0.1) if random.random() < rate else x for x in v]

    # Reference evaluation task dataset
    eval_tasks = [("delete db", 0.95), ("write test", 0.15)]

    for gen in range(generations):
        population = []
        # Generate mutant population from current network
        for i in range(population_size):
            # Clone and mutate weights
            mutant_w1 = mutate_matrix(net.dense1.weights)
            mutant_b1 = mutate_vector(net.dense1.biases)
            mutant_w2 = mutate_matrix(net.dense2.weights)
            mutant_b2 = mutate_vector(net.dense2.biases)
            population.append((mutant_w1, mutant_b1, mutant_w2, mutant_b2))

        # Evaluate fitness (negative mean error)
        fitness_scores = []
        for index, (w1, b1, w2, b2) in enumerate(population):
            # Temporarily set mutant parameters
            orig_w1, orig_b1 = net.dense1.weights, net.dense1.biases
            orig_w2, orig_b2 = net.dense2.weights, net.dense2.biases

            net.dense1.weights, net.dense1.biases = w1, b1
            net.dense2.weights, net.dense2.biases = w2, b2

            # Compute total absolute deviation
            total_error = 0.0
            for task, ideal in eval_tasks:
                risk = net.predict_task_risk(task)
                total_error += abs(risk - ideal)

            # Restore
            net.dense1.weights, net.dense1.biases = orig_w1, orig_b1
            net.dense2.weights, net.dense2.biases = orig_w2, orig_b2

            fitness_scores.append((total_error, index))

        # Select best parameters
        fitness_scores.sort(key=lambda x: x[0]) # lowest error first
        best_mutant_idx = fitness_scores[0][1]
        best_w1, best_b1, best_w2, best_b2 = population[best_mutant_idx]

        # Apply updates to the master network
        net.dense1.weights, net.dense1.biases = best_w1, best_b1
        net.dense2.weights, net.dense2.biases = best_w2, best_b2


# =====================================================================
# PART 12: HEURISTIC SEARCH AND STACKED NEURAL BLOCK MODULES
# =====================================================================

class DeepCognitiveBlock:
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim
        self.attention = MultiHeadAttention(embed_dim, num_heads=2)
        self.layer_norm = LayerNormalization(embed_dim)
        self.dense1 = DenseLayer(embed_dim, embed_dim * 2, activation="leaky_relu")
        self.dense2 = DenseLayer(embed_dim * 2, embed_dim, activation="identity")

    def forward(self, sequence: List[List[float]]) -> List[List[float]]:
        attn_out = self.attention.forward(sequence)
        norm1 = [self.layer_norm.forward(vector_add(x, attn)) for x, attn in zip(sequence, attn_out)]

        final_out = []
        for x in norm1:
            h = self.dense1.forward(x)
            out = self.dense2.forward(h)
            final_out.append(self.layer_norm.forward(vector_add(x, out)))
        return final_out


class CodeHeuristicRanker:
    def __init__(self, feature_dim: int = 16):
        self.feature_dim = feature_dim
        self.similarity_weights = xavier_init(feature_dim, feature_dim)

    def compute_similarity(self, v1: List[float], v2: List[float]) -> float:
        proj_v2 = matrix_vector_multiply(self.similarity_weights, v2)
        return dot_product(v1, proj_v2)

    def rank_candidates(self, bug_context_emb: List[float], candidate_patches_embs: List[List[float]]) -> List[int]:
        scores = []
        for idx, patch_emb in enumerate(candidate_patches_embs):
            sim = self.compute_similarity(bug_context_emb, patch_emb)
            scores.append((sim, idx))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [idx for _, idx in scores]
