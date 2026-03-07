"""
Activation Functions and Their Derivatives
Implements: Linear (identity), ReLU, Sigmoid, Tanh, Softmax
"""
import numpy as np


class Linear:
    """Identity activation — used on the output layer so logits pass straight through."""
    def forward(self, z):
        return z

    def backward(self, grad):
        return grad


class ReLU:
    def forward(self, z):
        self.mask = z > 0
        return z * self.mask

    def backward(self, grad):
        return grad * self.mask


class Sigmoid:
    def forward(self, z):
        self.out = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        return self.out

    def backward(self, grad):
        return grad * self.out * (1.0 - self.out)


class Tanh:
    def forward(self, z):
        self.out = np.tanh(z)
        return self.out

    def backward(self, grad):
        return grad * (1.0 - self.out ** 2)


class Softmax:
    """
    Standalone Softmax — kept for completeness.
    NOTE: The network's output layer uses Linear + CrossEntropyLoss (which
    applies log-softmax internally) for numerical stability. Do not use
    Softmax as an output activation together with CrossEntropyLoss.
    """
    def forward(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        e = np.exp(z)
        self.out = e / np.sum(e, axis=1, keepdims=True)
        return self.out

    def backward(self, grad):
        dot = np.sum(grad * self.out, axis=1, keepdims=True)
        return self.out * (grad - dot)


def get_activation(name: str):
    mapping = {
        "relu":    ReLU,
        "sigmoid": Sigmoid,
        "tanh":    Tanh,
        "softmax": Softmax,
        "linear":  Linear,
        "none":    Linear,
    }
    name = name.lower()
    if name not in mapping:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(mapping)}")
    return mapping[name]()
