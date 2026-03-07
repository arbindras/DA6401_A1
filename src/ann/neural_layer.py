"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
from src.ann.activations import get_activation


class NeuralLayer:
    """
    A single fully-connected (dense) layer with an activation function.
    Stores pre-activation (z) and post-activation (a) for backprop.
    Exposes self.grad_W and self.grad_b after every backward() call,
    as required by the autograder.
    """

    def __init__(self, in_features: int, out_features: int,
                 activation: str = "relu", weight_init: str = "random",
                 weight_decay: float = 0.0):
        self.in_features  = in_features
        self.out_features = out_features
        self.activation   = get_activation(activation)
        self.weight_decay = weight_decay   # L2 regularisation coefficient

        # Gradient placeholders (exposed for autograder)
        self.grad_W = None
        self.grad_b = None

        self._init_weights(weight_init)

    # ── weight initialisation ─────────────────────────────────────────────────

    def _init_weights(self, method: str):
        method = method.lower()
        if method in ("xavier", "glorot"):
            # Xavier uniform: U(-sqrt(1/fan_in), sqrt(1/fan_in))
            limit = np.sqrt(1.0 / self.in_features)
            self.W = np.random.uniform(-limit, limit,
                                       (self.in_features, self.out_features))
        elif method in ("he", "kaiming"):
            std = np.sqrt(2.0 / self.in_features)
            self.W = np.random.randn(self.in_features, self.out_features) * std
        else:                                        # "random" / default
            self.W = np.random.randn(self.in_features, self.out_features) * 0.01
        self.b = np.zeros((1, self.out_features))

        # Also expose as dW/db aliases (backward not yet called)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, a_prev: np.ndarray) -> np.ndarray:
        """
        a_prev : (batch, in_features)
        returns: (batch, out_features) — post-activation output
        """
        self.a_prev = a_prev                        # cache for backprop
        self.z      = a_prev @ self.W + self.b      # pre-activation
        self.a      = self.activation.forward(self.z)
        return self.a

    # ── backward ──────────────────────────────────────────────────────────────

    def backward(self, grad_a: np.ndarray) -> np.ndarray:
        """
        grad_a  : gradient of loss w.r.t. this layer's output (batch, out_features)
        returns : gradient of loss w.r.t. this layer's input  (batch, in_features)
        Sets self.grad_W, self.grad_b (and aliases dW, db) for the optimiser.
        Weight decay (L2) is added to grad_W: grad_W += lambda * W
        """
        grad_z = self.activation.backward(grad_a)               # (batch, out)

        # dL/dW = a_prev.T @ grad_z  (no extra /N: loss.backward already averages)
        # L2 term: add weight_decay * W
        self.dW = self.a_prev.T @ grad_z + self.weight_decay * self.W
        self.db = grad_z.sum(axis=0, keepdims=True)

        # Expose under the autograder-required names
        self.grad_W = self.dW
        self.grad_b = self.db

        return grad_z @ self.W.T                                 # (batch, in)
