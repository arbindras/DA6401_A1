"""
Loss / Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
import numpy as np

class CrossEntropyLoss:
    """
    Numerically-stable cross-entropy loss that expects raw logits.
    Applies log-softmax internally — do NOT pass softmax probabilities.
    Accepts integer class labels OR one-hot encoded targets.
    """

    def __init__(self):
        self.probs = None
        self.one_hot = None
        self.N = None

    def forward(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        N = logits.shape[0]

        x = logits - np.max(logits, axis=1, keepdims=True)
        lse = np.log(np.sum(np.exp(x), axis=1, keepdims=True))

        self.probs = np.exp(x - lse)

        if y_true.ndim == 1:
            self.one_hot = np.zeros_like(self.probs)
            self.one_hot[np.arange(N), y_true] = 1.0
        else:
            self.one_hot = y_true.astype(float)

        self.N = N

        return float(-np.sum(self.one_hot * (x - lse)) / N)

    def backward(self, logits=None, y_true=None) -> np.ndarray:
        # If logits/y_true provided, recompute forward state first
        if logits is not None and y_true is not None:
            self.forward(logits, y_true)
        if self.probs is None or self.one_hot is None:
            raise RuntimeError("Call forward() before backward(), or pass logits and y_true.")
        return (self.probs - self.one_hot) / self.N


class MSELoss:
    """
    Mean Squared Error loss.
    Treats the network's output probabilities as regression targets.
    Accepts integer class labels OR one-hot encoded targets.
    """
    def __init__(self):
        self.one_hot = None
        self.diff = None

    def forward(self, pred: np.ndarray, y_true: np.ndarray) -> float:
        N = pred.shape[0]
        if y_true.ndim == 1:
            self.one_hot = np.zeros_like(pred)
            self.one_hot[np.arange(N), y_true] = 1.0
        else:
            self.one_hot = y_true.astype(float)

        self.diff = pred - self.one_hot
        return float(np.mean(self.diff ** 2))

    def backward(self, pred=None, y_true=None):
        if pred is not None and y_true is not None:
            self.forward(pred, y_true)
        if self.diff is None:
            raise RuntimeError("Call forward() before backward(), or pass pred and y_true.")
        return 2.0 * self.diff / self.diff.size


def get_loss(name: str):
    mapping = {
        "cross_entropy":      CrossEntropyLoss,
        "mse":                MSELoss,
        "mean_squared_error": MSELoss,   # accept both spellings
    }
    name = name.lower()
    if name not in mapping:
        raise ValueError(f"Unknown loss '{name}'. Choose from {list(mapping)}")
    return mapping[name]()
