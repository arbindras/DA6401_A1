"""
Optimization Algorithms
Implements: SGD, Momentum, NAG, RMSProp, Adam, Nadam
All optimizers respect the weight_decay (L2) already baked into layer.grad_W.
NAG exposes apply_lookahead() which must be called before the forward pass.
"""
import numpy as np


class SGD:
    def __init__(self, lr=0.01, **_):
        self.lr = lr

    def step(self, layers):
        for layer in layers:
            layer.W -= self.lr * layer.grad_W
            layer.b -= self.lr * layer.grad_b


class Momentum:
    def __init__(self, lr=0.01, beta=0.9, **_):
        self.lr   = lr
        self.beta = beta
        self.vW   = {}
        self.vb   = {}

    def step(self, layers):
        for i, layer in enumerate(layers):
            vW = self.vW.get(i, np.zeros_like(layer.W))
            vb = self.vb.get(i, np.zeros_like(layer.b))
            vW = self.beta * vW + self.lr * layer.grad_W
            vb = self.beta * vb + self.lr * layer.grad_b
            self.vW[i], self.vb[i] = vW, vb
            layer.W -= vW
            layer.b -= vb


class NAG:
    """
    Nesterov Accelerated Gradient.
    Call apply_lookahead(layers) BEFORE the forward pass each batch,
    then call step(layers) AFTER the backward pass.
    """
    def __init__(self, lr=0.01, beta=0.9, **_):
        self.lr   = lr
        self.beta = beta
        self.vW   = {}
        self.vb   = {}

    def apply_lookahead(self, layers):
        """Temporarily shift weights to the look-ahead position."""
        for i, layer in enumerate(layers):
            vW = self.vW.get(i, np.zeros_like(layer.W))
            vb = self.vb.get(i, np.zeros_like(layer.b))
            layer.W -= self.beta * vW
            layer.b -= self.beta * vb

    def step(self, layers):
        for i, layer in enumerate(layers):
            vW = self.vW.get(i, np.zeros_like(layer.W))
            vb = self.vb.get(i, np.zeros_like(layer.b))
            # Restore from lookahead, then apply full Nesterov update
            layer.W += self.beta * vW
            layer.b += self.beta * vb
            vW = self.beta * vW + self.lr * layer.grad_W
            vb = self.beta * vb + self.lr * layer.grad_b
            self.vW[i], self.vb[i] = vW, vb
            layer.W -= vW
            layer.b -= vb


class RMSProp:
    def __init__(self, lr=0.001, beta=0.9, eps=1e-8, **_):
        self.lr   = lr
        self.beta = beta
        self.eps  = eps
        self.sW   = {}
        self.sb   = {}

    def step(self, layers):
        for i, layer in enumerate(layers):
            sW = self.sW.get(i, np.zeros_like(layer.W))
            sb = self.sb.get(i, np.zeros_like(layer.b))
            sW = self.beta * sW + (1 - self.beta) * layer.grad_W ** 2
            sb = self.beta * sb + (1 - self.beta) * layer.grad_b ** 2
            self.sW[i], self.sb[i] = sW, sb
            layer.W -= self.lr * layer.grad_W / (np.sqrt(sW) + self.eps)
            layer.b -= self.lr * layer.grad_b / (np.sqrt(sb) + self.eps)


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, **_):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.mW, self.mb = {}, {}
        self.vW, self.vb = {}, {}
        self.t = 0

    def step(self, layers):
        self.t += 1
        for i, layer in enumerate(layers):
            mW = self.mW.get(i, np.zeros_like(layer.W))
            mb = self.mb.get(i, np.zeros_like(layer.b))
            vW = self.vW.get(i, np.zeros_like(layer.W))
            vb = self.vb.get(i, np.zeros_like(layer.b))

            mW = self.beta1 * mW + (1 - self.beta1) * layer.grad_W
            mb = self.beta1 * mb + (1 - self.beta1) * layer.grad_b
            vW = self.beta2 * vW + (1 - self.beta2) * layer.grad_W ** 2
            vb = self.beta2 * vb + (1 - self.beta2) * layer.grad_b ** 2

            self.mW[i], self.mb[i] = mW, mb
            self.vW[i], self.vb[i] = vW, vb

            mW_h = mW / (1 - self.beta1 ** self.t)
            mb_h = mb / (1 - self.beta1 ** self.t)
            vW_h = vW / (1 - self.beta2 ** self.t)
            vb_h = vb / (1 - self.beta2 ** self.t)

            layer.W -= self.lr * mW_h / (np.sqrt(vW_h) + self.eps)
            layer.b -= self.lr * mb_h / (np.sqrt(vb_h) + self.eps)


class Nadam:
    """Nesterov-accelerated Adam."""
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, **_):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.mW, self.mb = {}, {}
        self.vW, self.vb = {}, {}
        self.t = 0

    def step(self, layers):
        self.t += 1
        b1t  = self.beta1 ** self.t
        b1t1 = self.beta1 ** (self.t + 1)
        b2t  = self.beta2 ** self.t

        for i, layer in enumerate(layers):
            mW = self.mW.get(i, np.zeros_like(layer.W))
            mb = self.mb.get(i, np.zeros_like(layer.b))
            vW = self.vW.get(i, np.zeros_like(layer.W))
            vb = self.vb.get(i, np.zeros_like(layer.b))

            mW = self.beta1 * mW + (1 - self.beta1) * layer.grad_W
            mb = self.beta1 * mb + (1 - self.beta1) * layer.grad_b
            vW = self.beta2 * vW + (1 - self.beta2) * layer.grad_W ** 2
            vb = self.beta2 * vb + (1 - self.beta2) * layer.grad_b ** 2

            self.mW[i], self.mb[i] = mW, mb
            self.vW[i], self.vb[i] = vW, vb

            vW_h = vW / (1 - b2t)
            vb_h = vb / (1 - b2t)
            # Nesterov blend: mix current and next-step momentum estimates
            mW_n = (self.beta1 * mW / (1 - b1t1) +
                    (1 - self.beta1) * layer.grad_W / (1 - b1t))
            mb_n = (self.beta1 * mb / (1 - b1t1) +
                    (1 - self.beta1) * layer.grad_b / (1 - b1t))

            layer.W -= self.lr * mW_n / (np.sqrt(vW_h) + self.eps)
            layer.b -= self.lr * mb_n / (np.sqrt(vb_h) + self.eps)


def get_optimizer(name: str, **kwargs):
    mapping = {
        "sgd":      SGD,
        "momentum": Momentum,
        "nag":      NAG,
        "rmsprop":  RMSProp,
        "adam":     Adam,
        "nadam":    Nadam,
    }
    name = name.lower()
    if name not in mapping:
        raise ValueError(f"Unknown optimizer '{name}'. Choose from {list(mapping)}")
    return mapping[name](**kwargs)
