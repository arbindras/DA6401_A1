"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""
import numpy as np


def load_dataset(name: str = "fashion_mnist"):
    """
    Load MNIST or Fashion-MNIST via Keras.
    Returns (X_train, y_train), (X_val, y_val), (X_test, y_test)
    where X is float32 in [0, 1], shape (N, 784), and y is int labels.
    """
    from tensorflow import keras  # keras is bundled with tensorflow

    name = name.lower()
    if name == "mnist":
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    elif name in ("fashion_mnist", "fashion-mnist"):
        (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
    else:
        raise ValueError(f"Unknown dataset '{name}'. Choose 'mnist' or 'fashion_mnist'.")

    # flatten and normalise
    X_train = X_train.reshape(-1, 784).astype(np.float32) / 255.0
    X_test  = X_test.reshape(-1, 784).astype(np.float32)  / 255.0

    # 90/10 train-val split from the original training set
    split   = int(0.9 * len(X_train))
    X_val, y_val     = X_train[split:], y_train[split:]
    X_train, y_train = X_train[:split],  y_train[:split]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def get_batches(X, y, batch_size: int, shuffle: bool = True):
    """Generator that yields (X_batch, y_batch) mini-batches."""
    N = X.shape[0]
    idx = np.random.permutation(N) if shuffle else np.arange(N)
    for start in range(0, N, batch_size):
        b = idx[start: start + batch_size]
        yield X[b], y[b]
