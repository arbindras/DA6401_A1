"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.ann.neural_layer import NeuralLayer
from src.ann.objective_functions import get_loss
from src.ann.optimizers import get_optimizer, NAG


class NeuralNetwork:
    """
    Main model class that orchestrates neural network training and inference.
    Accepts a cli_args Namespace (or any object with the same attributes).
    Hidden layer sizes are capped at 128 neurons as per assignment spec.
    """

    MAX_NEURONS = 128   # assignment constraint: hidden neurons per layer ≤ 128

    def __init__(self, cli_args):
        args = cli_args

        # ── build hidden layer sizes ──────────────────────────────────────────
        # Priority: hidden_size list > (num_layers × hidden_size scalar)
        if hasattr(args, 'hidden_size') and args.hidden_size:
            hidden_sizes = [min(s, self.MAX_NEURONS)
                            for s in list(args.hidden_size)]
        else:
            n_hidden     = getattr(args, 'num_layers', 1)
            neuron_count = min(getattr(args, 'hidden_size_scalar',
                               getattr(args, 'num_neurons', 128)),
                               self.MAX_NEURONS)
            hidden_sizes = [neuron_count] * n_hidden

        input_size   = getattr(args, 'input_size',   784)
        output_size  = getattr(args, 'output_size',  10)
        activation   = getattr(args, 'activation',   'relu')
        weight_init  = getattr(args, 'weight_init',  'random')
        weight_decay = getattr(args, 'weight_decay', 0.0)

        sizes = [input_size] + hidden_sizes + [output_size]

        self.layers = []
        for i in range(len(sizes) - 1):
            act = activation if i < len(sizes) - 2 else 'linear'
            self.layers.append(
                NeuralLayer(sizes[i], sizes[i + 1],
                            activation=act,
                            weight_init=weight_init,
                            weight_decay=weight_decay)
            )

        # ── loss & optimiser ──────────────────────────────────────────────────
        loss_name = getattr(args, 'loss', 'cross_entropy')
        self.loss_fn = get_loss(loss_name)

        opt_name   = getattr(args, 'optimizer', 'adam')
        opt_kwargs = dict(
            lr    = getattr(args, 'learning_rate', 1e-3),
            beta1 = getattr(args, 'beta1',   0.9),
            beta2 = getattr(args, 'beta2',   0.999),
            beta  = getattr(args, 'beta',    0.9),
            eps   = getattr(args, 'epsilon', 1e-8),
        )
        self.optimizer = get_optimizer(opt_name, **opt_kwargs)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        X      : (batch, D_in)
        returns: logits shape (batch, D_out)  — output of last (softmax) layer
        """
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    # ── backward ──────────────────────────────────────────────────────────────

    def backward(self, y_true, y_pred):
        """
        Backpropagate and compute gradients for every layer.
        After this call:
          - Each layer exposes layer.grad_W and layer.grad_b  (autograder requirement)
          - self.grad_W[0] / self.grad_b[0] = gradients of the LAST layer
        Returns (grad_W, grad_b) as object arrays indexed from last→first layer.
        """
        grad = self.loss_fn.backward()

        grad_W_list = []
        grad_b_list = []

        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            grad_W_list.append(layer.grad_W)   # already set on layer by backward()
            grad_b_list.append(layer.grad_b)

        # Store as object arrays (index 0 = last layer, per spec)
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        return self.grad_W, self.grad_b

    # ── weight update ─────────────────────────────────────────────────────────

    def update_weights(self):
        self.optimizer.step(self.layers)

    # ── training loop ─────────────────────────────────────────────────────────

    def train(self, X_train, y_train, epochs=1, batch_size=32,
              X_val=None, y_val=None, wandb_run=None):
        """
        Mini-batch SGD training loop with optional W&B logging.
        NAG lookahead is applied automatically when that optimizer is selected.
        """
        N = X_train.shape[0]
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(1, epochs + 1):
            idx      = np.random.permutation(N)
            X_s, y_s = X_train[idx], y_train[idx]

            epoch_loss = 0.0
            n_batches  = 0

            for start in range(0, N, batch_size):
                xb = X_s[start: start + batch_size]
                yb = y_s[start: start + batch_size]

                # NAG: shift weights to look-ahead point before forward pass
                if isinstance(self.optimizer, NAG):
                    self.optimizer.apply_lookahead(self.layers)

                logits = self.forward(xb)
                loss   = self.loss_fn.forward(logits, yb)
                self.backward(yb, logits)
                self.update_weights()

                epoch_loss += loss
                n_batches  += 1

            epoch_loss    /= n_batches
            train_metrics  = self.evaluate(X_train, y_train)
            history["train_loss"].append(epoch_loss)
            history["train_acc"].append(train_metrics["accuracy"])

            log_dict = {
                "epoch":      epoch,
                "train_loss": epoch_loss,
                "train_acc":  train_metrics["accuracy"],
            }

            if X_val is not None and y_val is not None:
                val_metrics = self.evaluate(X_val, y_val)
                history["val_loss"].append(val_metrics["loss"])
                history["val_acc"].append(val_metrics["accuracy"])
                log_dict.update({"val_loss": val_metrics["loss"],
                                 "val_acc":  val_metrics["accuracy"]})
                print(f"Epoch {epoch:3d}/{epochs}  "
                      f"loss={epoch_loss:.4f}  acc={train_metrics['accuracy']:.4f}  "
                      f"val_loss={val_metrics['loss']:.4f}  "
                      f"val_acc={val_metrics['accuracy']:.4f}")
            else:
                print(f"Epoch {epoch:3d}/{epochs}  "
                      f"loss={epoch_loss:.4f}  acc={train_metrics['accuracy']:.4f}")

            if wandb_run is not None:
                wandb_run.log(log_dict)

        return history

    # ── evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, X, y) -> dict:
        logits   = self.forward(X)
        loss     = self.loss_fn.forward(logits, y)
        preds    = np.argmax(logits, axis=1)
        labels   = y if y.ndim == 1 else np.argmax(y, axis=1)
        accuracy = float(np.mean(preds == labels))
        return {"loss": float(loss), "accuracy": accuracy,
                "logits": logits, "predictions": preds}

    # ── serialisation ─────────────────────────────────────────────────────────

    def get_weights(self) -> dict:
        """Return a flat dict {W0, b0, W1, b1, ...}."""
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict: dict):
        """Load weights from a dict produced by get_weights() or np.load()."""
        for i, layer in enumerate(self.layers):
            if f"W{i}" in weight_dict:
                layer.W = np.array(weight_dict[f"W{i}"]).copy()
            if f"b{i}" in weight_dict:
                layer.b = np.array(weight_dict[f"b{i}"]).copy()
