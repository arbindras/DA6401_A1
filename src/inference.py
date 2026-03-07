"""
Inference Script
Load a serialised best_model.npy and evaluate on the test set.
Outputs: Accuracy, Precision, Recall, F1-score.
"""
import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.data_loader import load_dataset
from src.ann.neural_network import NeuralNetwork


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run inference with a saved .npy model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path",
                        default="models/best_model.npy",
                        help="Relative path to saved model weights (.npy)")
    parser.add_argument("-d",   "--dataset",
                        default="fashion_mnist",
                        choices=["mnist", "fashion_mnist"])
    parser.add_argument("-b",   "--batch_size",  type=int, default=256)

    # Architecture — must match the saved model
    parser.add_argument("-nhl", "--num_layers",  type=int, default=3)
    parser.add_argument("-sz",  "--hidden_size", type=int, nargs="+", default=None)
    parser.add_argument("-a",   "--activation",  default="relu",
                        choices=["sigmoid", "tanh", "relu"])
    parser.add_argument("-l",   "--loss",        default="cross_entropy",
                        choices=["mean_squared_error", "cross_entropy"])
    parser.add_argument("-w_i", "--weight_init", default="xavier",
                        choices=["random", "xavier"])
    parser.add_argument("-wd",  "--weight_decay", type=float, default=0.0)

    # Fixed dataset params
    parser.add_argument("--input_size",   type=int, default=784)
    parser.add_argument("--output_size",  type=int, default=10)

    # Dummy optimiser fields (needed only so NeuralNetwork.__init__ is happy)
    parser.add_argument("-o",  "--optimizer",     default="adam")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
    parser.add_argument("--beta1",   type=float, default=0.9)
    parser.add_argument("--beta2",   type=float, default=0.999)
    parser.add_argument("--beta",    type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=1e-8)

    return parser.parse_args()


def load_model(model_path: str, args) -> NeuralNetwork:
    """Reconstruct the network architecture and load serialised weights."""
    # Attempt to auto-load config from best_config.json in same directory
    config_path = os.path.join(os.path.dirname(os.path.abspath(model_path)),
                               "best_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        # Override args with saved config values
        for key, val in cfg.items():
            if hasattr(args, key):
                setattr(args, key, val)

    # Normalise loss name
    if args.loss == "mean_squared_error":
        args.loss = "mse"

    # Resolve hidden sizes
    cap = NeuralNetwork.MAX_NEURONS
    if args.hidden_size:
        if len(args.hidden_size) == 1:
            args.hidden_size = [min(args.hidden_size[0], cap)] * args.num_layers
        else:
            args.hidden_size = [min(s, cap) for s in args.hidden_size]
    else:
        args.hidden_size = [cap] * args.num_layers

    model   = NeuralNetwork(args)
    weights = np.load(model_path, allow_pickle=True).item()  # loads the dict
    model.set_weights(weights)
    return model


def evaluate_model(model: NeuralNetwork, X_test, y_test) -> dict:
    """Return loss, accuracy, precision, recall, F1 (weighted)."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    result = model.evaluate(X_test, y_test)
    preds  = result["predictions"]
    labels = y_test if y_test.ndim == 1 else np.argmax(y_test, axis=1)

    result["f1"]        = float(f1_score(labels, preds, average="weighted"))
    result["precision"] = float(precision_score(labels, preds, average="weighted",
                                                zero_division=0))
    result["recall"]    = float(recall_score(labels, preds, average="weighted"))
    return result


def main() -> dict:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')
    args = parse_arguments()

    print(f"Loading dataset: {args.dataset}…")
    _, _, (X_test, y_test) = load_dataset(args.dataset)

    print(f"Loading model from: {args.model_path}…")
    model = load_model(args.model_path, args)

    metrics = evaluate_model(model, X_test, y_test)

    print("\n── Evaluation Results ──────────────────")
    print(f"  Loss      : {metrics['loss']:.4f}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    print("────────────────────────────────────────")
    print("Evaluation complete!")
    return metrics


if __name__ == "__main__":
    main()
