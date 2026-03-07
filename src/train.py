"""
Main Training Script
Entry point for training neural networks with command-line arguments.
CLI flags match the assignment specification exactly.
"""
import argparse
import json
import os
import sys
import numpy as np
import codecs
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

from src.utils.data_loader import load_dataset
from src.ann.neural_network import NeuralNetwork


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train a NumPy MLP on MNIST / Fashion-MNIST",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── dataset & training ────────────────────────────────────────────────────
    parser.add_argument("-d",   "--dataset",
                        default="fashion_mnist",
                        choices=["mnist", "fashion_mnist"],
                        help="Dataset to train on")
    parser.add_argument("-e",   "--epochs",
                        type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("-b",   "--batch_size",
                        type=int, default=32,
                        help="Mini-batch size")

    # ── architecture ──────────────────────────────────────────────────────────
    parser.add_argument("-nhl", "--num_layers",
                        type=int, default=3,
                        help="Number of hidden layers")
    parser.add_argument("-sz",  "--hidden_size",
                        type=int, nargs="+", default=None,
                        help="Neurons per hidden layer. If a single int, all "
                             "layers use that size. If a list, overrides -nhl. "
                             "Capped at 128 per assignment spec.")
    parser.add_argument("-a",   "--activation",
                        default="relu",
                        choices=["sigmoid", "tanh", "relu"],
                        help="Hidden layer activation function")

    # ── optimiser & regularisation ────────────────────────────────────────────
    parser.add_argument("-o",   "--optimizer",
                        default="adam",
                        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],
                        help="Optimisation algorithm")
    parser.add_argument("-lr",  "--learning_rate",
                        type=float, default=1e-3,
                        help="Initial learning rate")
    parser.add_argument("-wd",  "--weight_decay",
                        type=float, default=0.0,
                        help="L2 regularisation coefficient (weight decay)")
    parser.add_argument("--beta1",
                        type=float, default=0.9,
                        help="Adam/Nadam β₁ (first moment decay)")
    parser.add_argument("--beta2",
                        type=float, default=0.999,
                        help="Adam/Nadam β₂ (second moment decay)")
    parser.add_argument("--beta",
                        type=float, default=0.9,
                        help="Momentum / NAG / RMSProp decay coefficient")
    parser.add_argument("--epsilon",
                        type=float, default=1e-8,
                        help="Numerical stability constant for Adam/Nadam/RMSProp")

    # ── loss & initialisation ─────────────────────────────────────────────────
    parser.add_argument("-l",   "--loss",
                        default="cross_entropy",
                        choices=["mean_squared_error", "cross_entropy"],
                        help="Loss function")
    parser.add_argument("-w_i", "--weight_init",
                        default="xavier",
                        choices=["random", "xavier"],
                        help="Weight initialisation strategy")

    # ── W&B & persistence ─────────────────────────────────────────────────────
    parser.add_argument("--wandb_project",
                        default=None,
                        help="W&B project name (omit to skip W&B logging)")
    parser.add_argument("--wandb_entity",
                        default=None,
                        help="W&B entity / username")
    parser.add_argument("--model_save_path",
                        default="models/best_model.npy",
                        help="Relative path to save best model weights (.npy)")

    # ── fixed dataset params ──────────────────────────────────────────────────
    parser.add_argument("--input_size",  type=int, default=784)
    parser.add_argument("--output_size", type=int, default=10)

    return parser.parse_args()


def _build_hidden_sizes(args):
    """
    Resolve hidden layer sizes from CLI args.
    -sz 128 128 128   → [128, 128, 128]
    -sz 128           → [128] * num_layers
    (no -sz)          → [128] * num_layers  (default)
    All values capped at NeuralNetwork.MAX_NEURONS (128).
    """
    cap = NeuralNetwork.MAX_NEURONS
    if args.hidden_size:
        if len(args.hidden_size) == 1:
            return [min(args.hidden_size[0], cap)] * args.num_layers
        return [min(s, cap) for s in args.hidden_size]
    return [cap] * args.num_layers



def _save_model(model, args, save_path: str):
    save_dir = os.path.dirname(os.path.abspath(save_path))
    os.makedirs(save_dir, exist_ok=True)


    weights = model.get_weights()
    np.save(save_path, weights)
    print(f"Weights saved -> {save_path}")


    config = {
        "dataset": args.dataset,
        "num_layers": args.num_layers,
        "hidden_size": _build_hidden_sizes(args),
        "activation": args.activation,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "loss": args.loss,
        "weight_init": args.weight_init,
        "input_size": args.input_size,
        "output_size": args.output_size,
    }

    config_path = os.path.join(save_dir, "best_config.json")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Config saved -> {config_path}")


def main():
    args = parse_arguments()

    # Inject resolved hidden sizes back onto args so NeuralNetwork can use them
    args.hidden_size = _build_hidden_sizes(args)

    # Normalise loss name (allow both "mse" and "mean_squared_error")
    if args.loss == "mean_squared_error":
        args.loss = "mse"

    # ── optional W&B ─────────────────────────────────────────────────────────
    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity if args.wandb_entity else None,
            config=vars(args),
        )

    # ── data ──────────────────────────────────────────────────────────────────
    print(f"Loading {args.dataset}…")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_dataset(args.dataset)
    print(f"  train={X_train.shape}  val={X_val.shape}  test={X_test.shape}")

    # ── model ─────────────────────────────────────────────────────────────────
    model = NeuralNetwork(args)

    # ── train ─────────────────────────────────────────────────────────────────
    model.train(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        X_val=X_val, y_val=y_val,
        wandb_run=wandb_run,
    )

    # ── test evaluation ───────────────────────────────────────────────────────
    test_metrics = model.evaluate(X_test, y_test)
    print(f"\nTest  loss={test_metrics['loss']:.4f}  "
          f"acc={test_metrics['accuracy']:.4f}")

    if wandb_run:
        wandb_run.log({"test_loss": test_metrics["loss"],
                       "test_acc":  test_metrics["accuracy"]})
        wandb_run.finish()

    # ── save ──────────────────────────────────────────────────────────────────
    _save_model(model, args, args.model_save_path)
    print("Training complete!")


if __name__ == "__main__":
    main()
