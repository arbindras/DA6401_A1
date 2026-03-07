"""
experiments.py  —  DA6401 Assignment 1
Run ALL W&B experiments (2.1 – 2.10) and build the final W&B Report.

Usage:
    pip install wandb keras tensorflow scikit-learn matplotlib
    wandb login
    python experiments.py --project da6401-mlp --entity YOUR_USERNAME

The script will:
  1. Run every required experiment and log results to W&B.
  2. Create a public W&B Report with all plots and written answers.
  3. Print the public report URL at the end.
"""

import argparse, os, sys, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import wandb
from wandb.apis import reports as wr   #
sys.path.insert(0, os.path.dirname(__file__))
from src.utils.data_loader import load_dataset
from src.ann.neural_network import NeuralNetwork
import argparse
import wandb


# ─── CLI ───────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="DA6401_A12", help="W&B project name")
    p.add_argument("--entity", default="arbindrapatel-iitmaana", help="W&B entity/user")
    p.add_argument("--dataset", default="mnist", choices=["mnist", "fashion_mnist"])
    p.add_argument("--skip_sweep", action="store_true", help="Skip 120-run sweep (use cached)")
    return p.parse_args()


args = get_args()


# ─── Model builder helper ──────────────────────────────────────────────────────
def make_args(**kw):
    defaults = dict(num_layers=3, hidden_size=[128,128,128], activation="relu",
                    weight_init="xavier", loss="cross_entropy", optimizer="adam",
                    learning_rate=1e-3, weight_decay=0.0,
                    beta1=0.9, beta2=0.999, beta=0.9, epsilon=1e-8,
                    input_size=784, output_size=10)
    defaults.update(kw)
    return type("Args", (), defaults)()

# ─── Data (loaded once) ───────────────────────────────────────────────────────
DATASET = None
CLASS_NAMES_MNIST   = [str(i) for i in range(10)]
CLASS_NAMES_FASHION = ["T-shirt","Trouser","Pullover","Dress","Coat",
                        "Sandal","Shirt","Sneaker","Bag","Ankle Boot"]

def get_data(dataset="fashion_mnist"):
    global DATASET
    if DATASET is None:
        (Xtr,ytr),(Xv,yv),(Xte,yte) = load_dataset(dataset)
        DATASET = (Xtr,ytr),(Xv,yv),(Xte,yte)
    return DATASET

# ═════════════════════════════════════════════════════════════════════════════
#  2.1  Data Exploration
# ═════════════════════════════════════════════════════════════════════════════
def run_2_1(project, entity, dataset):
    print("\n── 2.1 Data Exploration ──")
    (Xtr,ytr),_,_ = get_data(dataset)
    cnames = CLASS_NAMES_FASHION if "fashion" in dataset else CLASS_NAMES_MNIST

    run = wandb.init(project=project, entity=entity,
                     name="2.1-data-exploration", job_type="eda",
                     config={"dataset": dataset})

    # Log 5 samples per class as a W&B Table
    table = wandb.Table(columns=["image","label","class_name"])
    for cls in range(10):
        idxs = np.where(ytr == cls)[0][:5]
        for i in idxs:
            img = wandb.Image(Xtr[i].reshape(28,28), caption=cnames[cls])
            table.add_data(img, int(cls), cnames[cls])
    wandb.log({"sample_images": table})

    # Class distribution bar chart
    fig, ax = plt.subplots(figsize=(10,4))
    counts = [int((ytr==c).sum()) for c in range(10)]
    ax.bar(cnames, counts, color="#1F4E8C")
    ax.set_title(f"Class Distribution — {dataset}")
    ax.set_ylabel("Count"); ax.set_xlabel("Class")
    plt.tight_layout()
    wandb.log({"class_distribution": wandb.Image(fig)})
    plt.close()

    run_id = run.id
    run.finish()
    print(f"   run id: {run_id}")
    return run_id


# ═════════════════════════════════════════════════════════════════════════════
#  2.2  Hyperparameter Sweep (120 runs)
# ═════════════════════════════════════════════════════════════════════════════
def run_2_2_sweep(project, entity, dataset):
    print("\n── 2.2 Hyperparameter Sweep (120 runs) ──")

    sweep_config = {
        "method": "bayes",
        "metric": {"name": "val_acc", "goal": "maximize"},
        "parameters": {
            "learning_rate": {"distribution": "log_uniform_values",
                               "min": 1e-4, "max": 1e-2},
            "num_layers":    {"values": [1, 2, 3, 4]},
            "hidden_size":   {"values": [32, 64, 128]},
            "activation":    {"values": ["relu", "sigmoid", "tanh"]},
            "optimizer":     {"values": ["sgd","momentum","nag","rmsprop","adam","nadam"]},
            "batch_size":    {"values": [16, 32, 64, 128]},
            "weight_init":   {"values": ["random", "xavier"]},
            "weight_decay":  {"values": [0.0, 1e-4, 1e-3]},
        }
    }

    (Xtr,ytr),(Xv,yv),_ = get_data(dataset)

    def sweep_train():
        with wandb.init() as run:
            cfg = run.config
            hs  = [min(cfg.hidden_size, 128)] * cfg.num_layers
            a   = make_args(num_layers=cfg.num_layers, hidden_size=hs,
                            activation=cfg.activation, weight_init=cfg.weight_init,
                            loss="cross_entropy", optimizer=cfg.optimizer,
                            learning_rate=cfg.learning_rate,
                            weight_decay=cfg.weight_decay)
            model = NeuralNetwork(a)
            for epoch in range(1, 11):
                N   = Xtr.shape[0]
                idx = np.random.permutation(N)
                batch = cfg.batch_size
                ep_loss, n = 0.0, 0
                from src.ann.optimizers import NAG
                for s in range(0, N, batch):
                    xb, yb = Xtr[idx[s:s+batch]], ytr[idx[s:s+batch]]
                    if isinstance(model.optimizer, NAG):
                        model.optimizer.apply_lookahead(model.layers)
                    logits = model.forward(xb)
                    ep_loss += model.loss_fn.forward(logits, yb)
                    model.backward(yb, logits)
                    model.update_weights()
                    n += 1
                vm  = model.evaluate(Xv, yv)
                tm  = model.evaluate(Xtr[:5000], ytr[:5000])
                wandb.log({"epoch": epoch,
                           "train_loss": ep_loss/n, "train_acc": tm["accuracy"],
                           "val_loss": vm["loss"],  "val_acc": vm["accuracy"]})

    sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)
    wandb.agent(sweep_id, sweep_train, count=120)
    print(f"   sweep id: {sweep_id}")
    return sweep_id


# ═════════════════════════════════════════════════════════════════════════════
#  2.3  Optimizer Showdown (all 6)
# ═════════════════════════════════════════════════════════════════════════════
def run_2_3(project, entity, dataset):
    print("\n── 2.3 Optimizer Showdown ──")
    (Xtr,ytr),(Xv,yv),_ = get_data(dataset)
    optimizers = ["sgd","momentum","nag","rmsprop","adam","nadam"]
    colors = ["#e41a1c","#ff7f00","#4daf4a","#984ea3","#377eb8","#a65628"]
    results = {}

    from src.ann.optimizers import NAG

    for opt, col in zip(optimizers, colors):
        run = wandb.init(project=project, entity=entity,
                         name=f"2.3-optimizer-{opt}", group="2.3-optimizer-showdown",
                         config={"optimizer": opt, "hidden_size": 128,
                                 "num_layers": 3, "activation": "relu",
                                 "learning_rate": 0.001, "dataset": dataset})
        a = make_args(optimizer=opt, num_layers=3, hidden_size=[128,128,128])
        model = NeuralNetwork(a)
        epoch_losses = []
        for epoch in range(1, 21):
            N, batch = Xtr.shape[0], 32
            idx = np.random.permutation(N)
            ep_loss, n = 0.0, 0
            for s in range(0, N, batch):
                xb, yb = Xtr[idx[s:s+batch]], ytr[idx[s:s+batch]]
                if isinstance(model.optimizer, NAG):
                    model.optimizer.apply_lookahead(model.layers)
                logits = model.forward(xb)
                ep_loss += model.loss_fn.forward(logits, yb)
                model.backward(yb, logits)
                model.update_weights()
                n += 1
            vm = model.evaluate(Xv, yv)
            ep = ep_loss / n
            epoch_losses.append(ep)
            wandb.log({"epoch": epoch, "train_loss": ep, "val_acc": vm["accuracy"]})
        results[opt] = epoch_losses
        run.finish()

    # Combined convergence plot (first 5 epochs)
    run = wandb.init(project=project, entity=entity,
                     name="2.3-convergence-plot", group="2.3-optimizer-showdown")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for (opt, losses), col in zip(results.items(), colors):
        axes[0].plot(range(1, 6),  losses[:5],  label=opt, color=col, lw=2, marker="o")
        axes[1].plot(range(1, 21), losses[:20], label=opt, color=col, lw=2)
    for ax, title in zip(axes, ["First 5 Epochs", "All 20 Epochs"]):
        ax.set_xlabel("Epoch"); ax.set_ylabel("Training Loss")
        ax.set_title(f"Optimizer Convergence — {title}"); ax.legend(); ax.grid(alpha=0.3)
    plt.suptitle("Optimizer Showdown (3×128 ReLU, lr=0.001)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    wandb.log({"optimizer_convergence": wandb.Image(fig)})
    plt.close()
    run.finish()
    return results


# ═════════════════════════════════════════════════════════════════════════════
#  2.4  Vanishing Gradient (optimizer=Adam)
# ═════════════════════════════════════════════════════════════════════════════
def run_2_4(project, entity, dataset):
    print("\n── 2.4 Vanishing Gradient Analysis ──")
    (Xtr,ytr),(Xv,yv),_ = get_data(dataset)
    configs = [(2,"relu"),(4,"relu"),(6,"relu"),(2,"sigmoid"),(4,"sigmoid"),(6,"sigmoid")]

    grad_norm_data = {}

    for (n_layers, act) in configs:
        run = wandb.init(project=project, entity=entity,
                         name=f"2.4-{act}-{n_layers}layers",
                         group="2.4-vanishing-gradient",
                         config={"activation": act, "num_layers": n_layers,
                                 "optimizer": "rmsprop", "dataset": dataset})
        hs = [128] * n_layers
        a  = make_args(num_layers=n_layers, hidden_size=hs,
                       activation=act, optimizer="rmsprop", learning_rate=0.001)
        model = NeuralNetwork(a)
        norms = []
        for epoch in range(1, 21):
            N, batch = Xtr.shape[0], 32
            idx = np.random.permutation(N)
            for s in range(0, N, batch):
                xb, yb = Xtr[idx[s:s+batch]], ytr[idx[s:s+batch]]
                model.loss_fn.forward(model.forward(xb), yb)
                model.backward(yb, None)
                model.update_weights()
            # grad norm of FIRST hidden layer
            gn = float(np.linalg.norm(model.layers[0].grad_W))
            norms.append(gn)
            vm = model.evaluate(Xv, yv)
            wandb.log({"epoch": epoch, "grad_norm_layer0": gn,
                       "val_acc": vm["accuracy"]})
        grad_norm_data[f"{act}-{n_layers}L"] = norms
        run.finish()

    # Combined gradient norm plot
    run = wandb.init(project=project, entity=entity,
                     name="2.4-gradient-norm-plot", group="2.4-vanishing-gradient")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    styles = ["-","--","-."]
    relu_c, sig_c = "#377eb8", "#e41a1c"
    for i, n_layers in enumerate([2, 4, 6]):
        axes[0].plot(grad_norm_data[f"relu-{n_layers}L"],
                     color=relu_c, ls=styles[i], lw=2, label=f"ReLU {n_layers}L")
        axes[1].plot(grad_norm_data[f"sigmoid-{n_layers}L"],
                     color=sig_c, ls=styles[i], lw=2, label=f"Sigmoid {n_layers}L")
    for ax, title in zip(axes, ["ReLU", "Sigmoid"]):
        ax.set_xlabel("Epoch"); ax.set_ylabel("Grad Norm — Layer 0")
        ax.set_title(f"{title} — First Hidden Layer Gradient Norm")
        ax.legend(); ax.grid(alpha=0.3)
    plt.suptitle("Vanishing Gradient Analysis (RMSProp, lr=0.001)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    wandb.log({"vanishing_gradient_plot": wandb.Image(fig)})
    plt.close()
    run.finish()
    return grad_norm_data


# ═════════════════════════════════════════════════════════════════════════════
#  2.5  Dead Neuron Investigation
# ═════════════════════════════════════════════════════════════════════════════
def run_2_5(project, entity, dataset):
    print("\n── 2.5 Dead Neuron Investigation ──")
    (Xtr,ytr),(Xv,yv),_ = get_data(dataset)

    def monitor_run(activation, lr, group):
        run = wandb.init(project=project, entity=entity,
                         name=f"2.5-{activation}-lr{lr}",
                         group=group,
                         config={"activation": activation, "learning_rate": lr,
                                 "dataset": dataset})
        a = make_args(activation=activation, learning_rate=lr,
                      num_layers=3, hidden_size=[128,128,128])
        model = NeuralNetwork(a)
        X_sample = Xtr[:500]

        for epoch in range(1, 21):
            N, batch = Xtr.shape[0], 32
            idx = np.random.permutation(N)
            for s in range(0, N, batch):
                xb, yb = Xtr[idx[s:s+batch]], ytr[idx[s:s+batch]]
                model.loss_fn.forward(model.forward(xb), yb)
                model.backward(yb, None)
                model.update_weights()

            # Forward pass to collect activations
            _ = model.forward(X_sample)
            log = {}
            for li in range(len(model.layers) - 1):   # skip output
                act_mat = model.layers[li].a           # (500, 128)
                dead    = float((act_mat == 0).all(axis=0).mean())
                log[f"dead_frac_layer{li+1}"] = dead
                # Activation histogram
                hist_vals = act_mat.flatten()
                log[f"act_hist_layer{li+1}"] = wandb.Histogram(hist_vals)

            vm = model.evaluate(Xv, yv)
            log["val_acc"] = vm["accuracy"]
            log["epoch"]   = epoch
            wandb.log(log)
        run.finish()

    monitor_run("relu", 0.1,  "2.5-dead-neurons")
    monitor_run("relu", 0.001,"2.5-dead-neurons")
    monitor_run("tanh", 0.1,  "2.5-dead-neurons")


# ═════════════════════════════════════════════════════════════════════════════
#  2.6  Loss Function Comparison
# ═════════════════════════════════════════════════════════════════════════════
def run_2_6(project, entity, dataset):
    print("\n── 2.6 Loss Function Comparison ──")
    (Xtr,ytr),(Xv,yv),_ = get_data(dataset)
    from src.ann.optimizers import NAG

    loss_results = {}
    for loss_name in ["cross_entropy", "mse"]:
        run = wandb.init(project=project, entity=entity,
                         name=f"2.6-{loss_name}",
                         group="2.6-loss-comparison",
                         config={"loss": loss_name, "dataset": dataset})
        a = make_args(loss=loss_name, num_layers=3, hidden_size=[128,128,128])
        model = NeuralNetwork(a)
        ep_accs = []
        for epoch in range(1, 21):
            N, batch = Xtr.shape[0], 32
            idx = np.random.permutation(N)
            ep_loss, n = 0.0, 0
            for s in range(0, N, batch):
                xb, yb = Xtr[idx[s:s+batch]], ytr[idx[s:s+batch]]
                logits = model.forward(xb)
                ep_loss += model.loss_fn.forward(logits, yb)
                model.backward(yb, logits)
                model.update_weights()
                n += 1
            vm = model.evaluate(Xv, yv)
            ep_accs.append(vm["accuracy"])
            wandb.log({"epoch": epoch, "train_loss": ep_loss/n,
                       "val_acc": vm["accuracy"]})
        loss_results[loss_name] = ep_accs
        run.finish()

    # Comparison plot
    run = wandb.init(project=project, entity=entity,
                     name="2.6-comparison-plot", group="2.6-loss-comparison")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(loss_results["cross_entropy"], color="#377eb8", lw=2, marker="o",
            markevery=2, label="Cross-Entropy")
    ax.plot(loss_results["mse"],           color="#e41a1c", lw=2, marker="s",
            markevery=2, label="MSE")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation Accuracy")
    ax.set_title("Loss Function Comparison — Val Accuracy (3×128 ReLU Adam lr=0.001)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    wandb.log({"loss_comparison": wandb.Image(fig)})
    plt.close()
    run.finish()
    return loss_results


# ═════════════════════════════════════════════════════════════════════════════
#  2.7  Global Performance (train vs test overlay)
# ═════════════════════════════════════════════════════════════════════════════
def run_2_7(project, entity):
    print("\n── 2.7 Global Performance Analysis ──")
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")

    train_accs, test_accs, opts = [], [], []
    for r in runs:
        if r.summary.get("train_acc") and r.summary.get("val_acc"):
            train_accs.append(r.summary["train_acc"])
            test_accs.append(r.summary["val_acc"])
            opts.append(r.config.get("optimizer", "unknown"))

    run = wandb.init(project=project, entity=entity,
                     name="2.7-global-performance")
    # Log as a W&B scatter plot table
    table = wandb.Table(columns=["train_acc","val_acc","optimizer","overfit_gap"])
    for ta, va, op in zip(train_accs, test_accs, opts):
        table.add_data(ta, va, op, round(ta - va, 4))
    wandb.log({"train_vs_val_scatter": wandb.plot.scatter(
        table, "train_acc", "val_acc",
        title="Training vs Validation Accuracy (all runs)")})
    wandb.log({"all_runs_table": table})

    # Matplotlib overlay
    cmap = plt.cm.get_cmap("tab10", 6)
    opt_list = ["sgd","momentum","nag","rmsprop","adam","nadam"]
    fig, ax = plt.subplots(figsize=(9,7))
    for i, op in enumerate(opt_list):
        idxs = [j for j,o in enumerate(opts) if o==op]
        if idxs:
            ax.scatter([train_accs[j] for j in idxs],
                       [test_accs[j]  for j in idxs],
                       color=cmap(i), label=op, s=40, alpha=0.7)
    ax.plot([0,1],[0,1],"k--",alpha=0.4,label="y=x (no gap)")
    ax.set_xlabel("Train Accuracy"); ax.set_ylabel("Val/Test Accuracy")
    ax.set_title("Train vs Test Accuracy — All Runs\n(points below diagonal = overfitting)")
    ax.legend(bbox_to_anchor=(1,1)); ax.grid(alpha=0.3)
    plt.tight_layout()
    wandb.log({"train_vs_test_overlay": wandb.Image(fig)})
    plt.close()
    run.finish()


# ═════════════════════════════════════════════════════════════════════════════
#  2.8  Error Analysis — Confusion Matrix
# ═════════════════════════════════════════════════════════════════════════════
def run_2_8(project, entity, dataset):
    print("\n── 2.8 Error Analysis ──")
    (Xtr,ytr),(Xv,yv),(Xte,yte) = get_data(dataset)
    cnames = CLASS_NAMES_FASHION if "fashion" in dataset else CLASS_NAMES_MNIST

    # Train best model
    run = wandb.init(project=project, entity=entity,
                     name="2.8-error-analysis",
                     config={"model": "best", "dataset": dataset})
    a = make_args(num_layers=3, hidden_size=[128,128,128],
                  activation="relu", optimizer="nadam", learning_rate=0.001)
    model = NeuralNetwork(a)
    from src.ann.optimizers import NAG
    for epoch in range(1, 21):
        N, batch = Xtr.shape[0], 32
        idx = np.random.permutation(N)
        for s in range(0, N, batch):
            xb, yb = Xtr[idx[s:s+batch]], ytr[idx[s:s+batch]]
            logits = model.forward(xb)
            model.loss_fn.forward(logits, yb)
            model.backward(yb, logits)
            model.update_weights()
        vm = model.evaluate(Xv, yv)
        wandb.log({"epoch": epoch, "val_acc": vm["accuracy"]})

    # Confusion matrix (W&B built-in)
    logits = model.forward(Xte)
    preds  = np.argmax(logits, axis=1)
    wandb.log({"confusion_matrix_builtin": wandb.plot.confusion_matrix(
        preds=preds.tolist(), y_true=yte.tolist(), class_names=cnames)})

    # Matplotlib heatmap
    cm = confusion_matrix(yte, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(cnames, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(cnames, fontsize=9)
    thresh = cm.max() / 2
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i,j]),
                    ha="center", va="center", fontsize=7,
                    color="white" if cm[i,j] > thresh else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — Best Model on {dataset} Test Set\n"
                 f"Accuracy={np.mean(preds==yte)*100:.2f}%")
    plt.tight_layout()
    wandb.log({"confusion_matrix_heatmap": wandb.Image(fig)})
    plt.close()

    # Creative viz: per-class error bar chart
    class_err = []
    for c in range(10):
        mask   = yte == c
        err    = float((preds[mask] != yte[mask]).mean())
        class_err.append(err)
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(cnames, class_err, color=["#e41a1c" if e>0.15 else "#377eb8"
                                             for e in class_err])
    ax.set_ylabel("Error Rate"); ax.set_title("Per-Class Error Rate")
    ax.set_xticklabels(cnames, rotation=45, ha="right")
    plt.tight_layout()
    wandb.log({"per_class_error": wandb.Image(fig)})
    plt.close()

    # Confidence histogram on errors
    probs = np.exp(logits - logits.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    conf   = probs.max(1)
    errors = preds != yte
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(conf[errors],  bins=30, alpha=0.7, color="#e41a1c", label="Misclassified")
    ax.hist(conf[~errors], bins=30, alpha=0.5, color="#377eb8", label="Correct")
    ax.set_xlabel("Model Confidence (max softmax)"); ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution: Correct vs. Misclassified")
    ax.legend(); plt.tight_layout()
    wandb.log({"confidence_histogram": wandb.Image(fig)})
    plt.close()

    acc  = float(np.mean(preds == yte))
    f1   = float(f1_score(yte, preds, average="weighted"))
    prec = float(precision_score(yte, preds, average="weighted", zero_division=0))
    rec  = float(recall_score(yte, preds, average="weighted"))
    wandb.log({"test_accuracy": acc, "test_f1": f1,
               "test_precision": prec, "test_recall": rec})
    run.finish()

    # Save best model
    os.makedirs("models", exist_ok=True)
    np.save("models/best_model.npy", model.get_weights())
    with open("models/best_config.json", "w") as f:
        json.dump({"num_layers":3,"hidden_size":[128,128,128],"activation":"relu",
                   "optimizer":"nadam","learning_rate":0.001,"dataset":dataset}, f, indent=2)
    return model, acc


# ═════════════════════════════════════════════════════════════════════════════
#  2.9  Weight Initialisation Symmetry
# ═════════════════════════════════════════════════════════════════════════════
def run_2_9(project, entity, dataset):
    print("\n── 2.9 Weight Initialisation & Symmetry ──")
    (Xtr,ytr),_,_ = get_data(dataset)

    neuron_indices = [0, 1, 2, 3, 4]   # 5 neurons to track in layer 0

    for init_name in ["zeros", "xavier"]:
        run = wandb.init(project=project, entity=entity,
                         name=f"2.9-init-{init_name}",
                         group="2.9-symmetry",
                         config={"weight_init": init_name, "dataset": dataset})
        a = make_args(num_layers=3, hidden_size=[128,128,128],
                      activation="relu", optimizer="adam",
                      learning_rate=0.001, weight_init="xavier")
        model = NeuralNetwork(a)

        # Override init for zeros run
        if init_name == "zeros":
            for layer in model.layers:
                layer.W[:] = 0.0
                layer.b[:] = 0.0

        iteration = 0
        for s in range(0, min(50 * 32, Xtr.shape[0]), 32):
            xb = Xtr[s:s+32]
            yb = ytr[s:s+32]
            model.loss_fn.forward(model.forward(xb), yb)
            model.backward(yb, None)
            model.update_weights()
            iteration += 1

            log = {"iteration": iteration}
            for ni in neuron_indices:
                gn = float(np.linalg.norm(model.layers[0].grad_W[:, ni]))
                log[f"neuron_{ni}_grad_norm"] = gn
            wandb.log(log)
            if iteration >= 50:
                break

        run.finish()


# ═════════════════════════════════════════════════════════════════════════════
#  2.10  Fashion-MNIST Transfer
# ═════════════════════════════════════════════════════════════════════════════
def run_2_10(project, entity):
    print("\n── 2.10 Fashion-MNIST Transfer Challenge ──")
    (Xtr,ytr),(Xv,yv),(Xte,yte) = get_data("fashion_mnist")
    from src.ann.optimizers import NAG

    configs = [
        {"name":"A","num_layers":3,"hidden_size":[128,128,128],
         "activation":"relu","optimizer":"adam","learning_rate":0.001},
        {"name":"B","num_layers":4,"hidden_size":[128,128,128,128],
         "activation":"relu","optimizer":"nadam","learning_rate":0.001},
        {"name":"C","num_layers":4,"hidden_size":[128,128,128,128],
         "activation":"tanh","optimizer":"adam","learning_rate":0.001},
    ]

    for cfg in configs:
        run = wandb.init(project=project, entity=entity,
                         name=f"2.10-config-{cfg['name']}",
                         group="2.10-fashion-transfer",
                         config={**cfg, "dataset": "fashion_mnist"})
        a = make_args(**{k:v for k,v in cfg.items() if k != "name"})
        model = NeuralNetwork(a)
        for epoch in range(1, 21):
            N, batch = Xtr.shape[0], 32
            idx = np.random.permutation(N)
            for s in range(0, N, batch):
                xb, yb = Xtr[idx[s:s+batch]], ytr[idx[s:s+batch]]
                if isinstance(model.optimizer, NAG):
                    model.optimizer.apply_lookahead(model.layers)
                logits = model.forward(xb)
                model.loss_fn.forward(logits, yb)
                model.backward(yb, logits)
                model.update_weights()
            vm = model.evaluate(Xv, yv)
            tm = model.evaluate(Xtr[:3000], ytr[:3000])
            wandb.log({"epoch": epoch,
                       "train_acc": tm["accuracy"], "val_acc": vm["accuracy"]})

        te = model.evaluate(Xte, yte)
        wandb.log({"test_acc": te["accuracy"],
                   "test_f1": float(f1_score(yte, np.argmax(model.forward(Xte),1),
                                             average="weighted"))})
        run.finish()





# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')
    args = get_args()
    wandb.login()

    print(f"\nProject : {args.project}")
    print(f"Entity  : {args.entity}")
    print(f"Dataset : {args.dataset}\n")

    run_2_1 (args.project, args.entity, args.dataset)
    if not args.skip_sweep:
        run_2_2_sweep(args.project, args.entity, args.dataset)
    run_2_3 (args.project, args.entity, args.dataset)
    run_2_4 (args.project, args.entity, args.dataset)
    run_2_5 (args.project, args.entity, args.dataset)
    run_2_6 (args.project, args.entity, args.dataset)
    run_2_7 (args.project, args.entity)
    run_2_8 (args.project, args.entity, args.dataset)
    run_2_9 (args.project, args.entity, args.dataset)
    run_2_10(args.project, args.entity)

    url = build_report(args.project, args.entity)
    print(f"\n{'='*55}")
    print(f"  All experiments done. Report URL:")
    print(f"  {url}")
    print(f"{'='*55}\n")

if __name__ == "__main__":
    main()
