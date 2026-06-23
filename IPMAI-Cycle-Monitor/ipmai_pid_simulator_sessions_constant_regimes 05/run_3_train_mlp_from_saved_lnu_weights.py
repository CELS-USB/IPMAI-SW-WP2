# -*- coding: utf-8 -*-
"""
IPMAI MLP classifier trained from previously saved LNU weights.

This script does NOT run the simulator and does NOT train LNU models.
It only reads a previously saved weights table, e.g.

    lnu_run_sequence_analysis_results/sequence_weights.csv

and trains a feed-forward MLP classifier:

    LNU weights -> probabilities of operating state / cycle type

Author: IPMAI working script
"""

# ============================================================
# USER SETUP - keep all important settings here
# ============================================================

# ---- input / output ----
WEIGHTS_CSV = "lnu_run_sequence_analysis_results/sequence_weights.csv"
OUTPUT_DIR = "mlp_from_saved_lnu_weights_results"

# ---- column detection ----
WEIGHT_PREFIX = "w_"

# Prefer these columns as labels if available, in this order.
# The script falls back to automatic detection if none is found.
LABEL_COLUMN_CANDIDATES = [
    "fault_name",
    "state_name",
    "run_state",
    "regime_name",
    "behavior",
    "cycle_type",
    "run_name",
    "run_id",
]

# Optional columns to keep in outputs if present.
META_COLUMN_CANDIDATES = [
    "sequence_index",
    "run_name",
    "run_id",
    "fault_id",
    "fault_name",
    "state_name",
    "magnitude",
    "trial",
]

# ---- feature preprocessing ----
STANDARDIZE_WEIGHTS = True
STANDARDIZATION_EPS = 1e-12

# ---- train/test split ----
# With very small data, USE_TRAIN_TEST_SPLIT=False is often more useful
# just to verify that the pipeline trains and exports probabilities.
USE_TRAIN_TEST_SPLIT = False
TEST_SIZE = 0.30
RANDOM_SEED = 123

# ---- MLP architecture ----
HIDDEN_LAYER_SIZES = ([5])  # ([5]), or for more layers (5,3)
ACTIVATION = "tanh"          # "relu", "tanh", "logistic", "identity"
SOLVER = "lbfgs"             # "lbfgs" is robust for small datasets; "adam" for larger data
ALPHA = 1e-4                  # L2 regularization
MAX_ITER = 5000
LEARNING_RATE_INIT = 1e-3     # used mainly by adam/sgd

# ---- plotting ----
MAKE_PLOTS = True
PLOT_DPI = 140

# ============================================================
# Imports
# ============================================================

from pathlib import Path
import json
import pickle
import warnings

import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import matplotlib.pyplot as plt


# ============================================================
# Helpers
# ============================================================


def find_weight_columns(df, prefix=WEIGHT_PREFIX):
    cols = [c for c in df.columns if str(c).startswith(prefix)]
    if not cols:
        raise ValueError(
            f"No weight columns found with prefix {prefix!r}. "
            "Expected columns such as w_0, w_1, ..."
        )

    def key_fun(c):
        try:
            return int(str(c).split(prefix, 1)[1])
        except Exception:
            return str(c)

    return sorted(cols, key=key_fun)


def find_label_column(df):
    for c in LABEL_COLUMN_CANDIDATES:
        if c in df.columns:
            return c

    # Fallback: any non-weight object/string column with more than one unique value.
    weight_cols = set(find_weight_columns(df))
    for c in df.columns:
        if c in weight_cols:
            continue
        if df[c].dtype == object or str(df[c].dtype).startswith("category"):
            if df[c].nunique(dropna=True) > 1:
                return c

    raise ValueError(
        "Could not determine the label column. Add one of these columns to the weights CSV: "
        + ", ".join(LABEL_COLUMN_CANDIDATES)
    )


def standardize_train_apply(X):
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std < STANDARDIZATION_EPS] = 1.0
    return (X - mean) / std, mean.ravel(), std.ravel()


def make_confusion_plot(cm, labels, out_path):
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(labels)), max(5, 0.6 * len(labels))))
    im = ax.imshow(cm, aspect="auto")
    ax.set_title("Confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=PLOT_DPI)
    plt.close(fig)


def make_probability_plot(prob_df, class_names, out_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(prob_df))
    for cls in class_names:
        ax.plot(x, prob_df[f"P_{cls}"].to_numpy(), marker="o", linewidth=1.5, label=cls)
    ax.set_title("MLP output probabilities for saved LNU-weight samples")
    ax.set_xlabel("sample index")
    ax.set_ylabel("probability")
    ax.grid(True)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=PLOT_DPI)
    plt.close(fig)


# ============================================================
# Main
# ============================================================


def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_path = Path(WEIGHTS_CSV)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights CSV not found: {weights_path}\n"
            "Set WEIGHTS_CSV at the top of this script."
        )

    df = pd.read_csv(weights_path)
    weight_cols = find_weight_columns(df)
    label_col = find_label_column(df)
    meta_cols = [c for c in META_COLUMN_CANDIDATES if c in df.columns]

    X_raw = df[weight_cols].to_numpy(dtype=float)
    labels_raw = df[label_col].astype(str).to_numpy()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels_raw)
    class_names = list(label_encoder.classes_)

    if len(class_names) < 2:
        raise ValueError("At least two classes/states are needed to train a classifier.")

    if STANDARDIZE_WEIGHTS:
        X, x_mean, x_std = standardize_train_apply(X_raw)
    else:
        X = X_raw.copy()
        x_mean = np.zeros(X.shape[1])
        x_std = np.ones(X.shape[1])

    clf = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYER_SIZES,
        activation=ACTIVATION,
        solver=SOLVER,
        alpha=ALPHA,
        max_iter=MAX_ITER,
        learning_rate_init=LEARNING_RATE_INIT,
        random_state=RANDOM_SEED,
    )

    results_summary = {
        "weights_csv": str(weights_path),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": int(len(class_names)),
        "label_column": label_col,
        "classes": class_names,
        "weight_columns": weight_cols,
        "standardize_weights": bool(STANDARDIZE_WEIGHTS),
        "hidden_layer_sizes": list(HIDDEN_LAYER_SIZES),
        "activation": ACTIVATION,
        "solver": SOLVER,
        "alpha": ALPHA,
        "max_iter": MAX_ITER,
        "use_train_test_split": bool(USE_TRAIN_TEST_SPLIT),
    }

    if USE_TRAIN_TEST_SPLIT and X.shape[0] >= 4 and min(np.bincount(y)) >= 2:
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X,
            y,
            np.arange(X.shape[0]),
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            stratify=y,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(class_names)))
        report = classification_report(
            y_test,
            y_pred,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            zero_division=0,
        )
        results_summary["test_accuracy"] = float(acc)
        results_summary["test_indices"] = idx_test.tolist()
    else:
        if USE_TRAIN_TEST_SPLIT:
            warnings.warn(
                "Requested train/test split, but the dataset is too small or some classes have only one sample. "
                "Training on all samples instead."
            )
        clf.fit(X, y)
        y_pred = clf.predict(X)
        acc = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred, labels=np.arange(len(class_names)))
        report = classification_report(
            y,
            y_pred,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            zero_division=0,
        )
        results_summary["training_accuracy"] = float(acc)

    prob = clf.predict_proba(X)
    pred_idx = np.argmax(prob, axis=1)
    pred_label = label_encoder.inverse_transform(pred_idx)

    prob_df = pd.DataFrame()
    for c in meta_cols:
        prob_df[c] = df[c]
    prob_df["true_label"] = labels_raw
    prob_df["predicted_label"] = pred_label
    for i, cls in enumerate(class_names):
        prob_df[f"P_{cls}"] = prob[:, i]

    prob_df.to_csv(out_dir / "mlp_probabilities_from_lnu_weights.csv", index=False)

    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(out_dir / "confusion_matrix.csv")

    with open(out_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    with open(out_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    model_bundle = {
        "model": clf,
        "label_encoder": label_encoder,
        "weight_columns": weight_cols,
        "label_column": label_col,
        "x_mean": x_mean,
        "x_std": x_std,
        "standardize_weights": STANDARDIZE_WEIGHTS,
        "config": results_summary,
    }
    with open(out_dir / "mlp_lnu_weight_classifier.pkl", "wb") as f:
        pickle.dump(model_bundle, f)

    if MAKE_PLOTS:
        make_confusion_plot(cm, class_names, out_dir / "confusion_matrix.png")
        make_probability_plot(prob_df, class_names, out_dir / "mlp_output_probabilities.png")

    print("Done.")
    print(f"Input weights CSV: {weights_path}")
    print(f"Label column: {label_col}")
    print(f"Samples: {X.shape[0]}, features: {X.shape[1]}, classes: {len(class_names)}")
    print("Classes:", class_names)
    if "training_accuracy" in results_summary:
        print(f"Training accuracy: {results_summary['training_accuracy']:.4f}")
    if "test_accuracy" in results_summary:
        print(f"Test accuracy: {results_summary['test_accuracy']:.4f}")
    print(f"Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
