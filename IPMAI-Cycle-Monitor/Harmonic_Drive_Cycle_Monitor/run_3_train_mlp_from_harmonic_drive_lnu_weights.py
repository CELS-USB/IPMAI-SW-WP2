# -*- coding: utf-8 -*-
"""
IPMAI Harmonic Drive MLP classifier trained from saved LNU weights.

This is the Harmonic Drive equivalent of the PID-simulation MLP stage. It does not
simulate data and it does not retrain LNU models. It reads the output of

    run_2_harmonic_drive_lnu_cycle_monitor.py

especially

    harmonic_drive_lnu_cycle_monitor_results/weights_by_file.csv

and trains a feed-forward MLP classifier/regime recognizer using the final LNU
weights as features. Because the true healthy/faulty label is unknown for Harmonic Drive,
the default target label is derived from the file name metadata, not from health.

Typical pipeline:
    data_harmonic_drive_cycle_monitor/*.csv
        -> run_2_harmonic_drive_lnu_cycle_monitor.py
        -> weights_by_file.csv
        -> run_3_train_mlp_from_harmonic_drive_lnu_weights.py
        -> MLP probabilities over file-name-derived regimes

Windows usage:
    python run_3_train_mlp_from_harmonic_drive_lnu_weights.py

Authoring convention: all user setups are intentionally kept at the top.
"""

# ============================================================
# USER SETUP
# ============================================================

from pathlib import Path
import json
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ---- Input / output ----
WEIGHTS_CSV = Path("harmonic_drive_lnu_cycle_monitor_results") / "weights_by_file.csv"
OUTPUT_DIR = Path("harmonic_drive_mlp_from_lnu_weights_results")

# ---- Feature columns ----
WEIGHT_PREFIX = "w_"
STANDARDIZE_WEIGHTS = True
STANDARDIZATION_EPS = 1e-12

# ---- Target label construction ----
# Since health labels are unknown, choose how the MLP should distinguish runs.
# Options:
#   "file_name"        -> one class per CSV file name
#   "run_name"         -> one class per file stem
#   "load_angle"       -> class like load_0000g_angle_060d if metadata exist
#   "load"             -> class by load_g only
#   "angle"            -> class by angle_deg only
#   "custom_column"    -> use CUSTOM_LABEL_COLUMN from weights CSV
LABEL_MODE = "file_name"
CUSTOM_LABEL_COLUMN = "label"

# Optional: remove the reference file from MLP training.
# Usually False, because it can still be a useful operating class.
EXCLUDE_REFERENCE_FILE = False

# Optional metadata columns copied to output probability table if present.
META_COLUMN_CANDIDATES = [
    "file_name", "run_name", "is_reference_file",
    "N_samples", "N_training_samples", "dt_median_sec", "Q_final", "RMSE_norm", "MAE_norm",
]

# ---- Train/test split ----
# With few harmonic-drive files and possibly one sample per class, False is often more useful
# for verifying the pipeline and exporting class probabilities. Use True only when
# each class has enough examples.
USE_TRAIN_TEST_SPLIT = False
TEST_SIZE = 0.30
RANDOM_SEED = 123

# ---- MLP architecture ----
HIDDEN_LAYER_SIZES = (8,)       # e.g. (8,), (12, 6)
ACTIVATION = "tanh"            # "relu", "tanh", "logistic", "identity"
SOLVER = "lbfgs"               # "lbfgs" robust for small datasets; "adam" for larger data
ALPHA = 1e-4
MAX_ITER = 5000
LEARNING_RATE_INIT = 1e-3

# ---- Plotting ----
MAKE_PLOTS = True
PLOT_DPI = 140
CSV_FLOAT_FORMAT = "%.10g"

# ============================================================
# IMPLEMENTATION
# ============================================================


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_weight_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if str(c).startswith(WEIGHT_PREFIX)]
    if not cols:
        raise ValueError(
            f"No LNU weight columns found with prefix {WEIGHT_PREFIX!r}. "
            "Expected columns such as w_0, w_1, ... in weights_by_file.csv."
        )

    def key_fun(c):
        try:
            return int(str(c).split(WEIGHT_PREFIX, 1)[1])
        except Exception:
            return str(c)

    return sorted(cols, key=key_fun)


def make_labels(df: pd.DataFrame) -> tuple[np.ndarray, str]:
    if LABEL_MODE == "custom_column":
        if CUSTOM_LABEL_COLUMN not in df.columns:
            raise ValueError(f"CUSTOM_LABEL_COLUMN={CUSTOM_LABEL_COLUMN!r} not found in weights CSV.")
        return df[CUSTOM_LABEL_COLUMN].astype(str).to_numpy(), CUSTOM_LABEL_COLUMN

    if LABEL_MODE == "file_name":
        return df["file_name"].astype(str).to_numpy(), "file_name"

    if LABEL_MODE == "run_name":
        return df["run_name"].astype(str).to_numpy(), "run_name"

    if LABEL_MODE == "load_angle":
        if "load_g" not in df.columns or "angle_deg" not in df.columns:
            raise ValueError("LABEL_MODE='load_angle' requires load_g and angle_deg columns.")
        labels = []
        for load, angle, fname in zip(df["load_g"], df["angle_deg"], df.get("file_name", df.index)):
            if pd.isna(load) or pd.isna(angle):
                labels.append(str(fname))
            else:
                labels.append(f"load_{int(load):04d}g_angle_{int(angle):03d}d")
        return np.asarray(labels, dtype=str), "derived_load_angle"

    if LABEL_MODE == "load":
        if "load_g" not in df.columns:
            raise ValueError("LABEL_MODE='load' requires load_g column.")
        labels = [f"load_{int(x):04d}g" if not pd.isna(x) else "load_unknown" for x in df["load_g"]]
        return np.asarray(labels, dtype=str), "derived_load"

    if LABEL_MODE == "angle":
        if "angle_deg" not in df.columns:
            raise ValueError("LABEL_MODE='angle' requires angle_deg column.")
        labels = [f"angle_{int(x):03d}d" if not pd.isna(x) else "angle_unknown" for x in df["angle_deg"]]
        return np.asarray(labels, dtype=str), "derived_angle"

    raise ValueError("Unsupported LABEL_MODE. Use file_name, run_name, load_angle, load, angle, or custom_column.")


def standardize_train_apply(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std < STANDARDIZATION_EPS] = 1.0
    return (X - mean) / std, mean.ravel(), std.ravel()


def make_confusion_plot(cm: np.ndarray, labels: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(labels)), max(5, 0.6 * len(labels))))
    im = ax.imshow(cm, aspect="auto")
    ax.set_title("Harmonic Drive MLP confusion matrix from LNU weights")
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


def make_probability_plot(prob_df: pd.DataFrame, class_names: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(prob_df))
    for cls in class_names:
        ax.plot(x, prob_df[f"P_{cls}"].to_numpy(), marker="o", linewidth=1.5, label=cls)
    ax.set_title("Harmonic Drive MLP output probabilities from saved LNU weights")
    ax.set_xlabel("file/run index")
    ax.set_ylabel("probability")
    ax.grid(True)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=PLOT_DPI)
    plt.close(fig)


def main() -> None:
    ensure_dir(OUTPUT_DIR)

    if not WEIGHTS_CSV.exists():
        raise FileNotFoundError(
            f"Weights CSV not found: {WEIGHTS_CSV}\n"
            "Run run_2_harmonic_drive_lnu_cycle_monitor.py first, or set WEIGHTS_CSV at the top of this script."
        )

    df = pd.read_csv(WEIGHTS_CSV)
    if EXCLUDE_REFERENCE_FILE and "is_reference_file" in df.columns:
        df = df.loc[~df["is_reference_file"].astype(bool)].reset_index(drop=True)

    weight_cols = find_weight_columns(df)
    labels_raw, label_source = make_labels(df)
    meta_cols = [c for c in META_COLUMN_CANDIDATES if c in df.columns]

    X_raw = df[weight_cols].to_numpy(dtype=float)
    if not np.isfinite(X_raw).all():
        raise ValueError("Weight matrix contains NaN or Inf values.")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels_raw)
    class_names = list(label_encoder.classes_)

    if len(class_names) < 2:
        raise ValueError(
            "At least two classes are needed to train MLP. "
            "Change LABEL_MODE or add more harmonic-drive files."
        )

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
        "weights_csv": str(WEIGHTS_CSV),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": int(len(class_names)),
        "label_mode": LABEL_MODE,
        "label_source": label_source,
        "classes": class_names,
        "weight_columns": weight_cols,
        "standardize_weights": bool(STANDARDIZE_WEIGHTS),
        "hidden_layer_sizes": list(HIDDEN_LAYER_SIZES),
        "activation": ACTIVATION,
        "solver": SOLVER,
        "alpha": ALPHA,
        "max_iter": MAX_ITER,
        "use_train_test_split": bool(USE_TRAIN_TEST_SPLIT),
        "exclude_reference_file": bool(EXCLUDE_REFERENCE_FILE),
        "note": "Labels are filename/metadata-derived because true healthy/faulty labels are unknown for Harmonic Drive.",
    }

    can_split = USE_TRAIN_TEST_SPLIT and X.shape[0] >= 4 and min(np.bincount(y)) >= 2
    if can_split:
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X,
            y,
            np.arange(X.shape[0]),
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            stratify=y,
        )
        clf.fit(X_train, y_train)
        y_eval = y_test
        y_pred_eval = clf.predict(X_test)
        results_summary["test_indices"] = idx_test.tolist()
        results_summary["test_accuracy"] = float(accuracy_score(y_eval, y_pred_eval))
    else:
        if USE_TRAIN_TEST_SPLIT:
            warnings.warn(
                "Requested train/test split, but the dataset is too small or some classes have only one sample. "
                "Training/evaluating on all samples instead."
            )
        clf.fit(X, y)
        y_eval = y
        y_pred_eval = clf.predict(X)
        results_summary["training_accuracy"] = float(accuracy_score(y_eval, y_pred_eval))

    cm = confusion_matrix(y_eval, y_pred_eval, labels=np.arange(len(class_names)))
    report = classification_report(
        y_eval,
        y_pred_eval,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        zero_division=0,
    )

    prob = clf.predict_proba(X)
    pred_idx = np.argmax(prob, axis=1)
    pred_label = label_encoder.inverse_transform(pred_idx)

    prob_df = pd.DataFrame()
    for c in meta_cols:
        prob_df[c] = df[c]
    prob_df["target_label"] = labels_raw
    prob_df["predicted_label"] = pred_label
    for i, cls in enumerate(class_names):
        prob_df[f"P_{cls}"] = prob[:, i]
    prob_df.to_csv(OUTPUT_DIR / "mlp_probabilities_from_harmonic_drive_lnu_weights.csv", index=False, float_format=CSV_FLOAT_FORMAT)

    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(OUTPUT_DIR / "confusion_matrix.csv")
    with open(OUTPUT_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    with open(OUTPUT_DIR / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)

    model_bundle = {
        "model": clf,
        "label_encoder": label_encoder,
        "weight_columns": weight_cols,
        "label_mode": LABEL_MODE,
        "label_source": label_source,
        "x_mean": x_mean,
        "x_std": x_std,
        "standardize_weights": STANDARDIZE_WEIGHTS,
        "config": results_summary,
    }
    with open(OUTPUT_DIR / "mlp_harmonic_drive_lnu_weight_classifier.pkl", "wb") as f:
        pickle.dump(model_bundle, f)

    if MAKE_PLOTS:
        make_confusion_plot(cm, class_names, OUTPUT_DIR / "confusion_matrix.png")
        make_probability_plot(prob_df, class_names, OUTPUT_DIR / "mlp_output_probabilities.png")

    print("Done.")
    print(f"Input weights CSV: {WEIGHTS_CSV}")
    print(f"Label mode: {LABEL_MODE} ({label_source})")
    print(f"Samples: {X.shape[0]}, features: {X.shape[1]}, classes: {len(class_names)}")
    print("Classes:", class_names)
    if "training_accuracy" in results_summary:
        print(f"Training accuracy: {results_summary['training_accuracy']:.4f}")
    if "test_accuracy" in results_summary:
        print(f"Test accuracy: {results_summary['test_accuracy']:.4f}")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
