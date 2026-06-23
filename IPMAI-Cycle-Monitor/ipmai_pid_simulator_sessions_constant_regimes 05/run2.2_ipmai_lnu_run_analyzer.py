"""
IPMAI LNU sequential run analyzer for simulated run folders.

Purpose
-------
This script reads already generated IPMAI simulator runs in folders such as

    simulated_runs/sim_healthy_001/measured.csv
    simulated_runs/sim_ripple_001/measured.csv
    ...

It does NOT simulate data. It trains one LNU on each complete run, using:

    d(k) = desired variable      -> theta_ref
    y(k) = controlled variable   -> theta
    u(k) = control variable      -> u

The healthy run is used to define:
    1) z-score mean/std for all LNU input/output variables,
    2) one common PCA basis of the delay-vector x(k).

Then every run is transformed by the same z-score and the same PCA basis.
Therefore all final LNU weight vectors have the same length and comparable
meaning across runs.

Outputs are written to OUTPUT_DIR:
    weights_by_run.csv
    predictions_summary.csv
    learning_entropy_vs_healthy.csv
    saliency_vs_healthy.csv
    normalization_reference.csv
    pca_reference.npz
    lnu_run_dataset.npz
    plots/*.png

Authoring convention: all user setups are intentionally kept at the top.
"""

# ============================================================
# USER SETUP
# ============================================================

from pathlib import Path
import json
import math
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Input/output folders ----
DATA_ROOT = Path("simulated_runs")
OUTPUT_DIR = Path("lnu_run_sequence_analysis_results")
MEASURED_FILE_NAME = "measured.csv"
ANNOTATED_FILE_NAME = "annotated.csv"
EVENTS_FILE_NAME = "events.csv"

# ---- Run selection ----
RUN_GLOB = "sim_*"
HEALTHY_RUN_KEYWORD = "healthy"
SORT_RUNS_BY = "fault_then_name"  # "name" or "fault_then_name"

# ---- Sequential experiment protocol ----
# The same LNU is trained continuously through this sequence:
# healthy -> fault_1 -> healthy -> fault_2 -> healthy -> ...
# The healthy run is repeated as a return-to-baseline segment between faults.
SEQUENTIAL_PROTOCOL = True
ADD_FINAL_HEALTHY_SEGMENT = False

# ---- Variables from measured.csv ----
TIME_COLUMN = "t"
DESIRED_COLUMN = "theta_ref"     # d(k)
CONTROL_COLUMN = "u"             # u(k), current/actuator-like control quantity
OUTPUT_COLUMN = "theta"          # y(k), controlled variable

# Optional extra measured columns to include in the LNU delay vector.
# For a first clean Robotic-like conversion keep this empty.
# Later, for vibro-aware modeling, use e.g. ["hf_rms", "a2x", "amesh"].
EXTRA_INPUT_COLUMNS = []
N_EXTRA_DELAYS = 1

# ---- Preprocessing ----
DN = 1                            # use every DN-th sample
USE_HEALTHY_ONLY_FOR_NORMALIZATION = True
ZSCORE_SCALE = 3.0                # Robotic convention: (x - mean) / (3*std)
STD_EPS = 1e-12

# ---- LNU/HONU structure ----
R = 1                             # 1=LNU. QNU/CNU scaffolding is implemented, but R=1 is recommended here.
ND = 2                            # d[k-ND], ..., d[k-1]
NU = 9                            # u[k-(L_U+NU-1)], ..., u[k-L_U]
NY = 5                            # y[k-NY], ..., y[k-1]
L_U = 3                           # control delay; L_U=1 means u[k-NU],...,u[k-1]

# ---- PCA over delay vector x without bias ----
USE_PCA = True
PCA_VAR_THRESHOLD = 0.995
NPCA_MAX = 10000                  # max healthy delay vectors for PCA
MIN_PCA_COMPONENTS = 1
MAX_PCA_COMPONENTS = None         # set integer to force an upper bound; None = threshold only

# ---- Batch learning on one complete run ----
MU = 1000.0
EPOCHS = 20 * R**3
WEIGHT_INIT_STD = 0.0             # 0 -> deterministic zero initialization
RANDOM_SEED = 123
RIDGE_LAMBDA = 0.0                # optional ridge term in batch update matrix

# ---- Diagnostics ----
PLOT_RESULTS = True
SAVE_PLOTS = True
SHOW_PLOTS = False                # keep False for batch execution / Windows CLI use
PLOT_DPI = 140

# ============================================================
# IMPLEMENTATION
# ============================================================


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize_name(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_\-]+", "_", name)
    return name.strip("_")


def find_run_dirs(data_root: Path) -> list[Path]:
    if not data_root.exists():
        raise FileNotFoundError(f"DATA_ROOT does not exist: {data_root.resolve()}")
    run_dirs = [p for p in data_root.glob(RUN_GLOB) if p.is_dir() and (p / MEASURED_FILE_NAME).exists()]
    if not run_dirs:
        raise FileNotFoundError(f"No run folders with {MEASURED_FILE_NAME} found under {data_root.resolve()}")
    return run_dirs


def read_fault_metadata(run_dir: Path) -> dict:
    meta = {
        "run_name": run_dir.name,
        "fault_id": np.nan,
        "fault_name": "unknown",
        "fault_magnitude": np.nan,
    }
    ann_path = run_dir / ANNOTATED_FILE_NAME
    if ann_path.exists():
        try:
            ann = pd.read_csv(ann_path, nrows=20)
            for col in ["fault_id", "fault_name", "fault_magnitude"]:
                if col in ann.columns and len(ann[col]) > 0:
                    meta[col] = ann[col].iloc[0]
            return meta
        except Exception:
            pass

    events_path = run_dir / EVENTS_FILE_NAME
    if events_path.exists():
        try:
            ev = pd.read_csv(events_path)
            if len(ev) > 0:
                for col in ["fault_id", "fault_name", "fault_magnitude"]:
                    if col in ev.columns:
                        meta[col] = ev[col].iloc[0]
        except Exception:
            pass
    return meta


def sort_run_dirs(run_dirs: list[Path]) -> list[Path]:
    if SORT_RUNS_BY == "name":
        return sorted(run_dirs, key=lambda p: p.name)
    rows = []
    for p in run_dirs:
        m = read_fault_metadata(p)
        fid = m.get("fault_id", np.nan)
        fid_sort = 9999 if pd.isna(fid) else int(fid)
        rows.append((fid_sort, p.name, p))
    return [r[2] for r in sorted(rows)]


def load_measured_run(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / MEASURED_FILE_NAME)
    if DN > 1:
        df = df.iloc[::DN, :].reset_index(drop=True)
    required = [TIME_COLUMN, DESIRED_COLUMN, CONTROL_COLUMN, OUTPUT_COLUMN] + list(EXTRA_INPUT_COLUMNS)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {run_dir / MEASURED_FILE_NAME}: {missing}")
    return df


def choose_healthy_run(run_dirs: list[Path]) -> Path:
    candidates = [p for p in run_dirs if HEALTHY_RUN_KEYWORD.lower() in p.name.lower()]
    if not candidates:
        # fallback from annotation
        for p in run_dirs:
            m = read_fault_metadata(p)
            if str(m.get("fault_name", "")).lower() == HEALTHY_RUN_KEYWORD.lower() or m.get("fault_id", None) == 0:
                candidates.append(p)
    if not candidates:
        raise ValueError("Cannot find healthy run. Rename one run to include 'healthy' or set HEALTHY_RUN_KEYWORD.")
    return sorted(candidates, key=lambda p: p.name)[0]


def selected_signal_columns() -> list[str]:
    cols = [DESIRED_COLUMN, CONTROL_COLUMN, OUTPUT_COLUMN] + list(EXTRA_INPUT_COLUMNS)
    seen = []
    for c in cols:
        if c not in seen:
            seen.append(c)
    return seen


def compute_normalization_reference(healthy_df: pd.DataFrame) -> pd.DataFrame:
    cols = selected_signal_columns()
    rows = []
    for c in cols:
        vals = healthy_df[c].to_numpy(dtype=float)
        mu = float(np.mean(vals))
        sd = float(np.std(vals))
        if sd < STD_EPS:
            sd = 1.0
        rows.append({"column": c, "mean": mu, "std": sd, "zscore_scale": ZSCORE_SCALE})
    return pd.DataFrame(rows)


def apply_normalization(df: pd.DataFrame, norm_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for _, row in norm_df.iterrows():
        c = row["column"]
        out[c] = (out[c].to_numpy(dtype=float) - float(row["mean"])) / (ZSCORE_SCALE * float(row["std"]))
    return out


def start_index() -> int:
    starts = [ND, NY, L_U + NU - 1]
    if EXTRA_INPUT_COLUMNS:
        starts.append(N_EXTRA_DELAYS - 1)
    return max(starts)


def build_delay_matrix_from_arrays(d: np.ndarray, u: np.ndarray, y: np.ndarray, extras: dict[str, np.ndarray] | None = None):
    """Build Robotic-style delay matrix without bias.

    Row at sample k contains:
        d[k-ND], ..., d[k-1]
        u[k-(L_U+NU-1)], ..., u[k-L_U]
        y[k-NY], ..., y[k-1]
        optional extras: q[k-(N_EXTRA_DELAYS-1)], ..., q[k]
    """
    N = len(y)
    k0 = start_index()
    if N <= k0:
        raise ValueError(f"Run too short: N={N}, required start index={k0}")

    n_extra_total = len(EXTRA_INPUT_COLUMNS) * N_EXTRA_DELAYS
    nx_nobias = ND + NU + NY + n_extra_total
    X = np.zeros((N - k0, nx_nobias), dtype=float)
    labels = []

    labels += [f"d[k-{ND-i}]" for i in range(ND)]
    labels += [f"u[k-{L_U + NU - i - 1}]" for i in range(NU)]
    labels += [f"y[k-{NY-i}]" for i in range(NY)]
    for c in EXTRA_INPUT_COLUMNS:
        labels += [f"{c}[k-{N_EXTRA_DELAYS-i-1}]" for i in range(N_EXTRA_DELAYS)]

    for row_i, k in enumerate(range(k0, N)):
        pos = 0
        X[row_i, pos:pos + ND] = d[k - ND:k]
        pos += ND

        start_u = k - L_U - NU + 1
        end_u = k - L_U + 1
        X[row_i, pos:pos + NU] = u[start_u:end_u]
        pos += NU

        X[row_i, pos:pos + NY] = y[k - NY:k]
        pos += NY

        if EXTRA_INPUT_COLUMNS:
            extras = extras or {}
            for c in EXTRA_INPUT_COLUMNS:
                arr = extras[c]
                for lag_i in range(N_EXTRA_DELAYS):
                    delay = N_EXTRA_DELAYS - lag_i - 1
                    X[row_i, pos] = arr[k - delay]
                    pos += 1

    return X, labels, k0


def build_delay_matrix_from_df(df_norm: pd.DataFrame):
    d = df_norm[DESIRED_COLUMN].to_numpy(dtype=float)
    u = df_norm[CONTROL_COLUMN].to_numpy(dtype=float)
    y = df_norm[OUTPUT_COLUMN].to_numpy(dtype=float)
    extras = {c: df_norm[c].to_numpy(dtype=float) for c in EXTRA_INPUT_COLUMNS}
    return build_delay_matrix_from_arrays(d, u, y, extras)


def fit_pca_from_healthy(X_healthy: np.ndarray):
    Xp = X_healthy[:min(NPCA_MAX, X_healthy.shape[0]), :]
    pca_mean = Xp.mean(axis=0)
    Xc = Xp - pca_mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    eigvals = (S ** 2) / max(1, Xc.shape[0] - 1)
    total = float(np.sum(eigvals))
    if total <= 0:
        cumvar = np.ones_like(eigvals)
    else:
        cumvar = np.cumsum(eigvals) / total
    n_comp = int(np.searchsorted(cumvar, PCA_VAR_THRESHOLD) + 1)
    n_comp = max(MIN_PCA_COMPONENTS, n_comp)
    if MAX_PCA_COMPONENTS is not None:
        n_comp = min(int(MAX_PCA_COMPONENTS), n_comp)
    n_comp = min(n_comp, Vt.shape[0])
    components = Vt[:n_comp, :].T
    return pca_mean, components, eigvals, cumvar, n_comp


def transform_design(X_nobias: np.ndarray, pca_mean: np.ndarray, pca_components: np.ndarray):
    if USE_PCA:
        Z = (X_nobias - pca_mean) @ pca_components
        return np.column_stack([np.ones(Z.shape[0]), Z])
    return np.column_stack([np.ones(X_nobias.shape[0]), X_nobias])


def polynomial_structure(n_base: int, r: int):
    structure = []
    if r == 1:
        for i in range(n_base):
            structure.append([i])
    elif r == 2:
        for i in range(n_base):
            for j in range(i, n_base):
                structure.append([i, j])
    elif r == 3:
        for i in range(n_base):
            for j in range(i, n_base):
                for k in range(j, n_base):
                    structure.append([i, j, k])
    else:
        raise ValueError("R must be 1, 2, or 3.")
    return structure


def expand_honu_columns(X_base: np.ndarray, r: int):
    if r == 1:
        return X_base
    n_samples, n_base = X_base.shape
    struct = polynomial_structure(n_base, r)
    Xh = np.zeros((n_samples, len(struct)), dtype=float)
    for q, idxs in enumerate(struct):
        prod = np.ones(n_samples)
        for idx in idxs:
            prod *= X_base[:, idx]
        Xh[:, q] = prod
    return Xh


def train_lnu_batch(J: np.ndarray, y_target: np.ndarray, rng: np.random.Generator):
    nw = J.shape[1]
    if WEIGHT_INIT_STD > 0:
        w = rng.normal(0.0, WEIGHT_INIT_STD, size=nw)
    else:
        w = np.zeros(nw)

    L = np.eye(nw)
    Q = np.zeros(EPOCHS, dtype=float)
    Wall = np.zeros((EPOCHS + 1, nw), dtype=float)
    dWall = np.zeros((EPOCHS + 1, nw), dtype=float)
    Wall[0, :] = w

    A = J.T @ J + (1.0 / MU) * L + RIDGE_LAMBDA * L
    A_inv_JT = np.linalg.pinv(A) @ J.T

    for epoch in range(EPOCHS):
        yn = J @ w
        e = y_target - yn
        dw = A_inv_JT @ e
        w = w + dw
        Wall[epoch + 1, :] = w
        dWall[epoch + 1, :] = dw
        e2 = y_target - J @ w
        Q[epoch] = float(np.sum(e2 ** 2))

    yn = J @ w
    e = y_target - yn
    return w, Q, Wall, dWall, yn, e


def reconstruct_saliency_from_delta(delta_w: np.ndarray, pca_components: np.ndarray, x_labels: list[str], n_base: int):
    """Map absolute weight change back to original embedding coordinates.

    For R=1 and PCA, this maps each PCA-space weight via abs(P). For R>1, it
    distributes monomial weight changes using the same product-factor heuristic
    as the Robotic saliency script.
    """
    n_embed = len(x_labels)
    structure = polynomial_structure(n_base, R)
    if len(delta_w) != len(structure):
        raise ValueError("delta_w length does not match HONU polynomial structure.")

    if USE_PCA:
        P_matrix = pca_components
    else:
        P_matrix = np.eye(n_embed)

    contrib = np.zeros(n_embed, dtype=float)
    abs_delta_w = np.abs(delta_w)

    for q, base_idx_list in enumerate(structure):
        z_indices = [idx - 1 for idx in base_idx_list if idx > 0]  # skip bias index 0
        if not z_indices:
            continue

        factors = np.ones(n_embed, dtype=float)
        for z_idx in z_indices:
            if z_idx >= P_matrix.shape[1]:
                factors[:] = 0.0
                break
            factors *= np.abs(P_matrix[:, z_idx])

        s = float(factors.sum())
        if s > 0:
            contrib += abs_delta_w[q] * (factors / s)

    return contrib


def make_training_sequence(run_dirs: list[Path], healthy_run: Path) -> list[Path]:
    """Create healthy -> fault_1 -> healthy -> fault_2 -> ... sequence.

    The healthy run is intentionally reused multiple times. This is not a new
    simulation; it is an experimental protocol for sequential LNU adaptation.
    """
    nonhealthy = [p for p in run_dirs if p.name != healthy_run.name]
    nonhealthy = sort_run_dirs(nonhealthy)
    seq = [healthy_run]
    for p in nonhealthy:
        seq.append(p)
        if ADD_FINAL_HEALTHY_SEGMENT or p != nonhealthy[-1]:
            seq.append(healthy_run)
    return seq


def main():
    ensure_dir(OUTPUT_DIR)
    plot_dir = OUTPUT_DIR / "plots"
    ensure_dir(plot_dir)

    run_dirs = sort_run_dirs(find_run_dirs(DATA_ROOT))
    healthy_run = choose_healthy_run(run_dirs)
    print(f"Healthy reference run: {healthy_run.name}")
    print(f"Found {len(run_dirs)} unique runs.")

    # Normalization and PCA are always fitted from the healthy reference run.
    healthy_df = load_measured_run(healthy_run)
    norm_df = compute_normalization_reference(healthy_df)
    norm_df.to_csv(OUTPUT_DIR / "normalization_reference.csv", index=False)

    healthy_norm = apply_normalization(healthy_df, norm_df)
    Xh_nobias, x_labels, k0 = build_delay_matrix_from_df(healthy_norm)
    pca_mean, pca_components, pca_eigvals, pca_cumvar, n_comp = fit_pca_from_healthy(Xh_nobias)

    np.savez(
        OUTPUT_DIR / "pca_reference.npz",
        pca_mean=pca_mean,
        pca_components=pca_components,
        pca_eigvals=pca_eigvals,
        pca_cumvar=pca_cumvar,
        x_labels=np.array(x_labels, dtype=object),
        use_pca=USE_PCA,
        n_components=n_comp,
    )

    print(f"Delay vector dimension without bias: {len(x_labels)}")
    print(f"PCA components used: {n_comp}, captured variance: {pca_cumvar[n_comp-1] * 100:.3f}%")

    if SEQUENTIAL_PROTOCOL:
        train_sequence = make_training_sequence(run_dirs, healthy_run)
    else:
        train_sequence = run_dirs

    print("\nTraining sequence:")
    for i, p in enumerate(train_sequence):
        print(f"  {i:03d}: {p.name}")

    rows = []
    pred_rows = []
    W = []
    Q_all = []
    run_names = []
    segment_labels = []
    fault_ids = []
    fault_names = []

    rng = np.random.default_rng(RANDOM_SEED)
    w0 = None

    for seq_i, run_dir in enumerate(train_sequence):
        df = load_measured_run(run_dir)
        df_norm = apply_normalization(df, norm_df)
        X_nobias, _, k0_run = build_delay_matrix_from_df(df_norm)
        X_base = transform_design(X_nobias, pca_mean, pca_components)
        J = expand_honu_columns(X_base, R)
        y_target = df_norm[OUTPUT_COLUMN].to_numpy(dtype=float)[k0_run:]

        # Sequential learning: start each segment from previous final weights.
        if w0 is None:
            w_init_before = None
        else:
            w_init_before = w0.copy()

        # Use a small local trainer variant so initial weights can be passed.
        nw = J.shape[1]
        if w_init_before is None:
            if WEIGHT_INIT_STD > 0:
                w = rng.normal(0.0, WEIGHT_INIT_STD, size=nw)
            else:
                w = np.zeros(nw)
        else:
            w = w_init_before.copy()

        L = np.eye(nw)
        Q = np.zeros(EPOCHS, dtype=float)
        Wall = np.zeros((EPOCHS + 1, nw), dtype=float)
        dWall = np.zeros((EPOCHS + 1, nw), dtype=float)
        Wall[0, :] = w
        A = J.T @ J + (1.0 / MU) * L + RIDGE_LAMBDA * L
        A_inv_JT = np.linalg.pinv(A) @ J.T
        for epoch in range(EPOCHS):
            yn_tmp = J @ w
            e_tmp = y_target - yn_tmp
            dw = A_inv_JT @ e_tmp
            w = w + dw
            Wall[epoch + 1, :] = w
            dWall[epoch + 1, :] = dw
            e2 = y_target - J @ w
            Q[epoch] = float(np.sum(e2 ** 2))
        yn = J @ w
        e = y_target - yn
        w0 = w.copy()

        meta = read_fault_metadata(run_dir)
        run_name = run_dir.name
        segment_label = f"{seq_i:03d}_{run_name}"

        run_names.append(run_name)
        segment_labels.append(segment_label)
        fault_ids.append(meta.get("fault_id", np.nan))
        fault_names.append(meta.get("fault_name", "unknown"))
        W.append(w)
        Q_all.append(Q)

        dW_from_prev = np.zeros_like(w) if w_init_before is None else w - w_init_before
        row = {
            "seq_index": seq_i,
            "segment_label": segment_label,
            "run_name": run_name,
            "fault_id": meta.get("fault_id", np.nan),
            "fault_name": meta.get("fault_name", "unknown"),
            "fault_magnitude": meta.get("fault_magnitude", np.nan),
            "N_samples": len(df),
            "N_training_samples": len(y_target),
            "Q_final": float(Q[-1]) if len(Q) else np.nan,
            "RMSE_norm": float(np.sqrt(np.mean(e ** 2))),
            "MAE_norm": float(np.mean(np.abs(e))),
            "dW_norm_from_previous_segment": float(np.linalg.norm(dW_from_prev)),
        }
        for i, wi in enumerate(w):
            row[f"w_{i}"] = wi
        for i, dwi in enumerate(dW_from_prev):
            row[f"dw_prev_{i}"] = dwi
        rows.append(row)

        pred_df = pd.DataFrame({
            "t": df[TIME_COLUMN].to_numpy()[k0_run:],
            "y_norm": y_target,
            "yhat_norm": yn,
            "e_norm": e,
            "seq_index": seq_i,
            "segment_label": segment_label,
            "run_name": run_name,
            "fault_id": meta.get("fault_id", np.nan),
            "fault_name": meta.get("fault_name", "unknown"),
        })
        pred_path = OUTPUT_DIR / f"prediction_{sanitize_name(segment_label)}.csv"
        pred_df.to_csv(pred_path, index=False)

        pred_rows.append({
            "seq_index": seq_i,
            "segment_label": segment_label,
            "run_name": run_name,
            "prediction_csv": str(pred_path.name),
            "Q_final": row["Q_final"],
            "RMSE_norm": row["RMSE_norm"],
            "dW_norm_from_previous_segment": row["dW_norm_from_previous_segment"],
        })

        print(
            f"Segment {seq_i:03d} {run_name:24s} | "
            f"fault={str(meta.get('fault_name','unknown')):14s} | "
            f"RMSE_norm={row['RMSE_norm']:.6g} | "
            f"||dW_prev||={row['dW_norm_from_previous_segment']:.6g}"
        )

    W = np.asarray(W, dtype=float)
    Q_all = np.asarray(Q_all, dtype=float)

    weights_df = pd.DataFrame(rows)
    weights_df.to_csv(OUTPUT_DIR / "sequence_weights.csv", index=False)
    weights_df.to_csv(OUTPUT_DIR / "weights_by_run.csv", index=False)  # compatibility alias
    pd.DataFrame(pred_rows).to_csv(OUTPUT_DIR / "sequence_predictions_summary.csv", index=False)

    # Sequential deltas and Learning Entropy: consecutive segment changes.
    dW_seq = np.zeros_like(W)
    dW_seq[1:, :] = np.diff(W, axis=0)
    sigma_w = np.std(dW_seq[1:, :], axis=0) + 1e-12 if W.shape[0] > 1 else np.ones(W.shape[1])
    LE_seq = np.sum(np.abs(dW_seq) / sigma_w[None, :], axis=1)
    LE_seq_norm = (LE_seq - LE_seq.min()) / (LE_seq.max() - LE_seq.min() + 1e-12)

    le_df = pd.DataFrame({
        "seq_index": np.arange(len(segment_labels)),
        "segment_label": segment_labels,
        "run_name": run_names,
        "fault_id": fault_ids,
        "fault_name": fault_names,
        "LE_from_previous_segment": LE_seq,
        "LE_from_previous_segment_norm": LE_seq_norm,
    })
    le_df.to_csv(OUTPUT_DIR / "learning_entropy_sequence.csv", index=False)

    n_base = transform_design(Xh_nobias[:1], pca_mean, pca_components).shape[1]
    S = np.zeros((W.shape[0], len(x_labels)), dtype=float)
    for i in range(W.shape[0]):
        S[i, :] = reconstruct_saliency_from_delta(dW_seq[i, :], pca_components, x_labels, n_base)
    S_norm = S / (S.max() + 1e-12)

    saliency_df = pd.DataFrame(S_norm, columns=x_labels)
    saliency_df.insert(0, "fault_name", fault_names)
    saliency_df.insert(0, "fault_id", fault_ids)
    saliency_df.insert(0, "run_name", run_names)
    saliency_df.insert(0, "segment_label", segment_labels)
    saliency_df.insert(0, "seq_index", np.arange(len(segment_labels)))
    saliency_df.to_csv(OUTPUT_DIR / "saliency_sequence.csv", index=False)

    # PCA of sequential final weights.
    Wc = W - W.mean(axis=0, keepdims=True)
    Uw, Sw, Vtw = np.linalg.svd(Wc, full_matrices=False)
    if W.shape[0] >= 2:
        C2 = Wc @ Vtw[:2, :].T
    else:
        C2 = np.zeros((W.shape[0], 2))

    weight_pca_df = pd.DataFrame({
        "seq_index": np.arange(len(segment_labels)),
        "segment_label": segment_labels,
        "run_name": run_names,
        "fault_id": fault_ids,
        "fault_name": fault_names,
        "c1": C2[:, 0],
        "c2": C2[:, 1],
        "LE_from_previous_segment_norm": LE_seq_norm,
    })
    weight_pca_df.to_csv(OUTPUT_DIR / "weight_pca_2d_sequence.csv", index=False)
    weight_pca_df.to_csv(OUTPUT_DIR / "weight_pca_2d.csv", index=False)  # compatibility alias

    np.savez(
        OUTPUT_DIR / "lnu_sequence_dataset.npz",
        W=W,
        dW_seq=dW_seq,
        LE_seq=LE_seq,
        LE_seq_norm=LE_seq_norm,
        S=S,
        S_norm=S_norm,
        C2=C2,
        Q=Q_all,
        segment_labels=np.array(segment_labels, dtype=object),
        run_names=np.array(run_names, dtype=object),
        fault_ids=np.array(fault_ids, dtype=object),
        fault_names=np.array(fault_names, dtype=object),
        x_labels=np.array(x_labels, dtype=object),
    )

    config = {
        "DATA_ROOT": str(DATA_ROOT),
        "OUTPUT_DIR": str(OUTPUT_DIR),
        "SEQUENTIAL_PROTOCOL": SEQUENTIAL_PROTOCOL,
        "ADD_FINAL_HEALTHY_SEGMENT": ADD_FINAL_HEALTHY_SEGMENT,
        "TIME_COLUMN": TIME_COLUMN,
        "DESIRED_COLUMN": DESIRED_COLUMN,
        "CONTROL_COLUMN": CONTROL_COLUMN,
        "OUTPUT_COLUMN": OUTPUT_COLUMN,
        "EXTRA_INPUT_COLUMNS": EXTRA_INPUT_COLUMNS,
        "ND": ND,
        "NU": NU,
        "NY": NY,
        "L_U": L_U,
        "R": R,
        "USE_PCA": USE_PCA,
        "PCA_VAR_THRESHOLD": PCA_VAR_THRESHOLD,
        "PCA_COMPONENTS_USED": int(n_comp),
        "EPOCHS": EPOCHS,
        "MU": MU,
        "healthy_reference_run": healthy_run.name,
        "training_sequence": segment_labels,
        "x_labels": x_labels,
    }
    with open(OUTPUT_DIR / "analysis_config.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(config, indent=2, ensure_ascii=False))

    if PLOT_RESULTS:
        # Sequential weight PCA trajectory.
        plt.figure(figsize=(9, 7))
        unique_faults = list(dict.fromkeys([str(x) for x in fault_names]))
        for fname in unique_faults:
            idx = [i for i, v in enumerate(fault_names) if str(v) == fname]
            plt.scatter(C2[idx, 0], C2[idx, 1], s=75, label=fname, edgecolors="k", linewidths=0.5)
        plt.plot(C2[:, 0], C2[:, 1], "k--", linewidth=0.8, alpha=0.7, label="sequence path")
        for i, lab in enumerate(segment_labels):
            plt.text(C2[i, 0], C2[i, 1], str(i), fontsize=8)
        plt.grid(True)
        plt.xlabel("weight PC1")
        plt.ylabel("weight PC2")
        plt.title("Sequential 2-D PCA of LNU weights: healthy -> fault -> healthy -> ...")
        plt.legend(fontsize=8)
        plt.tight_layout()
        if SAVE_PLOTS:
            plt.savefig(plot_dir / "weight_pca_2d_sequence.png", dpi=PLOT_DPI)
        if SHOW_PLOTS:
            plt.show()
        plt.close()

        # LE sequence.
        plt.figure(figsize=(11, 4))
        x = np.arange(len(segment_labels))
        plt.plot(x, LE_seq_norm, "o-", linewidth=1.5)
        plt.xticks(x, [f"{i}:{name}" for i, name in enumerate(run_names)], rotation=45, ha="right", fontsize=8)
        plt.ylabel("LE from previous segment, normalized")
        plt.title("Learning Entropy proxy along sequential training")
        plt.grid(True)
        plt.tight_layout()
        if SAVE_PLOTS:
            plt.savefig(plot_dir / "learning_entropy_sequence.png", dpi=PLOT_DPI)
        if SHOW_PLOTS:
            plt.show()
        plt.close()

        # Saliency sequence.
        plt.figure(figsize=(11, max(4, 0.45 * len(x_labels))))
        plt.imshow(S_norm.T, aspect="auto", origin="lower")
        plt.colorbar(label="normalized saliency")
        plt.xticks(np.arange(len(segment_labels)), [f"{i}:{name}" for i, name in enumerate(run_names)], rotation=45, ha="right", fontsize=8)
        plt.yticks(np.arange(len(x_labels)), x_labels, fontsize=8)
        plt.title("Embedding saliency from previous segment change")
        plt.xlabel("training segment")
        plt.ylabel("embedding coordinate")
        plt.tight_layout()
        if SAVE_PLOTS:
            plt.savefig(plot_dir / "saliency_sequence.png", dpi=PLOT_DPI)
        if SHOW_PLOTS:
            plt.show()
        plt.close()

        # Weight heatmap.
        Wz = (W - W.mean(axis=0, keepdims=True)) / (W.std(axis=0, keepdims=True) + 1e-12)
        plt.figure(figsize=(11, max(4, 0.35 * W.shape[1])))
        plt.imshow(Wz.T, aspect="auto", origin="lower", vmin=-3, vmax=3)
        plt.colorbar(label="z-scored weight")
        plt.xticks(np.arange(len(segment_labels)), [f"{i}:{name}" for i, name in enumerate(run_names)], rotation=45, ha="right", fontsize=8)
        plt.ylabel("weight index")
        plt.title("Final LNU weights along sequential protocol")
        plt.tight_layout()
        if SAVE_PLOTS:
            plt.savefig(plot_dir / "weights_heatmap_sequence.png", dpi=PLOT_DPI)
        if SHOW_PLOTS:
            plt.show()
        plt.close()

    print("\nDone.")
    print(f"Results written to: {OUTPUT_DIR.resolve()}")
    print("Main files: sequence_weights.csv, weight_pca_2d_sequence.csv, learning_entropy_sequence.csv, saliency_sequence.csv")


if __name__ == "__main__":
    main()
