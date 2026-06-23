"""
IPMAI Harmonic Drive LNU cycle monitor for real data files.

Purpose
-------
This script applies the IPMAI cycle-monitor LNU pipeline to real harmonic-drive data files.
It does not simulate data and it does not insert artificial perturbations.
Each data file is treated as one run/session, and file names are used as labels.

Continual-monitoring convention
------------------------------
The first data file in sorted order is used as the reference/baseline file.
This does NOT mean that the file is known to be healthy; it is only the chosen
reference because no health label is available. Normalization statistics and
the PCA delay-vector basis are fitted only on this reference file. The learned
final LNU weights from one file are then used as the initial weights for the
next file, so newly added files can be processed as a continuation of the same
adaptive monitoring chain.

Expected input layout
---------------------
    data_harmonic_drive_cycle_monitor/
        0000g_060d_symFE_tv1_x005.csv
        0200g_090d_symFE_tv1_x005.csv
        ...

The script writes results to OUTPUT_DIR:
    weights_by_file.csv
    predictions_summary.csv
    learning_entropy_continual.csv
    learning_entropy_vs_reference.csv
    saliency_continual.csv
    saliency_vs_reference.csv
    normalization_reference.csv
    pca_reference.npz
    lnu_harmonic_drive_dataset.npz
    plots/*.png

Windows usage
-------------
    python run_2_harmonic_drive_lnu_cycle_monitor.py

Authoring convention: all user setups are intentionally kept at the top.
"""

# ============================================================
# USER SETUP
# ============================================================

from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Input/output folders ----
DATA_ROOT = Path("data_harmonic_drive_cycle_monitor")
OUTPUT_DIR = Path("harmonic_drive_lnu_cycle_monitor_results")
FILE_PATTERNS = ["*.csv", "*.xls", "*.xlsx"]
CSV_SEPARATOR = None                    # None -> standard comma CSV; Excel is read by pandas.read_excel
CSV_ENGINE = "python"

# ---- File/run selection ----
SORT_FILES_BY = "name"               # currently: "name"
REFERENCE_FILE_KEYWORD = None         # e.g. "0000g" if you want force reference; None -> first sorted file
REFERENCE_MODE_FOR_LE = "reference"   # "reference" or "median_weight"
REFERENCE_POLICY_NOTE = (
    "The first sorted data file is used as reference/baseline only because the healthy file is unknown. "
    "This is not a claim that the reference file is healthy."
)
CONTINUAL_WEIGHT_TRANSFER = True       # final weights from one file initialize the next file

# ---- Harmonic Drive variables and sample range ----
TIME_COLUMN = "t"
TARGET_COLUMN = "a"                  # predicted variable; change here, e.g. "a", "ab", "E", "W"
DESIRED_COLUMN = TARGET_COLUMN         # d(k), autoregressive reference channel by default
CONTROL_COLUMN = "W"                  # u(k), auxiliary input; if missing or equal to target, target is reused
OUTPUT_COLUMN = TARGET_COLUMN          # y(k), predicted output
SELECTED_VARIABLES = ["a", "as", "ab", "E", "W"]
EXTRA_INPUT_COLUMNS = ["as", "ab", "E"]  # optional additional normalized columns in delay vector
N_EXTRA_DELAYS = 1

NSTART = 0                         # skip initial stationary/zero part as in Harmonic Drive setup
NSTOP =None #100000                        # None -> until end
DN = 1                                # every DN-th sample
MIN_SAMPLES_AFTER_CROP = 100

# ---- Normalization ----
NORMALIZATION_MODE = "reference_file" # fixed reference stats; do not recompute from later/incoming files
ZSCORE_SCALE = 3.0                    # Harmonic Drive convention: (x - mean)/(3*std)
STD_EPS = 1e-12

# ---- LNU/HONU setup copied from Harmonic Drive script ----
R = 1                                 # 1=LNU, 2=QNU, 3=CNU
ND = 2                                # d[k-ND], ..., d[k-1]
NU = 9                                # u[k-(L_U+NU-1)], ..., u[k-L_U]
NY = 2                                # y[k-NY], ..., y[k-1]
L_U = 3                               # control delay

# ---- PCA over input delay vector x without bias ----
USE_PCA = True
PCA_VAR_THRESHOLD = 0.995
NPCA_MAX_PER_FILE = 20000
MIN_PCA_COMPONENTS = 1
MAX_PCA_COMPONENTS = None             # set integer if needed

# ---- Batch LNU learning on each complete file ----
# Initial weights are transferred between files when CONTINUAL_WEIGHT_TRANSFER=True.
MU = 1000.0
EPOCHS = 20 * R**3
WEIGHT_INIT_STD = 0.0                 # 0 -> deterministic zero initialization
RANDOM_SEED = 123
RIDGE_LAMBDA = 0.0

# ---- Diagnostics and plotting ----
PLOT_RESULTS = True
SAVE_PLOTS = True
SHOW_PLOTS = False
PLOT_DPI = 140
CSV_FLOAT_FORMAT = "%.10g"

# ============================================================
# IMPLEMENTATION
# ============================================================


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", name).strip("_")


def parse_filename_metadata(path: Path) -> dict:
    name = path.stem
    meta = {"file_name": path.name, "run_name": name, "load_g": np.nan, "angle_deg": np.nan}
    m_load = re.search(r"(\d+)g", name)
    m_angle = re.search(r"_(\d+)d_", name)
    if m_load:
        meta["load_g"] = int(m_load.group(1))
    if m_angle:
        meta["angle_deg"] = int(m_angle.group(1))
    return meta


def find_data_files() -> list[Path]:
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"DATA_ROOT does not exist: {DATA_ROOT.resolve()}")
    files = []
    for pat in FILE_PATTERNS:
        files.extend(DATA_ROOT.glob(pat))
    files = sorted(set(files), key=lambda p: p.name.lower())
    if not files:
        raise FileNotFoundError(f"No CSV/XLS/XLSX files found under {DATA_ROOT.resolve()}")
    return files


def choose_reference_file(files: list[Path]) -> Path:
    if REFERENCE_FILE_KEYWORD:
        candidates = [p for p in files if REFERENCE_FILE_KEYWORD.lower() in p.name.lower()]
        if not candidates:
            raise ValueError(f"No reference file contains keyword: {REFERENCE_FILE_KEYWORD}")
        return sorted(candidates, key=lambda p: p.name)[0]
    return files[0]


def read_harmonic_drive_data(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in [".xls", ".xlsx"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, sep=CSV_SEPARATOR, engine=CSV_ENGINE)

    # Make the script tolerant to accidental whitespace in headers.
    df.columns = [str(c).strip() for c in df.columns]

    global CONTROL_COLUMN, EXTRA_INPUT_COLUMNS, SELECTED_VARIABLES
    if CONTROL_COLUMN not in df.columns:
        CONTROL_COLUMN = TARGET_COLUMN
    SELECTED_VARIABLES = [c for c in SELECTED_VARIABLES if c in df.columns]
    if TARGET_COLUMN not in SELECTED_VARIABLES:
        SELECTED_VARIABLES.insert(0, TARGET_COLUMN)
    EXTRA_INPUT_COLUMNS = [c for c in EXTRA_INPUT_COLUMNS if c in df.columns and c != TARGET_COLUMN]

    missing = [c for c in [TIME_COLUMN, TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {missing}. Available columns: {list(df.columns)}")

    stop = None if NSTOP is None else int(NSTOP)
    df = df.iloc[int(NSTART):stop, :].copy()
    if DN > 1:
        df = df.iloc[::int(DN), :].copy()
    df = df.reset_index(drop=True)
    if len(df) < MIN_SAMPLES_AFTER_CROP:
        raise ValueError(f"Too few samples in {path.name} after crop/downsampling: {len(df)}")

    # local time starting at zero, if possible
    df[TIME_COLUMN] = df[TIME_COLUMN].to_numpy(dtype=float)
    df[TIME_COLUMN] = df[TIME_COLUMN] - df[TIME_COLUMN].iloc[0]
    return df


def signal_columns() -> list[str]:
    cols = [DESIRED_COLUMN, CONTROL_COLUMN, OUTPUT_COLUMN] + list(EXTRA_INPUT_COLUMNS)
    for c in SELECTED_VARIABLES:
        if c not in cols:
            cols.append(c)
    seen = []
    for c in cols:
        if c not in seen:
            seen.append(c)
    return seen


def compute_normalization(files: list[Path], reference_file: Path) -> pd.DataFrame:
    cols = signal_columns()
    if NORMALIZATION_MODE == "reference_file":
        dfs = [read_harmonic_drive_data(reference_file)]
    elif NORMALIZATION_MODE == "all_files":
        dfs = [read_harmonic_drive_data(p) for p in files]
    else:
        raise ValueError("NORMALIZATION_MODE must be 'all_files' or 'reference_file'.")

    rows = []
    for c in cols:
        vals = np.concatenate([df[c].to_numpy(dtype=float) for df in dfs])
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
        if c in out.columns:
            out[c] = (out[c].to_numpy(dtype=float) - float(row["mean"])) / (ZSCORE_SCALE * float(row["std"]))
    return out


def start_index() -> int:
    starts = [ND, NY, L_U + NU - 1]
    if EXTRA_INPUT_COLUMNS:
        starts.append(N_EXTRA_DELAYS - 1)
    return max(starts)


def build_delay_matrix(df_norm: pd.DataFrame):
    d = df_norm[DESIRED_COLUMN].to_numpy(dtype=float)
    u = df_norm[CONTROL_COLUMN].to_numpy(dtype=float)
    y = df_norm[OUTPUT_COLUMN].to_numpy(dtype=float)
    extras = {c: df_norm[c].to_numpy(dtype=float) for c in EXTRA_INPUT_COLUMNS}

    n_extra_total = len(EXTRA_INPUT_COLUMNS) * N_EXTRA_DELAYS
    nx_nobias = ND + NU + NY + n_extra_total
    k0 = start_index()
    N = len(y)
    if N <= k0:
        raise ValueError(f"Data too short: N={N}, required start index={k0}")

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
        for c in EXTRA_INPUT_COLUMNS:
            arr = extras[c]
            for lag_i in range(N_EXTRA_DELAYS):
                delay = N_EXTRA_DELAYS - lag_i - 1
                X[row_i, pos] = arr[k - delay]
                pos += 1
    return X, labels, k0


def fit_pca(reference_file: Path, norm_df: pd.DataFrame):
    # PCA is intentionally fitted only on the reference file.
    # Later files are projected into this fixed basis so that weights, LE,
    # and saliency remain comparable across runs.
    df = apply_normalization(read_harmonic_drive_data(reference_file), norm_df)
    Xp, labels_ref, _ = build_delay_matrix(df)
    Xp = Xp[:min(NPCA_MAX_PER_FILE, Xp.shape[0]), :]
    pca_mean = Xp.mean(axis=0)
    Xc = Xp - pca_mean
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    eigvals = (S ** 2) / max(1, Xc.shape[0] - 1)
    total = float(np.sum(eigvals))
    cumvar = np.ones_like(eigvals) if total <= 0 else np.cumsum(eigvals) / total
    n_comp = int(np.searchsorted(cumvar, PCA_VAR_THRESHOLD) + 1)
    n_comp = max(MIN_PCA_COMPONENTS, n_comp)
    if MAX_PCA_COMPONENTS is not None:
        n_comp = min(int(MAX_PCA_COMPONENTS), n_comp)
    n_comp = min(n_comp, Vt.shape[0])
    return pca_mean, Vt[:n_comp, :].T, eigvals, cumvar, n_comp, labels_ref


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
    struct = polynomial_structure(X_base.shape[1], r)
    Xh = np.zeros((X_base.shape[0], len(struct)), dtype=float)
    for q, idxs in enumerate(struct):
        prod = np.ones(X_base.shape[0])
        for idx in idxs:
            prod *= X_base[:, idx]
        Xh[:, q] = prod
    return Xh


def train_batch(J: np.ndarray, y_target: np.ndarray, rng: np.random.Generator, w0: np.ndarray | None = None):
    nw = J.shape[1]
    if w0 is not None:
        if len(w0) != nw:
            raise ValueError(f"Initial weight length mismatch: len(w0)={len(w0)}, expected {nw}")
        w = np.asarray(w0, dtype=float).copy()
    else:
        w = rng.normal(0.0, WEIGHT_INIT_STD, nw) if WEIGHT_INIT_STD > 0 else np.zeros(nw)
    L = np.eye(nw)
    A = J.T @ J + (1.0 / MU) * L + RIDGE_LAMBDA * L
    A_inv_JT = np.linalg.pinv(A) @ J.T
    Q = np.zeros(EPOCHS, dtype=float)
    Wall = np.zeros((EPOCHS + 1, nw), dtype=float)
    dWall = np.zeros((EPOCHS + 1, nw), dtype=float)
    Wall[0, :] = w
    for epoch in range(EPOCHS):
        e = y_target - J @ w
        dw = A_inv_JT @ e
        w = w + dw
        Wall[epoch + 1, :] = w
        dWall[epoch + 1, :] = dw
        Q[epoch] = float(np.sum((y_target - J @ w) ** 2))
    yn = J @ w
    e = y_target - yn
    return w, Q, Wall, dWall, yn, e


def reconstruct_saliency(delta_w: np.ndarray, pca_components: np.ndarray, x_labels: list[str], n_base: int):
    n_embed = len(x_labels)
    P = pca_components if USE_PCA else np.eye(n_embed)
    contrib = np.zeros(n_embed, dtype=float)
    struct = polynomial_structure(n_base, R)
    for q, idxs in enumerate(struct):
        z_indices = [idx - 1 for idx in idxs if idx > 0]
        if not z_indices:
            continue
        factors = np.ones(n_embed, dtype=float)
        for z_idx in z_indices:
            if z_idx >= P.shape[1]:
                factors[:] = 0.0
                break
            factors *= np.abs(P[:, z_idx])
        s = float(factors.sum())
        if s > 0:
            contrib += abs(delta_w[q]) * factors / s
    return contrib


def main():
    ensure_dir(OUTPUT_DIR)
    plot_dir = OUTPUT_DIR / "plots"
    ensure_dir(plot_dir)

    files = find_data_files()
    reference_file = choose_reference_file(files)
    print(f"Found {len(files)} harmonic-drive data files.")
    print(f"Reference file: {reference_file.name}")

    print("Reference policy:", REFERENCE_POLICY_NOTE)

    norm_df = compute_normalization(files, reference_file)
    norm_df.to_csv(OUTPUT_DIR / "normalization_reference.csv", index=False, float_format=CSV_FLOAT_FORMAT)

    pca_mean, pca_components, pca_eigvals, pca_cumvar, n_comp, x_labels = fit_pca(reference_file, norm_df)
    np.savez(
        OUTPUT_DIR / "pca_reference.npz",
        pca_mean=pca_mean,
        pca_components=pca_components,
        pca_eigvals=pca_eigvals,
        pca_cumvar=pca_cumvar,
        x_labels=np.array(x_labels, dtype=object),
        n_components=n_comp,
        use_pca=USE_PCA,
    )
    print(f"Delay vector dimension without bias: {len(x_labels)}")
    print(f"PCA components used: {n_comp}, captured variance: {pca_cumvar[n_comp-1] * 100:.3f}%")

    rng = np.random.default_rng(RANDOM_SEED)
    rows, pred_rows = [], []
    W, W_initial, Q_all = [], [], []
    run_names, file_names, load_g, angle_deg = [], [], [], []

    for p in files:
        meta = parse_filename_metadata(p)
        df = read_harmonic_drive_data(p)
        df_norm = apply_normalization(df, norm_df)
        X_nobias, _, k0 = build_delay_matrix(df_norm)
        X_base = transform_design(X_nobias, pca_mean, pca_components)
        J = expand_honu_columns(X_base, R)
        y_target = df_norm[OUTPUT_COLUMN].to_numpy(dtype=float)[k0:]

        w0_current = W[-1].copy() if (CONTINUAL_WEIGHT_TRANSFER and len(W) > 0) else None
        w, Q, Wall, dWall, yn, e = train_batch(J, y_target, rng, w0=w0_current)

        W_initial.append(Wall[0, :].copy())
        W.append(w)
        Q_all.append(Q)
        run_names.append(meta["run_name"])
        file_names.append(meta["file_name"])
        load_g.append(meta["load_g"])
        angle_deg.append(meta["angle_deg"])

        row = {
            **meta,
            "N_samples": len(df),
            "N_training_samples": len(y_target),
            "dt_median_sec": float(np.median(np.diff(df[TIME_COLUMN].to_numpy(dtype=float)))) if len(df) > 1 else np.nan,
            "Q_final": float(Q[-1]) if len(Q) else np.nan,
            "RMSE_norm": float(np.sqrt(np.mean(e ** 2))),
            "MAE_norm": float(np.mean(np.abs(e))),
            "is_reference_file": bool(p.name == reference_file.name),
            "reference_policy_note": REFERENCE_POLICY_NOTE if p.name == reference_file.name else "",
            "continual_weight_transfer": bool(CONTINUAL_WEIGHT_TRANSFER),
        }
        for i, wi in enumerate(w):
            row[f"w_{i}"] = wi
        rows.append(row)

        pred_df = pd.DataFrame({
            TIME_COLUMN: df[TIME_COLUMN].to_numpy(dtype=float)[k0:],
            "y_norm": y_target,
            "yhat_norm": yn,
            "e_norm": e,
            "run_name": meta["run_name"],
            "file_name": meta["file_name"],
            "load_g": meta["load_g"],
            "angle_deg": meta["angle_deg"],
        })
        pred_csv = OUTPUT_DIR / f"prediction_{sanitize_name(meta['run_name'])}.csv"
        pred_df.to_csv(pred_csv, index=False, float_format=CSV_FLOAT_FORMAT)
        pred_rows.append({"run_name": meta["run_name"], "file_name": meta["file_name"], "prediction_csv": pred_csv.name, "Q_final": row["Q_final"], "RMSE_norm": row["RMSE_norm"]})
        print(f"Trained {p.name:35s} | RMSE_norm={row['RMSE_norm']:.6g}")

    W = np.asarray(W, dtype=float)
    W_initial = np.asarray(W_initial, dtype=float)
    Q_all = np.asarray(Q_all, dtype=float)
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "weights_by_file.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    pd.DataFrame(pred_rows).to_csv(OUTPUT_DIR / "predictions_summary.csv", index=False, float_format=CSV_FLOAT_FORMAT)

    if REFERENCE_MODE_FOR_LE == "median_weight":
        w_ref = np.median(W, axis=0)
        reference_name = "median_weight"
    elif REFERENCE_MODE_FOR_LE == "reference":
        ref_idx = file_names.index(reference_file.name)
        w_ref = W[ref_idx, :]
        reference_name = reference_file.name
    else:
        raise ValueError("REFERENCE_MODE_FOR_LE must be 'reference' or 'median_weight'.")

    # LE 1: distance from the chosen reference file.
    dW = W - w_ref[None, :]
    sigma_w = np.std(dW, axis=0) + 1e-12
    LE = np.sum(np.abs(dW) / sigma_w[None, :], axis=1)
    LE_norm = (LE - LE.min()) / (LE.max() - LE.min() + 1e-12)

    le_df = pd.DataFrame({
        "run_name": run_names,
        "file_name": file_names,
        "load_g": load_g,
        "angle_deg": angle_deg,
        "reference_name": reference_name,
        "is_reference_file": [fn == reference_file.name for fn in file_names],
        "reference_policy_note": [REFERENCE_POLICY_NOTE if fn == reference_file.name else "" for fn in file_names],
        "LE_vs_reference": LE,
        "LE_vs_reference_norm": LE_norm,
    })
    le_df.to_csv(OUTPUT_DIR / "learning_entropy_vs_reference.csv", index=False, float_format=CSV_FLOAT_FORMAT)

    # LE 2: continual/incremental weight change between consecutive files.
    # This is the main cycle-monitor quantity when files are interpreted as an incoming sequence.
    dW_prev = np.zeros_like(W)
    if W.shape[0] > 1:
        dW_prev[1:, :] = W[1:, :] - W[:-1, :]
    sigma_prev = np.std(dW_prev[1:, :], axis=0) + 1e-12 if W.shape[0] > 1 else np.ones(W.shape[1])
    LE_prev = np.sum(np.abs(dW_prev) / sigma_prev[None, :], axis=1)
    if W.shape[0] > 1:
        LE_prev_norm = (LE_prev - LE_prev.min()) / (LE_prev.max() - LE_prev.min() + 1e-12)
    else:
        LE_prev_norm = np.zeros_like(LE_prev)

    le_prev_df = pd.DataFrame({
        "run_name": run_names,
        "file_name": file_names,
        "previous_file_name": ["" if i == 0 else file_names[i-1] for i in range(len(file_names))],
        "load_g": load_g,
        "angle_deg": angle_deg,
        "is_reference_file": [fn == reference_file.name for fn in file_names],
        "continual_weight_transfer": bool(CONTINUAL_WEIGHT_TRANSFER),
        "LE_vs_previous_file": LE_prev,
        "LE_vs_previous_file_norm": LE_prev_norm,
    })
    le_prev_df.to_csv(OUTPUT_DIR / "learning_entropy_continual.csv", index=False, float_format=CSV_FLOAT_FORMAT)

    n_base = 1 + (pca_components.shape[1] if USE_PCA else len(x_labels))
    S = np.zeros((W.shape[0], len(x_labels)), dtype=float)
    S_prev = np.zeros((W.shape[0], len(x_labels)), dtype=float)
    for i in range(W.shape[0]):
        S[i, :] = reconstruct_saliency(dW[i, :], pca_components, x_labels, n_base)
        S_prev[i, :] = reconstruct_saliency(dW_prev[i, :], pca_components, x_labels, n_base)
    S_norm = S / (S.max() + 1e-12)
    S_prev_norm = S_prev / (S_prev.max() + 1e-12)

    saliency_df = pd.DataFrame(S_norm, columns=x_labels)
    saliency_df.insert(0, "angle_deg", angle_deg)
    saliency_df.insert(0, "load_g", load_g)
    saliency_df.insert(0, "file_name", file_names)
    saliency_df.insert(0, "run_name", run_names)
    saliency_df.to_csv(OUTPUT_DIR / "saliency_vs_reference.csv", index=False, float_format=CSV_FLOAT_FORMAT)

    saliency_prev_df = pd.DataFrame(S_prev_norm, columns=x_labels)
    saliency_prev_df.insert(0, "angle_deg", angle_deg)
    saliency_prev_df.insert(0, "load_g", load_g)
    saliency_prev_df.insert(0, "previous_file_name", ["" if i == 0 else file_names[i-1] for i in range(len(file_names))])
    saliency_prev_df.insert(0, "file_name", file_names)
    saliency_prev_df.insert(0, "run_name", run_names)
    saliency_prev_df.to_csv(OUTPUT_DIR / "saliency_continual.csv", index=False, float_format=CSV_FLOAT_FORMAT)

    Wc = W - W.mean(axis=0, keepdims=True)
    _, _, Vtw = np.linalg.svd(Wc, full_matrices=False)
    C2 = Wc @ Vtw[:2, :].T if W.shape[0] >= 2 else np.zeros((W.shape[0], 2))
    weight_pca_df = pd.DataFrame({
        "run_name": run_names,
        "file_name": file_names,
        "load_g": load_g,
        "angle_deg": angle_deg,
        "c1": C2[:, 0],
        "c2": C2[:, 1],
        "LE_vs_reference_norm": LE_norm,
    })
    weight_pca_df.to_csv(OUTPUT_DIR / "weight_pca_2d.csv", index=False, float_format=CSV_FLOAT_FORMAT)

    np.savez(
        OUTPUT_DIR / "lnu_harmonic_drive_dataset.npz",
        W=W,
        W_initial=W_initial,
        dW_vs_reference=dW,
        dW_vs_previous_file=dW_prev,
        LE=LE,
        LE_norm=LE_norm,
        LE_prev=LE_prev,
        LE_prev_norm=LE_prev_norm,
        S=S,
        S_norm=S_norm,
        S_prev=S_prev,
        S_prev_norm=S_prev_norm,
        C2=C2,
        Q=Q_all,
        run_names=np.array(run_names, dtype=object),
        file_names=np.array(file_names, dtype=object),
        load_g=np.array(load_g, dtype=object),
        angle_deg=np.array(angle_deg, dtype=object),
        x_labels=np.array(x_labels, dtype=object),
    )

    config = {k: str(v) if isinstance(v, Path) else v for k, v in {
        "DATA_ROOT": DATA_ROOT,
        "OUTPUT_DIR": OUTPUT_DIR,
        "FILE_PATTERNS": FILE_PATTERNS,
        "REFERENCE_FILE": reference_file.name,
        "REFERENCE_MODE_FOR_LE": REFERENCE_MODE_FOR_LE,
        "REFERENCE_POLICY_NOTE": REFERENCE_POLICY_NOTE,
        "CONTINUAL_WEIGHT_TRANSFER": CONTINUAL_WEIGHT_TRANSFER,
        "NORMALIZATION_MODE": NORMALIZATION_MODE,
        "SELECTED_VARIABLES": SELECTED_VARIABLES,
        "DESIRED_COLUMN": DESIRED_COLUMN,
        "CONTROL_COLUMN": CONTROL_COLUMN,
        "OUTPUT_COLUMN": OUTPUT_COLUMN,
        "EXTRA_INPUT_COLUMNS": EXTRA_INPUT_COLUMNS,
        "NSTART": NSTART,
        "NSTOP": NSTOP,
        "DN": DN,
        "ND": ND,
        "NU": NU,
        "NY": NY,
        "L_U": L_U,
        "R": R,
        "USE_PCA": USE_PCA,
        "PCA_VAR_THRESHOLD": PCA_VAR_THRESHOLD,
        "PCA_COMPONENTS_USED": int(n_comp),
        "MU": MU,
        "EPOCHS": EPOCHS,
        "x_labels": x_labels,
    }.items()}
    with open(OUTPUT_DIR / "analysis_config.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(config, indent=2, ensure_ascii=False))

    if PLOT_RESULTS:
        # Weight PCA scatter.
        plt.figure(figsize=(9, 7))
        sc = plt.scatter(C2[:, 0], C2[:, 1], c=np.asarray(load_g, dtype=float), s=80, edgecolors="k", linewidths=0.5)
        for i, name in enumerate(run_names):
            plt.text(C2[i, 0], C2[i, 1], name, fontsize=7)
        plt.colorbar(sc, label="load [g]")
        plt.grid(True)
        plt.xlabel("weight PC1")
        plt.ylabel("weight PC2")
        plt.title("2-D PCA of final LNU weights across harmonic-drive files")
        plt.tight_layout()
        if SAVE_PLOTS:
            plt.savefig(plot_dir / "weight_pca_2d.png", dpi=PLOT_DPI)
        if SHOW_PLOTS:
            plt.show()
        plt.close()

        # LE bar plot.
        plt.figure(figsize=(11, 4.5))
        x = np.arange(len(run_names))
        plt.bar(x, LE_norm)
        plt.xticks(x, run_names, rotation=45, ha="right", fontsize=8)
        plt.ylabel("LE vs reference, normalized")
        plt.title(f"Learning Entropy proxy from weight distance to {reference_name}\n(reference is not assumed healthy)")
        plt.grid(True, axis="y")
        plt.tight_layout()
        if SAVE_PLOTS:
            plt.savefig(plot_dir / "learning_entropy_vs_reference.png", dpi=PLOT_DPI)
        if SHOW_PLOTS:
            plt.show()
        plt.close()

        # Continual LE bar plot.
        plt.figure(figsize=(11, 4.5))
        x = np.arange(len(run_names))
        plt.bar(x, LE_prev_norm)
        plt.xticks(x, run_names, rotation=45, ha="right", fontsize=8)
        plt.ylabel("LE vs previous file, normalized")
        plt.title("Continual Learning Entropy proxy from consecutive weight changes")
        plt.grid(True, axis="y")
        plt.tight_layout()
        if SAVE_PLOTS:
            plt.savefig(plot_dir / "learning_entropy_continual.png", dpi=PLOT_DPI)
        if SHOW_PLOTS:
            plt.show()
        plt.close()

        # Saliency map.
        plt.figure(figsize=(11, max(4, 0.45 * len(x_labels))))
        plt.imshow(S_norm.T, aspect="auto", origin="lower")
        plt.colorbar(label="normalized saliency")
        plt.xticks(np.arange(len(run_names)), run_names, rotation=45, ha="right", fontsize=8)
        plt.yticks(np.arange(len(x_labels)), x_labels, fontsize=8)
        plt.title("Embedding saliency vs reference")
        plt.xlabel("harmonic-drive file/run")
        plt.ylabel("embedding coordinate")
        plt.tight_layout()
        if SAVE_PLOTS:
            plt.savefig(plot_dir / "saliency_vs_reference.png", dpi=PLOT_DPI)
        if SHOW_PLOTS:
            plt.show()
        plt.close()

        # Weight heatmap.
        Wz = (W - W.mean(axis=0, keepdims=True)) / (W.std(axis=0, keepdims=True) + 1e-12)
        plt.figure(figsize=(11, max(4, 0.35 * W.shape[1])))
        plt.imshow(Wz.T, aspect="auto", origin="lower", vmin=-3, vmax=3)
        plt.colorbar(label="z-scored weight")
        plt.xticks(np.arange(len(run_names)), run_names, rotation=45, ha="right", fontsize=8)
        plt.ylabel("weight index")
        plt.title("Final LNU weights across harmonic-drive files")
        plt.tight_layout()
        if SAVE_PLOTS:
            plt.savefig(plot_dir / "weights_heatmap.png", dpi=PLOT_DPI)
        if SHOW_PLOTS:
            plt.show()
        plt.close()

    print("\nDone.")
    print(f"Results written to: {OUTPUT_DIR.resolve()}")
    print("Main files: weights_by_file.csv, weight_pca_2d.csv, learning_entropy_continual.csv, learning_entropy_vs_reference.csv, saliency_continual.csv, saliency_vs_reference.csv")


if __name__ == "__main__":
    main()
