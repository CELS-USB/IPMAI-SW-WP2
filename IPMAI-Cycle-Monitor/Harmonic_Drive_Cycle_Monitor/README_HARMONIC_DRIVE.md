# IPMAI Harmonic Drive Cycle Monitor

This package is the Harmonic Drive version of the Robotic real-data cycle-monitor pipeline.
The script structure is intentionally kept equivalent to the Robotic Cycle Monitor module; the main difference is the data loader and the variable setup for Harmonic Drive CSV/XLS/XLSX files.

Expected layout:

```text
project_folder/
    data_harmonic_drive_cycle_monitor/
        *.csv / *.xls / *.xlsx
    run_0_harmonic_drive_launcher.py
    run_1_harmonic_drive_raw_xls_viewer.py
    run_2_harmonic_drive_lnu_cycle_monitor.py
    run_3_train_mlp_from_harmonic_drive_lnu_weights.py
    run_4_harmonic_drive_results_viewer.py
    requirements.txt
```

## 1. Install requirements

```bash
pip install -r requirements.txt
```

## 2. Select the predicted variable

At the top of `run_2_harmonic_drive_lnu_cycle_monitor.py`, set for example:

```python
TARGET_COLUMN = "a"
```

The first column `t` is used as the time column by default:

```python
TIME_COLUMN = "t"
```

The current default Harmonic Drive variables are:

```python
SELECTED_VARIABLES = ["a", "as", "ab", "E", "W"]
```

## 3. Run LNU / LE / saliency cycle monitor

```bash
python run_2_harmonic_drive_lnu_cycle_monitor.py
```

Main outputs are written to:

```text
harmonic_drive_lnu_cycle_monitor_results/
```

Important files:

- `weights_by_file.csv` — final LNU weights for each harmonic-drive file/run.
- `learning_entropy_continual.csv` — LE between consecutive files/runs.
- `learning_entropy_vs_reference.csv` — LE relative to the reference file.
- `saliency_continual.csv` — saliency between consecutive files/runs.
- `saliency_vs_reference.csv` — saliency relative to reference.
- `weight_pca_2d.csv` — 2-D PCA projection of final LNU weights.
- `prediction_*.csv` — measured/predicted output per file/run.

## 4. Train MLP from saved LNU weights

```bash
python run_3_train_mlp_from_harmonic_drive_lnu_weights.py
```

MLP outputs are written to:

```text
harmonic_drive_mlp_from_lnu_weights_results/
```

## 5. Viewers

```bash
python run_1_harmonic_drive_raw_xls_viewer.py
python run_4_harmonic_drive_results_viewer.py
```

The first viewer shows raw Harmonic Drive source signals. The second viewer shows outputs from `harmonic_drive_lnu_cycle_monitor_results` and `harmonic_drive_mlp_from_lnu_weights_results`.

## Main launcher

Run the complete package from one PySide6 window:

```bash
python run_0_harmonic_drive_launcher.py
```
