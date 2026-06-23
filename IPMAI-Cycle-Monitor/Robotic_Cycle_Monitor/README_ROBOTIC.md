# IPMAI Robotic Cycle Monitor

This package contains only scripts needed for the robotic real-CSV pipeline.
It does not include data files.

Expected layout:

```text
project_folder/
    data_robotic_cycle_monitor/
        *.csv
    run_2_robotic_lnu_cycle_monitor.py
    run_3_train_mlp_from_robotic_lnu_weights.py
    run_robotic_raw_csv_viewer.py
    run_robotic_results_viewer.py
    requirements.txt
```

## 1. Install requirements

```bash
pip install -r requirements.txt
```

## 2. Run LNU / LE / saliency cycle monitor

```bash
python run_2_robotic_lnu_cycle_monitor.py
```

The first sorted CSV file is used as the reference file for normalization and PCA. This is only a computational reference because the true healthy file is unknown.

Main outputs are written to:

```text
robotic_lnu_cycle_monitor_results/
```

Important files:

- `weights_by_file.csv` — final LNU weights for each CSV/run.
- `learning_entropy_continual.csv` — LE between consecutive runs.
- `learning_entropy_vs_reference.csv` — LE relative to the reference file.
- `saliency_continual.csv` — saliency between consecutive runs.
- `saliency_vs_reference.csv` — saliency relative to reference.
- `weight_pca_2d.csv` — 2-D PCA projection of final LNU weights.
- `prediction_*.csv` — measured/predicted output per run.

## 3. Train MLP from saved LNU weights

```bash
python run_3_train_mlp_from_robotic_lnu_weights.py
```

This is the Robotic version of the MLP stage from the PID-simulation package.
Because healthy/faulty labels are unknown, the default target label is derived from file metadata, currently `load_angle`.
You can change this at the top of the script:

```python
LABEL_MODE = "file_name"      # one class per CSV file
LABEL_MODE = "load_angle"     # one class per load-angle combination
LABEL_MODE = "load"           # one class per load
LABEL_MODE = "angle"          # one class per angle
LABEL_MODE = "custom_column"  # use your own label column
```

MLP outputs are written to:

```text
robotic_mlp_from_lnu_weights_results/
```

Important files:

- `mlp_probabilities_from_robotic_lnu_weights.csv`
- `confusion_matrix.csv`
- `classification_report.txt`
- `training_summary.json`
- `mlp_robotic_lnu_weight_classifier.pkl`

## 4. Viewers

```bash
python run_robotic_raw_csv_viewer.py
python run_robotic_results_viewer.py
```

The first viewer shows raw robotic CSV signals. The second viewer shows outputs from `robotic_lnu_cycle_monitor_results`.

## Main launcher

Run the complete package from one PySide6 window:

```bash
python run_0_robotic_launcher.py
```

The launcher can open the raw CSV viewer, run the LNU cycle monitor, train the MLP stage, open the results viewer, and open/build the Sphinx documentation.


### Launcher update

The launcher includes:
- a larger status panel on the left side,
- an indeterminate progress indicator while processing scripts are running,
- disabled processing buttons during active computation to avoid accidental parallel runs.
