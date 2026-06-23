# IPMAI Variable Comparison Viewer

Standalone PySide6 viewer for already generated IPMAI run folders.

It does **not** run the simulator and does **not** perform LNU/PCA/LE processing.
It only reads existing folders with `measured.csv` files and compares one selected variable across behavior types.

Expected structure:

```text
simulated_runs/
  sim_healthy_001/
    measured.csv
    annotated.csv      # optional, used for behavior name if present
    events.csv         # optional, used for behavior name if present
    config.txt         # optional
  sim_ripple_001/
    measured.csv
  ...
```

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python run_ipmai_variable_comparison_viewer.py
```

## Behavior

- Select a runs root directory.
- Select one common numeric variable from `measured.csv`.
- Each subplot shows one behavior group.
- All runs/trials belonging to the same behavior are overlaid in the corresponding subplot.
- Both x and y axes are synchronized, because all subplots show the same variable.
- Double-click any plot or press **Reset zoom** to restore the full common range.
- Line width is adjustable in the GUI.

Behavior grouping is inferred from `annotated.csv` or `events.csv` columns such as `fault_name`, `behavior`, `regime`, or `fault_id`. If these are unavailable, the folder name is used.
