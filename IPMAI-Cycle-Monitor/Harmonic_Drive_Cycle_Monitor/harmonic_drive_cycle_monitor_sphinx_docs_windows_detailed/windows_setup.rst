Windows Setup
=============

Requirements
------------

Recommended:

- Windows 11,
- Python 3.12,
- Anaconda.

Install packages
----------------

::

   pip install numpy pandas matplotlib PySide6 pyqtgraph scikit-learn scipy

Run order
---------

1. Raw XLS/XLSX/CSV viewer:

::

   python run_1_harmonic_drive_raw_csv_viewer.py

2. LNU continual monitoring:

::

   python run_2_harmonic_drive_lnu_cycle_monitor.py

3. MLP training:

::

   python run_3_train_mlp_from_harmonic_drive_lnu_weights.py

4. Results viewer:

::

   python run_4_harmonic_drive_results_viewer.py

Sphinx build
------------

::

   sphinx-build -b html . _build/html
