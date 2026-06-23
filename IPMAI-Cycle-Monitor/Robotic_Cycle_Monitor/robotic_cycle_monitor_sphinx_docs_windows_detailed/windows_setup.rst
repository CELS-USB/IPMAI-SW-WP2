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

1. Raw CSV viewer:

::

   python run_1_robotic_raw_csv_viewer.py

2. LNU continual monitoring:

::

   python run_2_robotic_lnu_cycle_monitor.py

3. MLP training:

::

   python run_3_train_mlp_from_robotic_lnu_weights.py

4. Results viewer:

::

   python run_4_robotic_results_viewer.py

Sphinx build
------------

::

   sphinx-build -b html . _build/html
