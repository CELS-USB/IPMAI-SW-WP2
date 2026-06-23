Harmonic Drive Data Structure
=============================

Input data
----------

The package processes XLS/XLSX/CSV files from ``data_harmonic_drive_cycle_monitor``.

Typical filename:

::

   0200g_090d_symFE_tv1_x005.csv

Possible information encoded in names:

- load in grams,
- angle,
- operating condition.

Reference run
-------------

The first file is automatically selected as the reference run.

Reference run is used for:

- normalization,
- PCA basis,
- weight comparison,
- Learning Entropy reference,
- saliency comparison.

Signal processing
-----------------

Signals are loaded from XLS/XLSX/CSV columns and embedded into delayed vectors.

Typical embedding:

.. math::

   \mathbf{x}(k)=\begin{bmatrix}u(k)&u(k-1)&\cdots&y(k-1)&y(k-2)&\cdots\end{bmatrix}^T

The embedding labels are stored in:

::

   lnu_harmonic_drive_dataset.npz
