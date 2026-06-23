Pipeline Overview
=================

Complete processing pipeline:

1. Load XLS/XLSX/CSV files.
2. Use first XLS/XLSX/CSV file as reference run.
3. Compute normalization statistics from reference file.
4. Create PCA embedding space.
5. Train LNU for each run.
6. Transfer trained weights to next run.
7. Store final LNU weights.
8. Compute Learning Entropy and saliency.
9. Train MLP from LNU weights.
10. Visualize diagnostics.

Pipeline principle
------------------

The package uses continual learning:

.. math::

   W_{k+1}^{(0)} \leftarrow W_k^{(\mathrm{final})}

where:

- :math:`W_k^{(\mathrm{final})}` are trained weights from run :math:`k`,
- :math:`W_{k+1}^{(0)}` are initial weights for the next run.

This creates smooth evolution in the weight space.
