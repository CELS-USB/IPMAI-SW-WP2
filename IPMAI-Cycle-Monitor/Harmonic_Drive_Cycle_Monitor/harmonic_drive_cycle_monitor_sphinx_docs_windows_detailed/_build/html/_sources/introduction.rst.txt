Introduction
============

The Harmonic Drive Cycle Monitor package performs cycle monitoring by combining local LNU modeling, Learning Entropy, saliency analysis, PCA-based visualization, and MLP classification in the learned weight space.

The central idea is not to classify raw time series directly. Instead, the software first learns a local dynamic model of each run or cycle. The learned weight vector :math:`\mathbf{w}` is then interpreted as a compact representation of the current dynamic state.

The main quantities are introduced consistently with the notation used in the HONU, IPLNA, and Learning Entropy publications. Scalars are written as italic symbols, vectors as bold lowercase symbols, and matrices as bold uppercase symbols. See :doc:`notation` for the full convention.

The practical processing chain is:

1. load measured or simulated cycle data,
2. construct normalized and possibly PCA-transformed feature vectors :math:`\mathbf{x}(k)`,
3. train the LNU model :math:`\tilde{y}(k)=\mathbf{w}^T(k)\mathbf{x}(k)`,
4. evaluate :math:`\Delta\mathbf{w}` and LE-like novelty measures,
5. reconstruct saliency vector :math:`\mathbf{s}`,
6. classify and visualize the learned dynamic states.
