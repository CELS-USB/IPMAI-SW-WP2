Complete Pipeline
=================

The current pipeline consists of five main layers.

1. Simulation or data acquisition
---------------------------------

The simulator generates run folders. Each run represents one dominant operating regime, for example healthy operation or one type of fault.

2. Signal visualization
-----------------------

The PySide6 viewers load the run folders and display either all variables of one run or one selected variable across all operating regimes.

3. LNU modeling
---------------

For each run, an LNU model is trained. In the current first-stage approach, one complete run is treated as one batch-learning window.

4. Weight-space analysis
------------------------

The learned LNU weights are projected to a two-dimensional PCA plane. This plane is interpreted as an empirical map of learned dynamic states.

5. MLP classification
---------------------

A small feedforward MLP can be trained on the learned LNU weight vectors. The MLP output estimates probabilities of operating states.

The full conceptual pipeline is:

.. code-block:: text

    measured.csv / simulated runs
        ↓
    signal inspection
        ↓
    healthy normalization
        ↓
    healthy PCA basis
        ↓
    LNU learning per run
        ↓
    learned weights
        ↓
    weight-space PCA
        ↓
    Learning Entropy and saliency
        ↓
    optional MLP classifier
