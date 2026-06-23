MLP Classifier
==============

Purpose
-------

MLP classifies runs using final LNU weights.

Input vector
------------

Each run is represented by:

.. math::

   x_{MLP}=W_{final}

Architecture
------------

Typical architecture:

::

   input -> hidden layer -> output classes

Outputs
-------

::

   mlp_probabilities.csv
   confusion_matrix.csv
   training_summary.json

Interpretation
--------------

The MLP operates in the explainable LNU weight space instead of raw signals.
