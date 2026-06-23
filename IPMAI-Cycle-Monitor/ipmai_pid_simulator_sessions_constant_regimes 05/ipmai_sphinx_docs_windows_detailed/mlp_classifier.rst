MLP Classifier
==============

The MLP classifier is trained on saved LNU weights.

Input
-----

.. math::

    \mathbf{w}^{(j)}.

Output
------

.. math::

    P(c|\mathbf{w}^{(j)}),

where :math:`c` is an operating state.

Current output classes
----------------------

The number of output neurons is determined from the number of detected states in the saved weight dataset.

Examples:

* healthy,
* ripple,
* low-order,
* mesh amplitude modulation,
* impacts and resonance.

Purpose
-------

The MLP is a secondary classifier.

The main representation is still the explainable LNU weight vector. The MLP only maps this learned dynamic state to probabilities of known regimes.

Small-data warning
------------------

At the current stage there may be very few samples. Therefore the MLP results should be interpreted as a proof of concept, not as a validated industrial classifier.
