Practical Industrial Interpretation
===================================

Why simulate a PID loop?
------------------------

Real industrial processes are often controlled. The observed output is not only a passive response of a machine. It is affected by feedback.

A fault can therefore appear in:

* the output signal,
* the control signal,
* the regulation error,
* vibration features,
* the learned relation between input and output.

Why use LNU weights?
--------------------

Raw signals can be noisy and regime-dependent. LNU weights summarize the local dynamic relation between reference, actuator and output.

Thus:

.. math::

    \mathbf{w}^{(j)}

acts as a compact learned descriptor of the system state.

Why use healthy normalization?
------------------------------

Healthy normalization defines what is normal. Fault runs are then evaluated in the same coordinate system.

Why use PCA from healthy operation?
-----------------------------------

A shared PCA basis ensures that weights learned on different runs are comparable.

Why use MLP after LNU?
----------------------

The MLP does not replace the explainable model. It only learns a decision boundary in the space of already learned dynamic states.

This keeps the diagnostic pipeline more interpretable than direct black-box classification of raw signals.
