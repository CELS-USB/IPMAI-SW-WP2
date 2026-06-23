Saliency Maps
=============

Purpose
-------

Saliency maps identify which delayed variables, PCA components, or reconstructed physical features contribute most to a change in the learned dynamic law.

Definition
----------

For run or segment :math:`j`, the saliency vector is denoted by

.. math::

   \mathbf{s}^{(j)}.

The vector :math:`\mathbf{s}^{(j)}` is reconstructed from :math:`\Delta\mathbf{w}^{(j)}` and from the same feature mapping that was used to construct the LNU input vector :math:`\mathbf{x}(k)`.

Interpretation
--------------

A large component of :math:`\mathbf{s}^{(j)}` means that the corresponding variable, delay, or reconstructed feature contributed strongly to the observed weight change. Thus, the map should be interpreted as a contribution to the learned dynamic law, not as a generic black-box feature importance score.

Axis labels
-----------

The y-axis labels in the saliency viewer should therefore correspond to meaningful delayed variables or reconstructed physical quantities whenever this information is available. If only numerical labels are available, they denote components of :math:`\mathbf{x}(k)` or components reconstructed from the PCA embedding.
