LNU Analysis
============

Linear Neural Unit
------------------

The package uses a Linear Neural Unit (LNU) as a local in-parameter-linear model. For the selected delayed variables or PCA components, the input vector is denoted by :math:`\mathbf{x}(k)` and the weight vector by :math:`\mathbf{w}(k)`.

.. math::

   \tilde{y}(k)=\mathbf{w}^T(k)\mathbf{x}(k).

The measured or target value is :math:`y(k)`, and the model residual used for adaptation is

.. math::

   e(k)=y(k)-\tilde{y}(k).

Interpretation of weights
-------------------------

The vector :math:`\mathbf{w}` represents the locally learned dynamic law of the observed cycle. Changes of :math:`\mathbf{w}` between cycles therefore indicate changes in the learned dynamics rather than merely changes in raw signal amplitude.

Continual training
------------------

For a run or cycle :math:`j`, the resulting weight vector is denoted by :math:`\mathbf{w}^{(j)}`. The next run is initialized from the previous trained vector whenever the corresponding script uses continual adaptation. This preserves the intended learning behavior of the package.

Weight increments
-----------------

The basic quantity used for novelty and saliency analysis is

.. math::

   \Delta\mathbf{w}^{(j)}=\mathbf{w}^{(j)}-\mathbf{w}^{(j-1)}.

The magnitude and direction of :math:`\Delta\mathbf{w}^{(j)}` are interpreted as the learning effort needed to adapt the local model to the next cycle.
