PCA and Weight-Space Interpretation
===================================

Normalization
-------------

Signals are normalized using the statistics defined by the corresponding reference data or reference run. In the notation of this documentation, a normalized scalar signal is written as

.. math::

   z(k)=\frac{x(k)-\mu_x}{\sigma_x+\varepsilon}.

When multiple variables are processed jointly, the normalized feature vector is denoted by :math:`\mathbf{x}(k)`.

PCA embedding
-------------

PCA may be used to suppress collinearity and to obtain a compact embedding for LNU learning. Let :math:`\mathbf{X}` denote the data matrix whose rows are samples and whose columns are variables or delayed features. PCA gives a loading matrix :math:`\mathbf{V}`. The embedded data are

.. math::

   \mathbf{Z}=\mathbf{X}\mathbf{V}.

If only the first :math:`m` components are retained, then

.. math::

   \mathbf{Z}_m=\mathbf{X}\mathbf{V}_m.

Weight-space PCA
----------------

For visualization of learned run states, the final LNU weight vectors may be stacked into a matrix

.. math::

   \mathbf{W}=\begin{bmatrix}
   (\mathbf{w}^{(1)})^T\\
   (\mathbf{w}^{(2)})^T\\
   \vdots
   \end{bmatrix}.

A two-dimensional visualization of the weight space can then be written as

.. math::

   \mathbf{C}=\mathbf{W}\mathbf{V}_2,

where :math:`\mathbf{V}_2` contains the first two principal directions of the weight-space representation.
