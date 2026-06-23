Learning Entropy
================

Learning Entropy (LE) is interpreted here as a learning-based novelty measure. It is derived from the effort needed to adapt the local LNU model, not from Shannon-type probability estimates.

Weight increment
----------------

For consecutive runs, windows, or cycles, the learned weight increment is

.. math::

   \Delta\mathbf{w}^{(j)}=\mathbf{w}^{(j)}-\mathbf{w}^{(j-1)}.

Large or unusual values of :math:`\Delta\mathbf{w}^{(j)}` indicate that the local dynamic law has changed and that the learning system must adapt more strongly.

Learning Information and AISLE
------------------------------

For an individual sample or learning step, Learning Information is denoted by :math:`L(k)`. When the individual-sample approximation is emphasized, the notation :math:`AISLE(k)` is used.

In the practical implementation, the score is evaluated from the magnitude or unusualness of the weight increments. A generic normalized form may be written as

.. math::

   AISLE(k) \sim F\left(\left|\Delta\mathbf{w}(k)\right|\right),

where :math:`F(\cdot)` denotes the implemented normalization, thresholding, or neighborhood-based evaluation.

Frame-level Learning Entropy
----------------------------

For a frame of :math:`N` samples or for a sequence of learning steps, Learning Entropy can be interpreted as an aggregation of individual Learning Information values:

.. math::

   LE \approx \frac{1}{N}\sum_{k=1}^{N}L(k).

In this package, the saved LE-like curves are therefore best read as indicators of changing learned dynamics. They are complementary to prediction errors and saliency maps.

Relation to saliency
--------------------

LE indicates how unusual the learning effort is. Saliency further distributes this learning effort back to the delayed variables or PCA-derived components that contributed to the change.
