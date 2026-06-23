Notation and Terminology
========================

This documentation follows the notation used in the HONU, IPLNA, and Learning Entropy publications. Scalar variables are written as italic symbols, for example :math:`x`, :math:`y`, :math:`e`, and :math:`k`. Vectors are written as bold lowercase symbols, for example :math:`\mathbf{x}`, :math:`\mathbf{w}`, and :math:`\Delta\mathbf{w}`. Matrices are written as bold uppercase symbols, for example :math:`\mathbf{X}`, :math:`\mathbf{W}`, and :math:`\mathbf{C}`. The discrete-time or sample index is :math:`k`, and :math:`(\cdot)^T` denotes transposition.

Core LNU notation
-----------------

For the linear neural unit (LNU), the input vector, the weight vector, and the neural output are denoted by

.. math::

   \mathbf{x}(k)=\begin{bmatrix}x_0(k)&x_1(k)&\cdots&x_n(k)\end{bmatrix}^T,

.. math::

   \mathbf{w}(k)=\begin{bmatrix}w_0(k)&w_1(k)&\cdots&w_n(k)\end{bmatrix}^T,

and

.. math::

   \tilde{y}(k)=\mathbf{w}^T(k)\mathbf{x}(k).

The measured or target value is denoted by :math:`y(k)`. The model residual used for learning is

.. math::

   e(k)=y(k)-\tilde{y}(k).

In the PID simulation module, the control-loop regulation deviation is written as :math:`e_c(k)=d(k)-y(k)`, where :math:`d(k)` is the desired value and :math:`y(k)` is the controlled output. The subscript distinguishes the control-loop regulation deviation from the LNU model residual.

Weight increments and LE notation
---------------------------------

The weight increment between two consecutive runs, windows, or cycles is

.. math::

   \Delta\mathbf{w}^{(j)}=\mathbf{w}^{(j)}-\mathbf{w}^{(j-1)}.

Learning Information is denoted by :math:`L(k)` when it refers to the novelty of an individual sample or learning step. Approximate Individual Sample Learning Entropy is denoted by :math:`AISLE(k)` when the individual-sample approximation is emphasized. Learning Entropy over a frame or interval is denoted by :math:`LE` and may be interpreted as an aggregation of :math:`L(k)`, for example

.. math::

   LE \approx \frac{1}{N}\sum_{k=1}^{N}L(k).

In this software, the practical LE-like score is computed from the magnitude or unusualness of :math:`\Delta\mathbf{w}`. Thus, LE is not interpreted as a Shannon-probability measure. It is interpreted as a learning-based novelty measure derived from the effort needed to adapt the local dynamic model.

Practical variable names in code
--------------------------------

Some Python variables are kept short for compatibility with existing scripts and stored files. For example, ``W`` in code may store a matrix of run-wise weight vectors and corresponds to :math:`\mathbf{W}` in the documentation. Similarly, ``dW`` corresponds to :math:`\Delta\mathbf{w}` or to a matrix of such increments depending on context. The scientific notation in this documentation is authoritative; the code names are preserved for backward compatibility.
