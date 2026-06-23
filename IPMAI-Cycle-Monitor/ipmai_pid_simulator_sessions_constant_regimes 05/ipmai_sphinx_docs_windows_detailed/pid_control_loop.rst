PID Control Loop
================

The simulated process is a PID-controlled second-order mechanical system. The control loop is included so that the monitoring pipeline is tested on controlled dynamics, not only on passive open-loop signals.

Mechanical plant
----------------

The plant state is the vector

.. math::

   \mathbf{x}_p(k)=\begin{bmatrix}\theta(k)\\\omega(k)\end{bmatrix},

where :math:`\theta(k)` is angular position and :math:`\omega(k)` is angular velocity. The controlled output is

.. math::

   y(k)=\theta(k).

Reference and regulation deviation
----------------------------------

The desired value is denoted by :math:`d(k)`. The PID control-loop regulation deviation is

.. math::

   e_c(k)=d(k)-y(k).

The subscript in :math:`e_c(k)` distinguishes the control-loop regulation deviation from the LNU model residual :math:`e(k)=y(k)-\tilde{y}(k)`.

Controller
----------

The PID controller computes the control input :math:`u(k)` from proportional, integral, and derivative components of :math:`e_c(k)`. The simulated faults and disturbances affect the plant response, and the LNU module subsequently learns local dynamic laws from the delayed measured variables.

Relation to LNU monitoring
--------------------------

The LNU input vector :math:`\mathbf{x}(k)` may contain delayed values of the reference, control input, measured output, and vibration-derived features. The learned vector :math:`\mathbf{w}` therefore summarizes local controlled dynamics rather than only the raw output trajectory.
