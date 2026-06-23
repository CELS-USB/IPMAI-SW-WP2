Simulator Principle
===================

The simulator is not intended to be a perfect physical model of a particular machine. It is a controlled research generator of realistic dynamical patterns.

It combines:

* a second-order mechanical plant,
* a discrete PID controller,
* reference trajectory changes,
* disturbance torque models,
* high-frequency vibration synthesis,
* feature extraction from vibration-like signals,
* annotated operating regimes.

The goal is to generate data where different faults are not represented only by artificial labels, but also by physically motivated changes in the signals.

The simulator approximates the following industrial situation:

* a mechanical axis or rotary system is controlled by a PID loop,
* the measured output is the controlled position,
* the actuator current or command is available,
* the system may be affected by torque disturbances,
* vibration-related indicators can be measured or extracted,
* different fault mechanisms produce different dynamic signatures.

Important design choice
-----------------------

The current dataset generation follows:

.. code-block:: text

    1 run = 1 dominant operating regime

This means that a run is either healthy or belongs to one fault regime. This is useful for the first phase:

* clear class separability,
* simpler interpretation of weight-space clusters,
* simple training of the MLP classifier,
* easier validation of the LNU-based representation.

Later versions may include transition runs, for example:

.. code-block:: text

    healthy → fault onset → degradation → recovery
