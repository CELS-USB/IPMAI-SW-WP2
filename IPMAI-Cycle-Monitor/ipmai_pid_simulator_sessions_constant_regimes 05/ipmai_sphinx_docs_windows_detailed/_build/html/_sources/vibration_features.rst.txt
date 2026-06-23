Vibration Synthesis and Features
================================

The simulator has two time scales:

* control-loop scale,
* vibration scale.

Control-loop sampling
---------------------

The PID loop is sampled at:

.. math::

    f_s = 100\ \mathrm{Hz}.

This corresponds to the time step:

.. math::

    T_s = 0.01\ \mathrm{s}.

Vibration sampling
------------------

Within each control interval, a high-frequency vibration segment is synthesized at:

.. math::

    f_{vib} = 10\,000\ \mathrm{Hz}.

This is intended to approximate the fact that industrial systems often have:

* slow control measurements,
* fast vibration or acoustic measurements.

High-frequency envelope RMS
---------------------------

The vibration segment is band-pass filtered. Then the Hilbert envelope is computed and summarized by RMS:

.. math::

    hf\_rms =
    \sqrt{
        \frac{1}{N}
        \sum_{i=1}^{N}
        | \mathcal{H}\{v_{bp}(i)\} |^2
    }.

Practical meaning:

* broad high-frequency energy,
* resonance activity,
* impulsive excitation,
* high-frequency roughness.

Order amplitude ``a2x``
-----------------------

The ``a2x`` variable estimates the amplitude near twice the rotational frequency.

Practical meaning:

* second-order rotating component,
* misalignment-like behavior,
* low-order mechanical artifact.

Mesh amplitude ``amesh``
------------------------

The ``amesh`` variable estimates the amplitude near a high-order mesh-like component.

Practical meaning:

* gear mesh,
* repeated contact,
* high-order angular periodicity.

Why features instead of raw 10 kHz signal?
------------------------------------------

The LNU monitor currently operates at the control-loop time scale. Therefore, high-frequency vibration is summarized into physically meaningful features per control step.

This keeps the model compact while still allowing vibration-related regimes to affect the learned system state.
