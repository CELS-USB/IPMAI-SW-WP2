Fault and Artifact Regimes
==========================

The simulator currently includes five dominant regimes.

Healthy regime
--------------

``fault_id = 0``

No artificial disturbance torque is injected except nominal noise-like vibration features.

Practical interpretation:

* normal controlled operation,
* baseline dynamic behavior,
* reference for normalization,
* reference for PCA basis estimation.

Ripple regime
-------------

``fault_id = 1``

The disturbance torque is approximately:

.. math::

    \tau_d(t)=
    \left(A_0 + A_1|\omega(t)|\right)m
    \sin(n_{ripple}\theta(t)),

where:

* :math:`A_0` is a base ripple amplitude,
* :math:`A_1` scales the ripple with velocity,
* :math:`m` is the fault magnitude,
* :math:`n_{ripple}` is the ripple order.

Practical interpretation:

* periodic torque ripple,
* cogging-like behavior,
* repeated angular-position-dependent disturbance,
* eccentricity-like or rotor-related periodic artifact.

Expected signal effect:

* oscillatory modulation of control response,
* visible changes in control signal,
* order-related vibration component.

Low-order oscillation
---------------------

``fault_id = 2``

The disturbance torque is approximately:

.. math::

    \tau_d(t)=A_{order}m\sin(2\theta(t)).

Practical interpretation:

* low-order mechanical asymmetry,
* slow periodic load variation,
* misalignment-like effect,
* low-order angular dependency.

Expected signal effect:

* slower oscillation than high-order ripple,
* broader effect on output and controller action,
* potentially clearer in position or control signal than in high-frequency vibration.

Mesh + amplitude modulation
---------------------------

``fault_id = 3``

The disturbance torque includes high-order angular ripple with amplitude modulation:

.. math::

    \tau_d(t)=
    A_{mesh}m
    \left(1+0.6\sin(2\pi f_{mod}t)\right)
    \sin(n_{ripple}\theta(t)).

Practical interpretation:

* gear-mesh-like periodic component,
* modulation caused by changing load/contact condition,
* envelope-varying vibration component.

Expected signal effect:

* modulated vibration amplitude,
* visible changes in ``amesh``,
* possibly increased ``hf_rms``,
* dynamic pattern that differs from simple stationary ripple.

Impacts + resonance
-------------------

``fault_id = 4``

This regime generates short impacts and a decaying resonant response.

The impact sequence excites a resonance with frequency:

.. math::

    f_{res}

and exponential decay controlled by:

.. math::

    \alpha_{res}.

Practical interpretation:

* bearing impact,
* looseness,
* mechanical contact events,
* short shocks exciting structural resonance.

Expected signal effect:

* high-frequency ringing,
* increased high-frequency envelope,
* stronger ``hf_rms``,
* possible impulsive behavior in vibration features.

Magnitude levels
----------------

Fault regimes can be generated with several magnitudes, for example:

.. code-block:: text

    0.5, 1.0, 2.0

The magnitude scales the intensity of the disturbance or vibration feature. It allows testing whether the learned weights move gradually in weight space as fault severity increases.
