Variables and Their Meaning
===========================

This page lists the main variables stored in the generated CSV files.

Time variables
--------------

``t``
    Time in seconds at the control-loop sampling rate.

``k``
    Discrete sample index. It may not be explicitly stored in every file, but it is implied by the row number.

Reference and control variables
-------------------------------

``theta_ref``
    Desired position or setpoint. This is the reference signal that the PID loop should follow.

    In the LNU analysis this is used as:

    .. math::

        d(k)=\theta_{ref}(k).

``theta``
    Simulated controlled position. This is the main controlled output of the system.

    In the LNU analysis this is used as:

    .. math::

        y(k)=\theta(k).

``omega``
    Angular velocity. It is the derivative-like state of the mechanical plant.

``u``
    PID control command. In an industrial interpretation this may correspond to actuator command, motor torque command, or current-like control variable.

    In the LNU analysis this is used as:

    .. math::

        u(k).

``e``
    Control-loop regulation deviation stored by the simulator. In the documentation notation it is written with the subscript :math:`e_c(k)` to distinguish it from the LNU model residual:

    .. math::

        e_c(k)=d(k)-y(k)=\theta_{ref}(k)-\theta(k).

Disturbance and artifact variables
----------------------------------

``tau_d``
    Disturbance torque injected into the mechanical plant. It directly affects the mechanical equation of motion.

``hf_rms``
    High-frequency envelope RMS extracted from the simulated vibration segment. It represents broad high-frequency vibration activity.

``a2x``
    Amplitude of the second-order rotational component. It is computed from the simulated vibration segment using a single-frequency amplitude estimate.

``amesh``
    Gear-mesh-like amplitude component. It approximates a high-order vibration component related to repeated mechanical contact or meshing.

Annotation variables
--------------------

These are stored in the annotated CSV file.

``fault_id``
    Numerical identifier of the operating regime.

``fault_name``
    Human-readable name of the regime.

``fault_active``
    Binary indicator. In the current constant-regime dataset it is usually 0 for healthy and 1 for fault regimes.

``fault_magnitude``
    Scaling factor of the fault intensity.

``trial_id``
    Repetition number.

``run_id``
    Identifier of the generated run/session.

Practical note
--------------

The unannotated ``measured.csv`` should resemble what may be available in practice: measured variables without explicit fault labels. The ``annotated.csv`` file is intended for training, validation and simulation studies.
