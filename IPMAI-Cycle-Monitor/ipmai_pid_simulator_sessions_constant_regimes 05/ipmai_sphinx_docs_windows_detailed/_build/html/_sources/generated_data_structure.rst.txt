Generated Data Structure
========================

The simulator writes independent run folders.

Typical structure:

.. code-block:: text

    simulated_runs/
        sim_healthy_001/
            measured.csv
            annotated.csv
            events.csv
            config.txt

        sim_ripple_001/
            measured.csv
            annotated.csv
            events.csv
            config.txt

        sim_low_order_001/
            measured.csv
            annotated.csv
            events.csv
            config.txt

        sim_mesh_ampmod_001/
            measured.csv
            annotated.csv
            events.csv
            config.txt

        sim_impacts_res_001/
            measured.csv
            annotated.csv
            events.csv
            config.txt

``measured.csv``
----------------

Contains only measured or derived signal variables.

It is intended to mimic the file that may be obtained from real industrial measurements.

``annotated.csv``
-----------------

Contains the same variables as ``measured.csv`` plus regime labels and metadata.

It is intended for supervised learning, validation and simulated-data experiments.

``events.csv``
--------------

Stores one-row or event-level metadata describing the run.

``config.txt``
--------------

Stores simulation settings so that the run remains reproducible.

This file is important because different datasets may be generated days or weeks apart with different parameters.
