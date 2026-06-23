# IPMAI PID + Vibro Simulator - Constant-Regime Sessions

This package generates simulated PID-loop and vibro-feature data only. It does **not** perform LNU, PCA, Learning Entropy, saliency, or classification.

Each run/session represents one dominant regime for the whole run:

- healthy,
- original ripple,
- low-order disturbance,
- mesh with amplitude modulation,
- impacts with resonance.

There are no within-run fault transitions in this version. Transition/degradation scenarios should be generated later as a separate dataset type.

## Run

```bash
python run_pid_simulator_sessions_constant.py
```

## Output structure

```text
simulated_runs/
  runs_summary.csv
  global_config.txt
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
  ...
```

## CSV files

`measured.csv` contains unannotated measured/simulated channels:

```text
t, theta_ref, theta, omega, u, e, tau_d, hf_rms, a2x, amesh
```

`annotated.csv` contains the same signals plus labels:

```text
run_id, fault_id, fault_name, fault_active, fault_magnitude, trial, regime_is_constant
```

`events.csv` contains run-level metadata. Since the whole run is one constant regime, it has one row per run.

`config.txt` stores the exact simulator settings used for the run.

## Notes

The simulator keeps the important dual-rate structure from the original script:

- PID/control-loop variables are sampled at 100 Hz.
- Vibro segments are synthesized at 10 kHz inside each control step.
- Features `hf_rms`, `a2x`, and `amesh` are extracted per control sample.

The later LNU/HONU monitoring module should read these session folders and process each run independently.
