"""
IPMAI PID + vibro simulator, session-based constant-regime version.

Each generated run/session represents one dominant regime for the whole run:
healthy OR one fault type. No LNU/PCA/LE processing is performed here.

Outputs per run:
    measured.csv   - unannotated measured variables only
    annotated.csv  - measured variables + fault/regime metadata
    events.csv     - run-level event/label metadata
    config.txt     - full simulator configuration used for the run

Windows usage:
    python run_pid_simulator_sessions_constant.py
"""

# ============================================================
# USER CONFIG - all important settings are at the top
# ============================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.signal import firwin, lfilter, hilbert

# ---- output ----
OUTPUT_ROOT = Path("simulated_runs")
OVERWRITE_OUTPUT = True
CSV_FLOAT_FORMAT = "%.10g"

# ---- time + sampling ----
T = 20.0                       # duration of each independent run [s]
Ts = 1.0 / 100.0               # 100 Hz control loop sampling
FS_VIB = 10000.0               # 10 kHz vibro sampling inside each control sample

# ---- reference signal ----
REFERENCE_KIND = "steps"       # "steps" or "sine"
REFERENCE_STEP_LEVELS = [0.0, 1.0, -1.0, 0.0]
REFERENCE_STEP_TIMES = [0.0, 4.0, 10.0, 16.0]
REFERENCE_SINE_AMPLITUDE = 0.8
REFERENCE_SINE_FREQUENCY = 0.08
REFERENCE_SINE_OFFSET = 0.0

# ---- plant parameters ----
J = 0.05
B = 0.2
K = 0.0

# ---- PID gains ----
KP = 30.0
KI = 40.0
KD = 1.0

# ---- simulator cases: one run = one constant regime ----
# fault_id 0 is healthy. fault_id 1..4 follow the original script families.
RUN_DEFINITIONS = [
    dict(run_id="sim_healthy_001", fault_id=0, fault_name="healthy", magnitude=1.0, trial=0),
    dict(run_id="sim_ripple_001", fault_id=1, fault_name="orig_ripple", magnitude=1.0, trial=0),
    dict(run_id="sim_ripple_002", fault_id=1, fault_name="orig_ripple", magnitude=2.0, trial=1),
    dict(run_id="sim_low_order_001", fault_id=2, fault_name="low_order", magnitude=1.0, trial=0),
    dict(run_id="sim_mesh_ampmod_001", fault_id=3, fault_name="mesh_ampmod", magnitude=1.0, trial=0),
    dict(run_id="sim_impacts_res_001", fault_id=4, fault_name="impacts_res", magnitude=1.0, trial=0),
]

BASE_SEED = 123

# ---- vibro feature extraction ----
BP_LO = 200.0
BP_HI = 2500.0
BP_NUMTAPS = 201

# ---- artefact parameters, based on the previous PID/vibro script ----
N_RIPPLE = 80
A0 = 0.2
A1 = 0.5
A_ORDER = 0.25
A_MESH = 0.18
MOD_F = 12.0
IMPACT_F = 20.0
F_RES = 1600.0
ALPHA_RES = 350.0
VIB_NOISE_STD = 0.03
BASE_TAU_IMPULSE = 0.10
BASE_VIB_RING = 0.7

# ---- optional measured noise on slow variables ----
ADD_SLOW_MEASUREMENT_NOISE = False
THETA_NOISE_STD = 0.0
OMEGA_NOISE_STD = 0.0
U_NOISE_STD = 0.0

# ============================================================
# Derived global grids
# ============================================================

T_GRID = np.arange(0.0, T + Ts, Ts)
N = len(T_GRID)
DT_VIB = 1.0 / FS_VIB
NV = int(np.round(Ts / DT_VIB))
T_VIB_LOCAL = np.arange(NV) * DT_VIB
BP = firwin(BP_NUMTAPS, [BP_LO, BP_HI], pass_zero=False, fs=FS_VIB)


# ============================================================
# Simulator helpers
# ============================================================

def build_reference(t_grid):
    """Build theta reference d(k)."""
    if REFERENCE_KIND == "sine":
        return REFERENCE_SINE_OFFSET + REFERENCE_SINE_AMPLITUDE * np.sin(
            2.0 * np.pi * REFERENCE_SINE_FREQUENCY * t_grid
        )

    if REFERENCE_KIND != "steps":
        raise ValueError("REFERENCE_KIND must be 'steps' or 'sine'.")

    d = np.zeros_like(t_grid)
    levels = list(REFERENCE_STEP_LEVELS)
    times = list(REFERENCE_STEP_TIMES)
    if len(levels) != len(times):
        raise ValueError("REFERENCE_STEP_LEVELS and REFERENCE_STEP_TIMES must have the same length.")

    for level, t0 in zip(levels, times):
        d[t_grid >= t0] = level
    return d


def goertzel_amp(x, fs, f0):
    """Magnitude at frequency f0 via a single-bin DFT / Goertzel approximation."""
    if f0 <= 0.0 or f0 >= fs / 2.0:
        return 0.0
    n = len(x)
    k_bin = int(np.round(n * f0 / fs))
    w = 2.0 * np.pi * k_bin / n
    c = 2.0 * np.cos(w)
    s1 = 0.0
    s2 = 0.0
    for xn in x:
        s0 = xn + c * s1 - s2
        s2 = s1
        s1 = s0
    real = s1 - s2 * np.cos(w)
    imag = s2 * np.sin(w)
    return np.sqrt(real * real + imag * imag) / (n / 2.0 + 1e-12)


def vib_features(vib_seg, omega_mean, n_mesh):
    """Per slow control sample vibro features: HF envelope RMS, 2x order, mesh proxy."""
    vbp = lfilter(BP, [1.0], vib_seg)
    env = np.abs(hilbert(vbp))
    hf_env_rms = np.sqrt(np.mean(env ** 2))

    fin = abs(omega_mean) / (2.0 * np.pi)
    a_2x = goertzel_amp(vib_seg, FS_VIB, 2.0 * fin)
    a_mesh = goertzel_amp(vib_seg, FS_VIB, n_mesh * fin)
    return hf_env_rms, a_2x, a_mesh


def disturbance_torque(th, om, t, fault_id, magnitude):
    """Disturbance torque for one constant fault family."""
    if fault_id == 0:
        return 0.0
    if fault_id == 1:
        amp = (A0 + A1 * abs(om)) * magnitude
        return amp * np.sin(N_RIPPLE * th)
    if fault_id == 2:
        return (A_ORDER * magnitude) * np.sin(2.0 * th)
    if fault_id == 3:
        amp = (A_MESH * magnitude) * (1.0 + 0.6 * np.sin(2.0 * np.pi * MOD_F * t))
        return amp * np.sin(N_RIPPLE * th)
    if fault_id == 4:
        phase = t * IMPACT_F
        hit = 1.0 if (phase - np.floor(phase)) < (DT_VIB * IMPACT_F) else 0.0
        return (BASE_TAU_IMPULSE * magnitude) * hit
    raise ValueError(f"Unknown fault_id={fault_id}.")


def vibro_segment(th0, om0, t0, fault_id, magnitude, rng):
    """High-rate vibro segment inside one slow control sample."""
    t = t0 + T_VIB_LOCAL
    th = th0 + om0 * T_VIB_LOCAL
    vib = np.zeros_like(t)

    if fault_id == 0:
        pass
    elif fault_id == 1:
        amp = (A0 + A1 * abs(om0)) * magnitude
        vib += 0.8 * amp * np.sin(N_RIPPLE * th)
    elif fault_id == 2:
        vib += (A_ORDER * magnitude) * np.sin(2.0 * th)
    elif fault_id == 3:
        amp = (A_MESH * magnitude) * (1.0 + 0.6 * np.sin(2.0 * np.pi * MOD_F * (t - t0)))
        vib += amp * np.sin(N_RIPPLE * th)
    elif fault_id == 4:
        impulses = (np.mod((t - t0) * IMPACT_F, 1.0) < (DT_VIB * IMPACT_F)).astype(float)
        ring = np.zeros_like(t)
        a_state = 0.0
        b_state = 0.0
        w = 2.0 * np.pi * F_RES
        decay = np.exp(-ALPHA_RES * DT_VIB)
        c = np.cos(w * DT_VIB)
        s = np.sin(w * DT_VIB)
        for i, impulse in enumerate(impulses):
            if impulse > 0.5:
                a_state += 1.0
            a_new = decay * (c * a_state - s * b_state)
            b_new = decay * (s * a_state + c * b_state)
            a_state, b_state = a_new, b_new
            ring[i] = a_state
        vib += (BASE_VIB_RING * magnitude) * ring
    else:
        raise ValueError(f"Unknown fault_id={fault_id}.")

    vib += VIB_NOISE_STD * rng.standard_normal(len(vib))
    return vib


def simulate_run(run_def):
    """Simulate one independent run with one constant operating/fault regime."""
    fault_id = int(run_def["fault_id"])
    fault_name = str(run_def["fault_name"])
    magnitude = float(run_def["magnitude"])
    trial = int(run_def.get("trial", 0))
    run_id = str(run_def["run_id"])

    seed = BASE_SEED + 10000 * fault_id + 1000 * trial + int(round(100 * magnitude))
    rng = np.random.default_rng(seed)

    d = build_reference(T_GRID)
    theta = np.zeros(N)
    omega = np.zeros(N)
    u_cmd = np.zeros(N)
    tau_d = np.zeros(N)
    e_hist = np.zeros(N)
    hf_rms = np.zeros(N)
    a2x = np.zeros(N)
    amesh = np.zeros(N)

    integral = 0.0
    e_prev = 0.0
    state = np.array([0.0, 0.0], dtype=float)

    def rhs(t, x, u):
        th, om = x
        td = disturbance_torque(th, om, t, fault_id, magnitude)
        thdot = om
        omdot = (-B * om - K * th + u + td) / J
        return [thdot, omdot]

    for k in range(N - 1):
        th, om = state
        theta[k] = th
        omega[k] = om

        td = disturbance_torque(th, om, T_GRID[k], fault_id, magnitude)
        tau_d[k] = td

        e = d[k] - th
        e_hist[k] = e
        integral += Ts * e
        deriv = (e - e_prev) / Ts
        u = KP * e + KI * integral + KD * deriv
        u_cmd[k] = u
        e_prev = e

        vib = vibro_segment(th, om, T_GRID[k], fault_id, magnitude, rng)
        hf_rms[k], a2x[k], amesh[k] = vib_features(vib, omega_mean=om, n_mesh=N_RIPPLE)

        sol = solve_ivp(
            lambda tt, xx: rhs(tt, xx, u),
            [T_GRID[k], T_GRID[k + 1]],
            state,
            max_step=Ts / 20.0,
        )
        state = sol.y[:, -1]

    theta[-1], omega[-1] = state
    e_hist[-1] = d[-1] - theta[-1]
    u_cmd[-1] = u_cmd[-2]
    tau_d[-1] = disturbance_torque(theta[-1], omega[-1], T_GRID[-1], fault_id, magnitude)
    hf_rms[-1], a2x[-1], amesh[-1] = hf_rms[-2], a2x[-2], amesh[-2]

    if ADD_SLOW_MEASUREMENT_NOISE:
        theta_meas = theta + THETA_NOISE_STD * rng.standard_normal(N)
        omega_meas = omega + OMEGA_NOISE_STD * rng.standard_normal(N)
        u_meas = u_cmd + U_NOISE_STD * rng.standard_normal(N)
    else:
        theta_meas = theta.copy()
        omega_meas = omega.copy()
        u_meas = u_cmd.copy()

    measured = pd.DataFrame({
        "t": T_GRID,
        "theta_ref": d,
        "theta": theta_meas,
        "omega": omega_meas,
        "u": u_meas,
        "e": e_hist,
        "tau_d": tau_d,
        "hf_rms": hf_rms,
        "a2x": a2x,
        "amesh": amesh,
    })

    annotated = measured.copy()
    annotated.insert(0, "run_id", run_id)
    annotated["fault_id"] = fault_id
    annotated["fault_name"] = fault_name
    annotated["fault_active"] = int(fault_id != 0)
    annotated["fault_magnitude"] = magnitude
    annotated["trial"] = trial
    annotated["regime_is_constant"] = 1

    events = pd.DataFrame([
        dict(
            run_id=run_id,
            t_start=float(T_GRID[0]),
            t_end=float(T_GRID[-1]),
            fault_id=fault_id,
            fault_name=fault_name,
            fault_active=int(fault_id != 0),
            fault_magnitude=magnitude,
            trial=trial,
            seed=seed,
            description="Constant regime for entire run; no within-run fault transitions.",
        )
    ])

    return measured, annotated, events, seed


def simulator_config_dict(run_def=None, seed=None):
    """Collect settings for reproducibility."""
    return {
        "run_definition": run_def,
        "seed": seed,
        "output": {
            "output_root": str(OUTPUT_ROOT),
            "overwrite_output": OVERWRITE_OUTPUT,
            "csv_float_format": CSV_FLOAT_FORMAT,
        },
        "sampling": {
            "T": T,
            "Ts": Ts,
            "fs_control": 1.0 / Ts,
            "fs_vib": FS_VIB,
            "dt_vib": DT_VIB,
            "Nv_per_control_sample": NV,
        },
        "reference": {
            "kind": REFERENCE_KIND,
            "step_levels": REFERENCE_STEP_LEVELS,
            "step_times": REFERENCE_STEP_TIMES,
            "sine_amplitude": REFERENCE_SINE_AMPLITUDE,
            "sine_frequency": REFERENCE_SINE_FREQUENCY,
            "sine_offset": REFERENCE_SINE_OFFSET,
        },
        "plant": {"J": J, "B": B, "K": K},
        "pid": {"Kp": KP, "Ki": KI, "Kd": KD},
        "vibro_features": {
            "bp_lo": BP_LO,
            "bp_hi": BP_HI,
            "bp_numtaps": BP_NUMTAPS,
        },
        "artefacts": {
            "n_ripple": N_RIPPLE,
            "A0": A0,
            "A1": A1,
            "A_order": A_ORDER,
            "A_mesh": A_MESH,
            "mod_f": MOD_F,
            "impact_f": IMPACT_F,
            "f_res": F_RES,
            "alpha_res": ALPHA_RES,
            "vib_noise_std": VIB_NOISE_STD,
            "base_tau_impulse": BASE_TAU_IMPULSE,
            "base_vib_ring": BASE_VIB_RING,
        },
        "slow_measurement_noise": {
            "enabled": ADD_SLOW_MEASUREMENT_NOISE,
            "theta_noise_std": THETA_NOISE_STD,
            "omega_noise_std": OMEGA_NOISE_STD,
            "u_noise_std": U_NOISE_STD,
        },
    }


def write_run(run_def):
    measured, annotated, events, seed = simulate_run(run_def)
    run_dir = OUTPUT_ROOT / str(run_def["run_id"])
    run_dir.mkdir(parents=True, exist_ok=True)

    measured.to_csv(run_dir / "measured.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    annotated.to_csv(run_dir / "annotated.csv", index=False, float_format=CSV_FLOAT_FORMAT)
    events.to_csv(run_dir / "events.csv", index=False, float_format=CSV_FLOAT_FORMAT)

    cfg = simulator_config_dict(run_def=run_def, seed=seed)
    with open(run_dir / "config.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(cfg, indent=2, ensure_ascii=False))
        f.write("\n")

    return run_dir


def main():
    if OUTPUT_ROOT.exists() and OVERWRITE_OUTPUT:
        for child in OUTPUT_ROOT.iterdir():
            if child.is_dir():
                for f in child.iterdir():
                    f.unlink()
                child.rmdir()
            else:
                child.unlink()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    created = []
    for run_def in RUN_DEFINITIONS:
        run_dir = write_run(run_def)
        created.append(run_dir)
        print(f"Created {run_dir}")

    summary = pd.DataFrame(RUN_DEFINITIONS)
    summary.to_csv(OUTPUT_ROOT / "runs_summary.csv", index=False)

    with open(OUTPUT_ROOT / "global_config.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(simulator_config_dict(run_def=None, seed=None), indent=2, ensure_ascii=False))
        f.write("\n")

    print("\nDone. Generated constant-regime sessions:")
    for path in created:
        print(f"  {path}")


if __name__ == "__main__":
    main()
