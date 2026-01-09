import numpy as np

# ============================================================
# Setpoint Generator
# ============================================================

def generate_setpoints(
    duration_s=30.0,
    stabilize_every_n_seconds=5,
    stabilize_for=5,
    dt=0.01,                        # 100 Hz
    roll_range_deg=10,
    pitch_range_deg=45,
    vel_range=1.0,
    vel_slew_per_0p2s=0.3,           # m/s per 300 ms
    roll_pert_stddev=0,
    pitch_pert_stddev=0,
    vel_pert_stddev=0,
    random_seed=None
):
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    N = int(duration_s / dt)

    # Limits
    roll_min, roll_max = -roll_range_deg, roll_range_deg
    pitch_min, pitch_max = -pitch_range_deg, pitch_range_deg
    vel_min, vel_max = -vel_range, vel_range

    # Velocity rate limit
    steps_per_0p2s = int(0.2 / dt)
    dv_max = vel_slew_per_0p2s / steps_per_0p2s

    # Allocate arrays
    v_sp = np.zeros(N)
    roll_sp = np.zeros(N)
    pitch_sp = np.zeros(N)

    # Random segment durations
    seg_min = int(0.2 / dt)   # 200 ms
    seg_max = int(2.0 / dt)   # up to 1.5 s
    seg_max_pitch = int(1 / dt)   # up to 1.5 s

    # ------------------------------------------------------------
    # Random piecewise-constant generator
    # ------------------------------------------------------------
    def generate_random_piecewise_signal(low, high, seg_min=seg_min, seg_max=seg_max):
        sig = np.zeros(N)
        i = 0
        while i < N:
            seg_len = np.random.randint(seg_min, seg_max)
            val = np.random.uniform(low, high)
            sig[i:i+seg_len] = val
            i += seg_len
        return sig

    roll_base  = generate_random_piecewise_signal(roll_min, roll_max)
    pitch_base = generate_random_piecewise_signal(pitch_min, pitch_max, seg_max=seg_max_pitch)
    vel_base   = generate_random_piecewise_signal(vel_min, vel_max)

    # ------------------------------------------------------------
    # Small smooth perturbations (filtered noise)
    # ------------------------------------------------------------
    def filtered_noise(scale, cutoff_hz=1.5):
        noise = np.random.randn(N)
        alpha = np.exp(-2 * np.pi * cutoff_hz * dt)
        y = np.zeros(N)
        for k in range(1, N):
            y[k] = alpha * y[k-1] + (1 - alpha) * noise[k]
        return scale * y

    roll_pert  = filtered_noise(roll_pert_stddev)
    pitch_pert = filtered_noise(pitch_pert_stddev)
    vel_pert   = filtered_noise(vel_pert_stddev)


    roll_raw  = roll_base  + roll_pert
    pitch_raw = pitch_base + pitch_pert
    vel_raw   = vel_base   + vel_pert

    # Clip angles
    roll_sp  = np.clip(roll_raw,  roll_min,  roll_max)
    pitch_sp = np.clip(pitch_raw, pitch_min, pitch_max)
    # v_sp = np.clip(vel_raw, vel_min, vel_max)

    # # ------------------------------------------------------------
    # # Enforce velocity slew rate constraint
    # # ------------------------------------------------------------
    # v_sp[0] = np.clip(vel_raw[0], vel_min, vel_max)
    # Forward pass: slew rate limiting from v_sp[0] = 0
    v_sp[0] = 0
    for k in range(1, N):
        dv = vel_raw[k] - v_sp[k-1]
        dv = np.clip(dv, -dv_max, dv_max)
        v_sp[k] = np.clip(v_sp[k-1] + dv, vel_min, vel_max)
    
    # Backward pass: ensure v_sp[-1] = 0 with slew rate limiting
    v_sp[-1] = 0
    for k in range(N-2, -1, -1):
        dv = v_sp[k] - v_sp[k+1]
        dv = np.clip(dv, -dv_max, dv_max)
        v_sp[k] = np.clip(v_sp[k+1] + dv, vel_min, vel_max)

    # ------------------------------------------------------------
    # Insert stabilization segments (zeros)
    # ------------------------------------------------------------
    stabilize_steps = int(stabilize_for / dt)
    interval_steps = int(stabilize_every_n_seconds / dt)
    
    k = interval_steps
    while k < N:
        end_idx = min(k + stabilize_steps, N)
        v_sp[k:end_idx] = 0
        roll_sp[k:end_idx] = 0
        pitch_sp[k:end_idx] = 0
        k += interval_steps + stabilize_steps

    return v_sp, roll_sp, pitch_sp