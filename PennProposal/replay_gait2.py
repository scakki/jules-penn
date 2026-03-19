"""
replay_gait_substep.py — Replay Bolt gait with per-substep torques.

If you have the substep CSV (from bolt_simplified_substep.py), use:
  python replay_gait_substep.py --xml bolt_bipedal2.xml --substep-csv substeps_ep0001.csv

This applies the EXACT torque from each 2ms substep, eliminating
the intra-step interpolation error entirely.

If you only have the original 10ms CSV, use --csv for interpolated replay:
  python replay_gait_substep.py --xml bolt_bipedal2.xml --csv timeseries.csv
"""

import argparse, tempfile, os, re
import numpy as np
import pandas as pd
import mujoco
import mujoco.viewer


Q_COLS   = ["L_HAA_q",   "L_HFE_q",   "L_KFE_q",   "R_HAA_q",   "R_HFE_q",   "R_KFE_q"]
TAU_COLS = ["L_HAA_tau", "L_HFE_tau", "L_KFE_tau", "R_HAA_tau", "R_HFE_tau", "R_KFE_tau"]
QD_COLS  = ["L_HAA_qd",  "L_HFE_qd",  "L_KFE_qd",  "R_HAA_qd",  "R_HFE_qd",  "R_KFE_qd"]


def make_torque_xml(xml_path: str) -> str:
    """Replace position actuators with motors for direct torque control."""
    with open(xml_path) as f:
        txt = f.read()
    txt = re.sub(
        r'<position\s+kp="[^"]*"\s+forcelimited="true"\s+ctrllimited="true"\s+forcerange="([^"]*)"/>',
        r'<motor forcelimited="true" ctrllimited="true" forcerange="\1"/>',
        txt,
    )
    txt = re.sub(r"<position(\s+class)", r"<motor\1", txt)
    for cls in ("hip_abd_limited", "hip", "thigh", "calf"):
        pattern = (rf'(<default\s+class="{cls}">\s*<joint[^/]*/>\s*)'
                   rf'<position\s+ctrllimited="true"\s+ctrlrange="[^"]*"/>')
        replacement = rf'\1<motor ctrllimited="true" ctrlrange="-33.5 33.5"/>'
        txt = re.sub(pattern, replacement, txt)
    # Write next to original XML so relative meshdir paths resolve correctly
    xml_dir = os.path.dirname(os.path.abspath(xml_path))
    tmp = tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode="w", dir=xml_dir)
    tmp.write(txt); tmp.close()
    return tmp.name


def load_substep_csv(csv_path):
    """Load substep CSV → per-substep tau, q, qd, base state arrays."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    N = len(df)
    tau = np.column_stack([df[f"tau_{i}"].values for i in range(6)])
    q   = np.column_stack([df[f"q_{i}"].values for i in range(6)])
    qd  = np.column_stack([df[f"qd_{i}"].values for i in range(6)])
    ctrl = np.column_stack([df[f"ctrl_{i}"].values for i in range(6)])

    # Base state
    base_pos = np.column_stack([df["base_x"], df["base_y"], df["base_z"]])
    base_quat = np.column_stack([df["base_qw"], df["base_qx"], df["base_qy"], df["base_qz"]])

    return {
        "n": N, "time": df["time"].values,
        "tau": tau, "q": q, "qd": qd, "ctrl": ctrl,
        "base_pos": base_pos, "base_quat": base_quat,
        "ctrl_step": df["ctrl_step"].values, "substep": df["substep"].values,
    }


def load_original_csv(csv_path, frame_skip=5):
    """Load original 10ms CSV → expand to per-substep with interpolation."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    N_ctrl = len(df)
    tau_ctrl = df[TAU_COLS].values.astype(np.float64)   # (N_ctrl, 6)
    q_ctrl   = df[Q_COLS].values.astype(np.float64)
    qd_ctrl  = df[QD_COLS].values.astype(np.float64)

    # Interpolate torques between consecutive control steps
    N_sub = N_ctrl * frame_skip
    tau_sub = np.zeros((N_sub, 6))
    ctrl_step = np.zeros(N_sub, dtype=int)
    substep_idx = np.zeros(N_sub, dtype=int)

    for i in range(N_ctrl):
        for s in range(frame_skip):
            idx = i * frame_skip + s
            ctrl_step[idx] = i
            substep_idx[idx] = s
            if i < N_ctrl - 1:
                alpha = s / frame_skip
                tau_sub[idx] = tau_ctrl[i] + alpha * (tau_ctrl[i + 1] - tau_ctrl[i])
            else:
                tau_sub[idx] = tau_ctrl[i]

    dt_sub = float(df["time"].iloc[1] - df["time"].iloc[0]) / frame_skip
    time_sub = np.arange(N_sub) * dt_sub + float(df["time"].iloc[0])

    # Base state (hold constant within control step; we'll set from CSV)
    base_pos = np.zeros((N_sub, 3))
    base_quat = np.zeros((N_sub, 4))
    for i in range(N_ctrl):
        x = df["base_x"].iloc[i]
        y = df["base_y"].iloc[i]
        z = df["base_height"].iloc[i]
        for s in range(frame_skip):
            idx = i * frame_skip + s
            base_pos[idx] = [x, y, z]
            base_quat[idx] = [1, 0, 0, 0]

    return {
        "n": N_sub, "time": time_sub,
        "tau": tau_sub, "q": None, "qd": None, "ctrl": None,
        "base_pos": base_pos, "base_quat": base_quat,
        "ctrl_step": ctrl_step, "substep": substep_idx,
        "q_ctrl": q_ctrl, "qd_ctrl": qd_ctrl,  # for drift measurement
        "n_ctrl": N_ctrl, "frame_skip": frame_skip,
    }


def replay(xml_path, data_dict, playback_speed=1.0, render=True, loop=True,
           warmup_substeps=0):
    N = data_dict["n"]
    tau = data_dict["tau"]
    sim_dt = 0.002  # from XML

    # Determine control-step boundaries for drift reporting
    ctrl_steps = data_dict["ctrl_step"]
    frame_skip = int(np.max(data_dict["substep"])) + 1

    print(f"Substeps: {N} | sim_dt={sim_dt}s | duration={N*sim_dt:.2f}s")
    print(f"Frame skip: {frame_skip} | Warmup: {warmup_substeps} substeps ({warmup_substeps*sim_dt:.3f}s)")

    # ── load motor model ──
    tmp_xml = make_torque_xml(xml_path)
    model = mujoco.MjModel.from_xml_path(tmp_xml)
    os.unlink(tmp_xml)
    assert abs(model.opt.timestep - sim_dt) < 1e-6, f"XML timestep {model.opt.timestep} != expected {sim_dt}"

    d = mujoco.MjData(model)

    # ── set initial state ──
    def set_state_from_data(idx):
        d.qpos[0:3] = data_dict["base_pos"][idx]
        d.qpos[3:7] = data_dict["base_quat"][idx]
        if data_dict["q"] is not None:
            d.qpos[7:13] = data_dict["q"][idx]
        if data_dict["qd"] is not None:
            d.qvel[6:12] = data_dict["qd"][idx]
        mujoco.mj_forward(model, d)

    def reset():
        mujoco.mj_resetData(model, d)
        set_state_from_data(0)

    reset()

    if render:
        import time as _time

        with mujoco.viewer.launch_passive(model, d) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
            viewer.cam.distance = 2.0
            viewer.cam.elevation = -20

            step_idx = 0
            wall_start = _time.perf_counter()

            while viewer.is_running():
                i = step_idx % N

                if i == 0 and step_idx > 0:
                    if not loop:
                        break
                    reset()
                    wall_start = _time.perf_counter()

                # Warmup: pin state, then step
                if i < warmup_substeps:
                    set_state_from_data(i)

                # Apply torque and step ONE substep
                d.ctrl[:] = tau[i]
                mujoco.mj_step(model, d)

                # Sync viewer at control-step boundaries (every frame_skip substeps)
                if data_dict["substep"][i] == frame_skip - 1 or i == N - 1:
                    viewer.sync()

                # Real-time pacing (at control-step boundaries)
                if data_dict["substep"][i] == frame_skip - 1:
                    sim_time = (i + 1) * sim_dt
                    wall_elapsed = _time.perf_counter() - wall_start
                    target_wall = sim_time / playback_speed
                    sleep = target_wall - wall_elapsed
                    if sleep > 0:
                        _time.sleep(sleep)

                step_idx += 1

    else:
        # Headless with drift report
        drift_at_ctrl = []
        q_ctrl_ref = data_dict.get("q_ctrl")  # only available for original CSV
        q_sub_ref = data_dict.get("q")         # available for substep CSV

        for i in range(N):
            if i < warmup_substeps:
                set_state_from_data(i)

            d.ctrl[:] = tau[i]
            mujoco.mj_step(model, d)

            # Measure drift at end of each control step
            is_last_substep = (data_dict["substep"][i] == frame_skip - 1) or (i == N - 1)
            if is_last_substep:
                q_actual = d.qpos[7:13].copy()
                cs = int(ctrl_steps[i])

                if q_sub_ref is not None:
                    q_ref = q_sub_ref[i]
                elif q_ctrl_ref is not None and cs < len(q_ctrl_ref):
                    q_ref = q_ctrl_ref[cs]
                else:
                    q_ref = None

                if q_ref is not None:
                    drift_at_ctrl.append(q_actual - q_ref)

        # Report
        if drift_at_ctrl:
            errs = np.array(drift_at_ctrl)
            rmse = np.sqrt(np.mean(errs ** 2, axis=0))
            max_err = np.max(np.abs(errs), axis=0)
            joint_names = ["FL_HAA", "FL_HFE", "FL_KFE", "FR_HAA", "FR_HFE", "FR_KFE"]
            print(f"\n{'='*60}")
            print(f"DRIFT REPORT  (warmup={warmup_substeps} substeps)")
            print(f"{'='*60}")
            for j, name in enumerate(joint_names):
                print(f"  {name:8s}  RMSE={rmse[j]:.5f} rad  MAX={max_err[j]:.5f} rad")
            print(f"\n  Overall RMSE: {np.sqrt(np.mean(errs**2)):.5f} rad")
            print(f"  Final:  x={d.qpos[0]:.3f}  z={d.qpos[2]:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Replay Bolt gait from substep or original CSV")
    p.add_argument("--xml", default="bolt_bipedal2.xml")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--substep-csv", help="Substep CSV (from modified env, 2ms resolution)")
    g.add_argument("--csv", help="Original timeseries CSV (10ms, will be interpolated)")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--warmup", type=int, default=0,
                   help="Pin state from data for first N substeps")
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--no-loop", action="store_true")
    args = p.parse_args()

    if args.substep_csv:
        print(f"Loading SUBSTEP CSV (exact per-substep torques): {args.substep_csv}")
        data_dict = load_substep_csv(args.substep_csv)
    else:
        print(f"Loading original CSV (will interpolate torques): {args.csv}")
        data_dict = load_original_csv(args.csv)

    replay(args.xml, data_dict, playback_speed=args.speed,
           render=not args.no_render, loop=not args.no_loop,
           warmup_substeps=args.warmup)