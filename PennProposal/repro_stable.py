"""
repro_stable.py — Robustly reproduce a gait from CSV using PD + Feedforward control.

This version adds a 'warmup' phase to ensure the robot starts exactly as
recorded, pinning the state for the first few steps.

Usage:
  python repro_stable.py --xml src/models/bolt_bipedal2.xml --csv timeseries.csv
"""

import argparse, os, re
import numpy as np
import pandas as pd
import mujoco
import mujoco.viewer

# Column names in the CSV
Q_COLS   = ["L_HAA_q",   "L_HFE_q",   "L_KFE_q",   "R_HAA_q",   "R_HFE_q",   "R_KFE_q"]
QD_COLS  = ["L_HAA_qd",  "L_HFE_qd",  "L_KFE_qd",  "R_HAA_qd",  "R_HFE_qd",  "R_KFE_qd"]
TAU_COLS = ["L_HAA_tau", "L_HFE_tau", "L_KFE_tau", "R_HAA_tau", "R_HFE_tau", "R_KFE_tau"]

def load_csv(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

def setup_model(xml_path):
    with open(xml_path) as f:
        xml = f.read()

    xml = re.sub(r'<position\s+kp="[^"]*"\s+forcelimited="true"\s+ctrllimited="true"\s+forcerange="([^"]*)"/>',
                 r'<motor forcelimited="true" ctrllimited="true" forcerange="\1"/>', xml)
    xml = re.sub(r"<position(\s+class)", r"<motor\1", xml)

    for cls in ("hip_abd_limited", "hip", "thigh", "calf"):
        pattern = rf'(<default\s+class="{cls}">\s*<joint[^/]*/>\s*)<position\s+ctrllimited="true"\s+ctrlrange="[^"]*"/>'
        replacement = rf'\1<motor ctrllimited="true" ctrlrange="-33.5 33.5"/>'
        xml = re.sub(pattern, replacement, xml)

    xml_dir = os.path.dirname(os.path.abspath(xml_path))
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode="w", dir=xml_dir)
    tmp.write(xml); tmp.close()

    model = mujoco.MjModel.from_xml_path(tmp.name)
    os.unlink(tmp.name)
    return model

def reproduce(xml_path, csv_path, kp=50.0, kd=2.0, render=True, loop=True, warmup_steps=10):
    df = load_csv(csv_path)
    model = setup_model(xml_path)
    data = mujoco.MjData(model)

    sim_dt = model.opt.timestep
    csv_dt = df["time"].iloc[1] - df["time"].iloc[0]
    frame_skip = int(round(csv_dt / sim_dt))

    # Precompute arrays
    q_ref   = df[Q_COLS].values
    qd_ref  = df[QD_COLS].values
    tau_ff  = df[TAU_COLS].values
    n_steps = len(df)

    # Base positions
    base_x = df.get("base_x", pd.Series([0.0]*n_steps)).values
    base_y = df.get("base_y", pd.Series([0.0]*n_steps)).values
    base_z = df.get("base_height", pd.Series([0.474]*n_steps)).values

    # If base velocities are not in CSV, estimate them from positions
    if "base_vx" in df.columns:
        base_vx = df["base_vx"].values
        base_vy = df["base_vy"].values
        base_vz = df["base_vz"].values
    else:
        # Simple finite difference
        base_vx = np.gradient(base_x, csv_dt)
        base_vy = np.gradient(base_y, csv_dt)
        base_vz = np.gradient(base_z, csv_dt)

    def set_state(i):
        data.qpos[0] = base_x[i]
        data.qpos[1] = base_y[i]
        data.qpos[2] = base_z[i]
        # Assuming upright initially or using quaternion if available
        if "base_qw" in df.columns:
            data.qpos[3:7] = [df["base_qw"].iloc[i], df["base_qx"].iloc[i],
                             df["base_qy"].iloc[i], df["base_qz"].iloc[i]]
        else:
            data.qpos[3:7] = [1, 0, 0, 0]

        data.qvel[0] = base_vx[i]
        data.qvel[1] = base_vy[i]
        data.qvel[2] = base_vz[i]

        # Joint states
        data.qpos[7:13] = q_ref[i]
        data.qvel[6:12] = qd_ref[i]
        mujoco.mj_forward(model, data)

    def reset_to_start():
        mujoco.mj_resetData(model, data)
        set_state(0)

    reset_to_start()

    if render:
        import time as _time
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")

            step_idx = 0
            wall_start = _time.perf_counter()

            while viewer.is_running():
                i = step_idx % n_steps
                if i == 0 and step_idx > 0:
                    if not loop: break
                    reset_to_start()
                    wall_start = _time.perf_counter()

                # --- Warmup/Pin Phase ---
                # For the first few steps, we pin the robot's state to match CSV exactly.
                if i < warmup_steps:
                    set_state(i)

                # --- Control Phase ---
                q_curr = data.qpos[7:13]
                qd_curr = data.qvel[6:12]

                # PD + FF Control
                data.ctrl[:] = tau_ff[i] + kp * (q_ref[i] - q_curr) + kd * (qd_ref[i] - qd_curr)

                for _ in range(frame_skip):
                    mujoco.mj_step(model, data)

                viewer.sync()

                target_wall = (i * csv_dt)
                elapsed = _time.perf_counter() - wall_start
                if target_wall > elapsed:
                    _time.sleep(target_wall - elapsed)

                step_idx += 1
    else:
        # Headless mode
        for i in range(n_steps):
            if i < warmup_steps:
                set_state(i)

            q_curr = data.qpos[7:13]
            qd_curr = data.qvel[6:12]
            data.ctrl[:] = tau_ff[i] + kp * (q_ref[i] - q_curr) + kd * (qd_ref[i] - qd_curr)

            for _ in range(frame_skip):
                mujoco.mj_step(model, data)

        print(f"Reproduction complete (Headless). Final x: {data.qpos[0]:.3f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--xml", default="PennProposal/src/models/bolt_bipedal2.xml")
    p.add_argument("--csv", default="PennProposal/plots/bolt_simplified_Bolt-v3/run_1_20260304-120853/timeseries.csv")
    p.add_argument("--kp", type=float, default=50.0)
    p.add_argument("--kd", type=float, default=2.0)
    p.add_argument("--warmup", type=int, default=10, help="Number of initial steps to pin state")
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--no-loop", action="store_true")
    args = p.parse_args()

    reproduce(args.xml, args.csv, kp=args.kp, kd=args.kd, warmup_steps=args.warmup,
              render=not args.no_render, loop=not args.no_loop)
