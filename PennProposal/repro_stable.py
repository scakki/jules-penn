"""
repro_stable.py — Robustly reproduce a gait from CSV using PD + Feedforward control.

While raw torque replay often drifts due to simulation inaccuracies, a PD+FF
controller (Proportional-Derivative + Feedforward) provides the necessary
stability to track the recorded trajectory.

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
    # We want to apply raw torques (motors) and handle PD in Python
    with open(xml_path) as f:
        xml = f.read()

    # Replace position actuators with motors
    xml = re.sub(r'<position\s+kp="[^"]*"\s+forcelimited="true"\s+ctrllimited="true"\s+forcerange="([^"]*)"/>',
                 r'<motor forcelimited="true" ctrllimited="true" forcerange="\1"/>', xml)
    xml = re.sub(r"<position(\s+class)", r"<motor\1", xml)

    # Set default ranges
    for cls in ("hip_abd_limited", "hip", "thigh", "calf"):
        pattern = rf'(<default\s+class="{cls}">\s*<joint[^/]*/>\s*)<position\s+ctrllimited="true"\s+ctrlrange="[^"]*"/>'
        replacement = rf'\1<motor ctrllimited="true" ctrlrange="-33.5 33.5"/>'
        xml = re.sub(pattern, replacement, xml)

    # Use a temporary file in the same directory to resolve meshes
    xml_dir = os.path.dirname(os.path.abspath(xml_path))
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode="w", dir=xml_dir)
    tmp.write(xml); tmp.close()

    model = mujoco.MjModel.from_xml_path(tmp.name)
    os.unlink(tmp.name)
    return model

def reproduce(xml_path, csv_path, kp=50.0, kd=1.0, render=True, loop=True):
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

    def reset_to_start():
        mujoco.mj_resetData(model, data)
        row = df.iloc[0]
        # Base position and orientation
        data.qpos[0:3] = [row.get("base_x", 0), row.get("base_y", 0), row.get("base_height", 0.474)]
        data.qpos[3:7] = [1, 0, 0, 0] # Upright
        # Base velocity
        data.qvel[0] = row.get("speed", 0)
        # Joint states
        data.qpos[7:13] = q_ref[0]
        data.qvel[6:12] = qd_ref[0]
        mujoco.mj_forward(model, data)

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

                # Current joint states
                q_curr = data.qpos[7:13]
                qd_curr = data.qvel[6:12]

                # PD + FF Control Law:
                # tau = tau_recorded + kp * (q_recorded - q_current) + kd * (qd_recorded - qd_current)
                data.ctrl[:] = tau_ff[i] + kp * (q_ref[i] - q_curr) + kd * (qd_ref[i] - qd_curr)

                for _ in range(frame_skip):
                    mujoco.mj_step(model, data)

                viewer.sync()

                # Real-time pacing
                target_wall = (i * csv_dt)
                elapsed = _time.perf_counter() - wall_start
                if target_wall > elapsed:
                    _time.sleep(target_wall - elapsed)

                step_idx += 1
    else:
        # Headless mode: measure drift
        drifts = []
        for i in range(n_steps):
            q_curr = data.qpos[7:13]
            qd_curr = data.qvel[6:12]
            data.ctrl[:] = tau_ff[i] + kp * (q_ref[i] - q_curr) + kd * (qd_ref[i] - qd_curr)

            drifts.append(np.abs(q_ref[i] - q_curr))
            for _ in range(frame_skip):
                mujoco.mj_step(model, data)

        avg_drift = np.mean(drifts)
        print(f"Reproduction complete (Headless).")
        print(f"Average joint drift: {avg_drift:.5f} rad")
        print(f"Final position: x={data.qpos[0]:.3f}, z={data.qpos[2]:.3f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--xml", default="PennProposal/src/models/bolt_bipedal2.xml")
    p.add_argument("--csv", default="PennProposal/plots/bolt_simplified_Bolt-v3/run_1_20260304-120853/timeseries.csv")
    p.add_argument("--kp", type=float, default=50.0)
    p.add_argument("--kd", type=float, default=2.0)
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--no-loop", action="store_true")
    args = p.parse_args()

    reproduce(args.xml, args.csv, kp=args.kp, kd=args.kd,
              render=not args.no_render, loop=not args.no_loop)
