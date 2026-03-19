"""
repro_stable.py — Robustly reproduce a gait from CSV using PD + Feedforward control.

Logs joint position, velocity, torque, GRF and their errors at each time step.
At the end, outputs RMS tracking error and total PD correction effort.

Usage:
  python repro_stable.py --xml src/models/bolt_bipedal2.xml --csv timeseries.csv
"""

import argparse, os, re
import numpy as np
import pandas as pd
import mujoco
import mujoco.viewer

# Column names in the CSV
JOINT_NAMES = ["L_HAA", "L_HFE", "L_KFE", "R_HAA", "R_HFE", "R_KFE"]
Q_COLS   = [f"{j}_q" for j in JOINT_NAMES]
QD_COLS  = [f"{j}_qd" for j in JOINT_NAMES]
TAU_COLS = [f"{j}_tau" for j in JOINT_NAMES]
GRF_COLS = ["left_Fz", "right_Fz"]

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

def compute_grf(model, data):
    """Simple vertical GRF sum for each foot."""
    # Foot geoms (standard naming in Bolt)
    geom_FL = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "FL_foot_col")
    geom_FR = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "FR_foot_col")
    geom_floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    fz_L, fz_R = 0.0, 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        if (c.geom1 == geom_FL and c.geom2 == geom_floor) or (c.geom2 == geom_FL and c.geom1 == geom_floor):
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            fz_L += abs(force[0])
        elif (c.geom1 == geom_FR and c.geom2 == geom_floor) or (c.geom2 == geom_FR and c.geom1 == geom_floor):
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            fz_R += abs(force[0])
    return fz_L, fz_R

def reproduce(xml_path, csv_path, kp=50.0, kd=2.0, render=True, loop=True, warmup_steps=10, threshold=0.05):
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
    grf_ref = df[GRF_COLS].values
    n_steps = len(df)

    # Base positions and velocities
    base_x, base_y, base_z = df["base_x"].values, df["base_y"].values, df["base_height"].values
    base_vx = np.gradient(base_x, csv_dt)
    base_vy = np.gradient(base_y, csv_dt)
    base_vz = np.gradient(base_z, csv_dt)

    def set_state(i):
        data.qpos[0:3] = [base_x[i], base_y[i], base_z[i]]
        if "base_qw" in df.columns:
            data.qpos[3:7] = [df["base_qw"].iloc[i], df["base_qx"].iloc[i],
                             df["base_qy"].iloc[i], df["base_qz"].iloc[i]]
        else: data.qpos[3:7] = [1, 0, 0, 0]
        data.qvel[0:3] = [base_vx[i], base_vy[i], base_vz[i]]
        data.qpos[7:13] = q_ref[i]
        data.qvel[6:12] = qd_ref[i]
        mujoco.mj_forward(model, data)

    # Logging structures
    log = []

    def step_logic(i):
        if i < warmup_steps: set_state(i)

        q_curr = data.qpos[7:13].copy()
        qd_curr = data.qvel[6:12].copy()

        # PD effort calculation
        pd_effort = kp * (q_ref[i] - q_curr) + kd * (qd_ref[i] - qd_curr)
        data.ctrl[:] = tau_ff[i] + pd_effort

        # Capture state before stepping (for logging)
        fz_L, fz_R = compute_grf(model, data)

        step_log = {
            "time": i * csv_dt,
            "q_err": np.abs(q_ref[i] - q_curr),
            "qd_err": np.abs(qd_ref[i] - qd_curr),
            "tau_actual": data.ctrl[:].copy(),
            "pd_effort": np.abs(pd_effort),
            "grf_actual": [fz_L, fz_R],
            "grf_err": np.abs(grf_ref[i] - [fz_L, fz_R])
        }
        log.append(step_log)

        for _ in range(frame_skip):
            mujoco.mj_step(model, data)

    reset_to_start = lambda: [mujoco.mj_resetData(model, data), set_state(0), log.clear()]
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

                step_logic(i)
                viewer.sync()

                target_wall = i * csv_dt
                elapsed = _time.perf_counter() - wall_start
                if target_wall > elapsed: _time.sleep(target_wall - elapsed)
                step_idx += 1
    else:
        for i in range(n_steps): step_logic(i)

    # --- Post-Analysis ---
    q_errs = np.array([l["q_err"] for l in log])
    qd_errs = np.array([l["qd_err"] for l in log])
    pd_efforts = np.array([l["pd_effort"] for l in log])
    grf_errs = np.array([l["grf_err"] for l in log])

    q_rms = np.sqrt(np.mean(q_errs**2, axis=0))
    total_pd_work = np.sum(pd_efforts) * csv_dt

    print("\n" + "="*40)
    print("GAIT REPRODUCTION REPORT")
    print("="*40)
    for idx, name in enumerate(JOINT_NAMES):
        print(f"{name:8s} | RMS Q Err: {q_rms[idx]:.5f} rad")

    print("-" * 40)
    print(f"Total Integrated PD Effort: {total_pd_work:.3f} Nms")
    print(f"Mean GRF Error: {np.mean(grf_errs):.2f} N")

    overall_rms = np.sqrt(np.mean(q_rms**2))
    print(f"Overall Joint RMS Error: {overall_rms:.5f} rad")

    if overall_rms < threshold:
        print("\nSUCCESS: Gait validated in Earth gravity (below threshold).")
    else:
        print("\nFAILURE: Tracking error exceeds threshold.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--xml", default="PennProposal/src/models/bolt_bipedal2.xml")
    p.add_argument("--csv", default="PennProposal/plots/bolt_simplified_Bolt-v3/run_1_20260304-120853/timeseries.csv")
    p.add_argument("--kp", type=float, default=50.0)
    p.add_argument("--kd", type=float, default=2.0)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--threshold", type=float, default=0.1, help="RMS threshold for success")
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--no-loop", action="store_true")
    args = p.parse_args()

    reproduce(args.xml, args.csv, kp=args.kp, kd=args.kd, warmup_steps=args.warmup,
              threshold=args.threshold, render=not args.no_render, loop=not args.no_loop)
