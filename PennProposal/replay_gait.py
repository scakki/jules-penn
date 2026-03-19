"""
replay_gait.py — Replay a trained Bolt bipedal gait in MuJoCo from CSV timeseries.

Two modes:
  --mode position  (default)  Reconstruct PD targets from recorded tau and q:
                              ctrl = tau/kp + q, then let the PD actuator reproduce forces.
  --mode torque               Swap actuators to motors and apply recorded torques directly.

Usage:
  python replay_gait.py --xml bolt_bipedal2.xml --csv timeseries.csv [--mode position] [--speed 1.0] [--no-render]
"""

import argparse, sys, tempfile, os
import numpy as np
import pandas as pd
import mujoco
import mujoco.viewer

# ── CSV column names → actuator order ──────────────────────────────────────
#  Actuator order in XML:  FL_HAA, FL_HFE, FL_KFE, FR_HAA, FR_HFE, FR_KFE
Q_COLS   = ["L_HAA_q",   "L_HFE_q",   "L_KFE_q",   "R_HAA_q",   "R_HFE_q",   "R_KFE_q"]
TAU_COLS = ["L_HAA_tau", "L_HFE_tau", "L_KFE_tau", "R_HAA_tau", "R_HFE_tau", "R_KFE_tau"]
QD_COLS  = ["L_HAA_qd",  "L_HFE_qd",  "L_KFE_qd",  "R_HAA_qd",  "R_HFE_qd",  "R_KFE_qd"]


def make_torque_xml(xml_path: str) -> str:
    """Rewrite the XML: replace <position …/> actuators with <motor …/> actuators
    so we can apply raw torques.  Returns path to a temp XML file."""
    with open(xml_path) as f:
        txt = f.read()

    # Replace the default actuator class from position to motor
    # and remove kp (not used for motors)
    import re

    # Replace the top-level default <position .../> with <motor .../>
    txt = re.sub(
        r'<position\s+kp="[^"]*"\s+forcelimited="true"\s+ctrllimited="true"\s+forcerange="([^"]*)"/>',
        r'<motor forcelimited="true" ctrllimited="true" forcerange="\1"/>',
        txt,
    )
    # Replace each actuator line:  <position class=...  → <motor class=...
    txt = re.sub(r"<position(\s+class)", r"<motor\1", txt)

    # For motor actuators, ctrlrange should match forcerange, not joint range.
    # Update each class default to set ctrlrange = forcerange
    for cls in ("hip_abd_limited", "hip", "thigh", "calf"):
        pattern = rf'(<default\s+class="{cls}">\s*<joint[^/]*/>\s*)<position\s+ctrllimited="true"\s+ctrlrange="[^"]*"/>'
        replacement = rf'\1<motor ctrllimited="true" ctrlrange="-33.5 33.5"/>'
        txt = re.sub(pattern, replacement, txt)

    tmp = tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode="w")
    tmp.write(txt)
    tmp.close()
    return tmp.name


def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Strip whitespace from column names (Windows line endings etc.)
    df.columns = df.columns.str.strip()
    return df


def get_initial_qpos(model, df_row):
    """Build a full qpos from the first CSV row so the robot starts in the
    right configuration.  qpos layout for a free-joint body:
       [x, y, z, qw, qx, qy, qz,  FL_HAA, FL_HFE, FL_KFE, FR_HAA, FR_HFE, FR_KFE]
    """
    qpos = model.key_qpos[0].copy() if model.nkey > 0 else np.zeros(model.nq)

    # Free joint: set position from CSV
    qpos[0] = df_row.get("base_x", 0.0)
    qpos[1] = df_row.get("base_y", 0.0)
    qpos[2] = df_row.get("base_height", 0.474)
    # Quaternion: upright
    qpos[3] = 1.0; qpos[4] = 0.0; qpos[5] = 0.0; qpos[6] = 0.0

    # Joint angles (indices 7-12)
    for i, col in enumerate(Q_COLS):
        qpos[7 + i] = df_row[col]

    return qpos


def get_initial_qvel(model, df_row):
    """Build initial qvel.  qvel layout:
       [vx, vy, vz, wx, wy, wz,  FL_HAA_qd, FL_HFE_qd, FL_KFE_qd, FR_HAA_qd, FR_HFE_qd, FR_KFE_qd]
    """
    qvel = np.zeros(model.nv)
    qvel[0] = df_row.get("speed", 0.0)  # approximate vx from speed column
    for i, col in enumerate(QD_COLS):
        qvel[6 + i] = df_row[col]
    return qvel


def replay(xml_path, csv_path, mode="position", playback_speed=1.0, render=True,
           loop=True, save_video=None):
    df = load_csv(csv_path)
    dt_csv = float(df["time"].iloc[1] - df["time"].iloc[0])  # should be 0.01
    n_steps = len(df)

    print(f"CSV: {n_steps} steps, dt={dt_csv:.4f}s, duration={df['time'].iloc[-1]:.2f}s")
    print(f"Mode: {mode} | Playback speed: {playback_speed}x | Render: {render}")

    # ── load model ──
    if mode == "torque":
        tmp_xml = make_torque_xml(xml_path)
        model = mujoco.MjModel.from_xml_path(tmp_xml)
        os.unlink(tmp_xml)
        print("Loaded torque-actuator model (motors).")
    else:
        model = mujoco.MjModel.from_xml_path(xml_path)
        print("Loaded position-actuator model (PD control).")

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Verify timestep compatibility
    sim_dt = model.opt.timestep
    frame_skip = max(1, round(dt_csv / sim_dt))
    effective_dt = sim_dt * frame_skip
    print(f"Sim dt={sim_dt}, frame_skip={frame_skip}, effective control dt={effective_dt:.4f}s")

    # ── precompute control arrays ──
    q_array   = df[Q_COLS].values.astype(np.float64)    # (N, 6)
    tau_array = df[TAU_COLS].values.astype(np.float64)  # (N, 6)
    qd_array  = df[QD_COLS].values.astype(np.float64)   # (N, 6)

    if mode == "position":
        # CRITICAL: The CSV logs resulting joint positions, NOT the ctrl targets.
        # The PD actuator computes:  tau = kp * (ctrl - qpos)   (kv=0 by default)
        # So ctrl_target = tau/kp + qpos.
        # Feeding raw qpos as ctrl would give ~zero error → ~zero force → robot falls.
        kp = model.actuator_gainprm[0, 0]  # read kp from model (should be 50)
        print(f"Position actuator kp = {kp}")

        ctrl_array = tau_array / kp + q_array  # reconstruct PD targets

        # Clip to actuator ctrl ranges
        ctrl_low  = model.actuator_ctrlrange[:, 0]
        ctrl_high = model.actuator_ctrlrange[:, 1]
        ctrl_array = np.clip(ctrl_array, ctrl_low, ctrl_high)

        print(f"Reconstructed ctrl targets from tau/kp + q")
    else:
        ctrl_array = tau_array  # apply torques directly

    # ── set initial state ──
    def reset_to_start():
        mujoco.mj_resetData(model, data)
        row0 = df.iloc[0]
        data.qpos[:] = get_initial_qpos(model, row0)
        data.qvel[:] = get_initial_qvel(model, row0)
        mujoco.mj_forward(model, data)

    reset_to_start()

    # ── run ──
    if render:
        import time as _time

        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
            viewer.cam.distance = 2.0
            viewer.cam.elevation = -20

            step_idx = 0
            wall_start = _time.perf_counter()

            while viewer.is_running():
                row = step_idx % n_steps  # loop

                if row == 0 and step_idx > 0:
                    if not loop:
                        break
                    reset_to_start()
                    wall_start = _time.perf_counter()

                # Set control
                data.ctrl[:] = ctrl_array[row]

                # Step physics (frame_skip sub-steps)
                for _ in range(frame_skip):
                    mujoco.mj_step(model, data)

                viewer.sync()

                # Real-time pacing
                sim_time = (row + 1) * dt_csv
                wall_elapsed = _time.perf_counter() - wall_start
                target_wall = sim_time / playback_speed
                sleep = target_wall - wall_elapsed
                if sleep > 0:
                    _time.sleep(sleep)

                step_idx += 1

    else:
        # Headless — just run and print stats
        import time as _time
        t0 = _time.perf_counter()
        for row in range(n_steps):
            data.ctrl[:] = ctrl_array[row]
            for _ in range(frame_skip):
                mujoco.mj_step(model, data)

        wall = _time.perf_counter() - t0
        print(f"Headless replay done in {wall:.2f}s  (sim duration {df['time'].iloc[-1]:.2f}s)")
        print(f"Final position: x={data.qpos[0]:.3f}, y={data.qpos[1]:.3f}, z={data.qpos[2]:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay Bolt gait from CSV")
    parser.add_argument("--xml", default="bolt_bipedal2.xml", help="Path to MuJoCo XML")
    parser.add_argument("--csv", default="timeseries.csv", help="Path to timeseries CSV")
    parser.add_argument("--mode", choices=["position", "torque"], default="position",
                        help="'position' = feed q as PD targets (default), 'torque' = apply tau directly")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--no-render", action="store_true", help="Headless mode")
    parser.add_argument("--no-loop", action="store_true", help="Don't loop the replay")
    args = parser.parse_args()

    replay(args.xml, args.csv, mode=args.mode, playback_speed=args.speed,
           render=not args.no_render, loop=not args.no_loop)