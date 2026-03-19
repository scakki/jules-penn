# render_bolt_best.py
import os
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")  # NON-GUI backend
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import wandb
import csv
import imageio
import pickle


from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ============================================================
# Config
# ============================================================
ENV = "bolt_simplified:Bolt-v3"
ENV_ID = f"src.envs.{ENV}"
BASE_LOG_DIR = f"build/traininglogs/{ENV}"
N_STEPS = 1000
MAX_TIME = 20.0  # seconds

def save(fig, plots_dir, name):
    fig.tight_layout()

    fig.savefig(os.path.join(plots_dir, f"{name}.png"), dpi=300)

    with open(os.path.join(plots_dir, f"{name}.fig.pkl"), "wb") as f:
        pickle.dump(fig, f)

    plt.close(fig)


# ============================================================
# Utilities
# ============================================================
def list_runs(base_dir):
    runs = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("run_")
    ]

    def run_key(name):
        try:
            return int(name.split("_")[1])  # run_XXX
        except Exception:
            return -1

    runs.sort(key=run_key)
    return runs

def get_unique_render_dir(base_dir, run_folder):
    """
    Returns a unique directory path.
    If base_dir/run_folder exists, appends __render2, __render3, ...
    """
    root = os.path.join(base_dir, run_folder)

    if not os.path.exists(root):
        return root, run_folder

    idx = 2
    while True:
        candidate_name = f"{run_folder}__render{idx}"
        candidate_path = os.path.join(base_dir, candidate_name)
        if not os.path.exists(candidate_path):
            return candidate_path, candidate_name
        idx += 1

def pick_runs(base_dir):
    runs = list_runs(base_dir)
    if not runs:
        raise FileNotFoundError("No runs found")

    print("\nAvailable runs:")
    for r in runs:
        print(r)

    print("\nSelect run:")
    print("  - Single: 55")
    print("  - Range : 55-60")
    print("  - Enter : latest\n")

    choice = input("Selection: ").strip()

    if choice == "":
        return [runs[-1]]

    def parse_range(s):
        if "-" in s:
            a, b = s.split("-")
            return list(range(int(a), int(b) + 1))
        else:
            return [int(s)]

    requested = parse_range(choice)

    selected = []
    for rn in requested:
        prefix = f"run_{rn}_"
        match = next((r for r in runs if r.startswith(prefix)), None)
        if match is None:
            print(f"[WARN] run_{rn} not found — skipping")
            continue
        selected.append(match)

    if not selected:
        raise RuntimeError("No valid runs selected")

    return selected



# ============================================================
# Core render + plot
# ============================================================
def render_best(run_folder):
    LOG_DIR = os.path.join(BASE_LOG_DIR, run_folder)
    MODEL_PATH = os.path.join(LOG_DIR, "models", "best_model.zip")
    VECNORM_PATH = os.path.join(LOG_DIR, "vecnorm.pkl")

    assert os.path.exists(MODEL_PATH), f"Missing {MODEL_PATH}"
    assert os.path.exists(VECNORM_PATH), f"Missing {VECNORM_PATH}"

    plots_root = os.path.join("plots", ENV.replace(":", "_"))
    os.makedirs(plots_root, exist_ok=True)

    plots_dir, render_tag = get_unique_render_dir(plots_root, run_folder)
    os.makedirs(plots_dir)
    #Add video
    
    video_path = os.path.join(plots_dir, "simulation.mp4")
    video_writer = imageio.get_writer(
        video_path,
        fps=50,
        codec="libx264",
        quality=8,
    )


    # ---------------- W&B ----------------
    wandb.init(
        project="PennProposalBoltRender",
        name=f"{render_tag}_BEST",
        config={
            "env": ENV,
            "run_folder": run_folder,
            "render_tag": render_tag,
            "checkpoint": "best_model.zip",
        },
        reinit=True,
    )

    # ---------------- Environment ----------------
    def make_env():
        return gym.make(
            ENV_ID,
            render_mode="rgb_array",
            exclude_current_positions_from_observation=True,
        )
        #return gym.wrappers.TimeLimit(env, N_STEPS)

    venv = DummyVecEnv([make_env])
    venv = VecNormalize.load(VECNORM_PATH, venv)
    venv.training = False
    venv.norm_reward = False

    policy = PPO.load(MODEL_PATH)
    obs = venv.reset()

    env = venv.envs[0].unwrapped
    model, data = env.model, env.data

    print("\n--- GEOMS (id -> name) ---")
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name is not None:
            print(i, name)
    # env.mujoco_renderer.cam.fixedcamid = mujoco.mj_name2id(
    #         env.model, mujoco.mjtObj.mjOBJ_CAMERA, "track"
    #     )
    # env.mujoco_renderer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

    # ---------------- Joint & actuator IDs ----------------
    jid = {
        "L_HAA": env._jid_FL_HAA,
        "L_HFE": env._jid_FL_HFE,
        "L_KFE": env._jid_FL_KFE,
        "R_HAA": env._jid_FR_HAA,
        "R_HFE": env._jid_FR_HFE,
        "R_KFE": env._jid_FR_KFE,
    }

    aid = {
        "L_HAA": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "FL_SHOULDER"),
        "L_HFE": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "FL_UPPER_LEG"),
        "L_KFE": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "FL_LOWER_LEG"),
        "R_HAA": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "FR_SHOULDER"),
        "R_HFE": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "FR_UPPER_LEG"),
        "R_KFE": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "FR_LOWER_LEG"),
    }

    # ---------------- Logs ----------------
    t = []
    reward_step, reward_cum = [], []
    speed, base_x, base_y, base_h = [], [], [], []
    tangent_L_list = []
    tangent_R_list = []
    mu_L_list = []
    mu_R_list = []

    geom_FL = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "FL_foot_col")
    geom_FR = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "FR_foot_col")
    geom_ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    print("geom_FL:", geom_FL)
    print("geom_FR:", geom_FR)
    print("geom_ground:", geom_ground)

    contact = {"Left": [], "Right": []}
    force   = {"Left": [], "Right": []}

    q   = {k: [] for k in jid}
    qd  = {k: [] for k in jid}
    tau = {k: [] for k in jid}

    ep_return = 0.0
    dt = model.opt.timestep
    total_energy = 0.0

    def compute_grf_per_foot(model, data, geom_foot, geom_ground):
        """
        Returns:
            F_normal_mag
            F_tangent_mag
        """

        F_normal = 0.0
        F_tangent = 0.0

        for i in range(data.ncon):
            contact = data.contact[i]

            if (
                (contact.geom1 == geom_foot and contact.geom2 == geom_ground) or
                (contact.geom2 == geom_foot and contact.geom1 == geom_ground)
            ):

                force_contact = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, force_contact)

                fn = force_contact[0]          # contact normal
                ft1 = force_contact[1]
                ft2 = force_contact[2]

                F_normal += abs(fn)
                F_tangent += np.sqrt(ft1**2 + ft2**2)

        return F_normal, F_tangent
    
    def geom_in_contact(model, data, geom_foot, geom_ground):
        for i in range(data.ncon):
            c = data.contact[i]
            if (
                (c.geom1 == geom_foot and c.geom2 == geom_ground) or
                (c.geom2 == geom_foot and c.geom1 == geom_ground)
            ):
                return True
        return False

    # ---------------- Rollout ----------------
    # for _ in range(N_STEPS):
    #     action, _ = policy.predict(obs, deterministic=True)
    #     obs, rew, done, _ = venv.step(action)
    #     # venv.envs[0].render()
    #
    #     if done[0]:
    #         break
    #     frame = venv.envs[0].render()
    #
    #     video_writer.append_data(frame)
    
    while True:
        action, _ = policy.predict(obs, deterministic=True)
        obs, rew, done, _ = venv.step(action)

        frame = venv.envs[0].render()
        video_writer.append_data(frame)

        
        
        r = float(rew[0])
        ep_return += r
        if done[0] or float(data.time) >= MAX_TIME:
            break
        time_s = float(data.time)
        vx, vy = float(data.qvel[0]), float(data.qvel[1])

        t.append(time_s)
        reward_step.append(r)
        reward_cum.append(ep_return)
        speed.append(np.hypot(vx, vy))
        base_x.append(float(data.qpos[0]))
        base_y.append(float(data.qpos[1]))
        base_h.append(float(data.qpos[2]))

        # --- Contacts ---
        # cL, cR, fL, fR = env._foot_contacts()
        # contact["Left"].append(int(cL))
        # contact["Right"].append(int(cR))



        # --- GRF magnitudes in contact frame ---
        Fn_L, Ft_L = compute_grf_per_foot(model, data, geom_FL, geom_ground)
        Fn_R, Ft_R = compute_grf_per_foot(model, data, geom_FR, geom_ground)

        force["Left"].append(Fn_L)
        force["Right"].append(Fn_R)

        tangent_L_list.append(Ft_L)
        tangent_R_list.append(Ft_R)

        eps_mu = 1e-6
        mu_L_list.append(Ft_L / (Fn_L + eps_mu))
        mu_R_list.append(Ft_R / (Fn_R + eps_mu))

        cL = geom_in_contact(model, data, geom_FL, geom_ground)
        cR = geom_in_contact(model, data, geom_FR, geom_ground)

        contact["Left"].append(int(cL))
        contact["Right"].append(int(cR))
        # --- Joint states ---
        for k, j in jid.items():
            q[k].append(float(data.qpos[env._qadr[j]]))
            qd[k].append(float(data.qvel[model.jnt_dofadr[j]]))
            tau[k].append(float(data.actuator_force[aid[k]]))

        # --- Mechanical power ---
        step_power = 0.0
        for j in ["L_HAA", "L_HFE", "L_KFE", "R_HAA", "R_HFE", "R_KFE"]:
            step_power += abs(tau[j][-1] * qd[j][-1])

        total_energy += step_power * dt

        wandb.log({
            "time": time_s,
            "reward_step": r,
            "reward_cumulative": ep_return,
            "speed": speed[-1],
            "base_height": base_h[-1],
        })

        
    video_writer.close()   
    venv.close()

    # =========================
    # Stance-averaged forces
    # =========================
    cL = np.asarray(contact["Left"], dtype=np.float64)   # 0/1
    cR = np.asarray(contact["Right"], dtype=np.float64)

    Fz_L = np.asarray(force["Left"], dtype=np.float64)  # you stored normal here
    Fz_R = np.asarray(force["Right"], dtype=np.float64)

    Ft_L = np.asarray(tangent_L_list, dtype=np.float64)
    Ft_R = np.asarray(tangent_R_list, dtype=np.float64)

    stance_L = float(np.sum(cL))
    stance_R = float(np.sum(cR))
    eps = 1e-8

    mean_Fz_L_stance = float(np.sum(Fz_L * cL) / (stance_L + eps))
    mean_Fz_R_stance = float(np.sum(Fz_R * cR) / (stance_R + eps))

    mean_Ft_L_stance = float(np.sum(Ft_L * cL) / (stance_L + eps))
    mean_Ft_R_stance = float(np.sum(Ft_R * cR) / (stance_R + eps))

    # =============================
    # Peak forces (stance only)
    # =============================

    Fz_L_stance = Fz_L * cL
    Fz_R_stance = Fz_R * cR

    Ft_L_stance = Ft_L * cL
    Ft_R_stance = Ft_R * cR

    max_Fz_L = float(np.max(Fz_L_stance)) if len(Fz_L_stance) else 0.0
    max_Fz_R = float(np.max(Fz_R_stance)) if len(Fz_R_stance) else 0.0

    max_Ft_L = float(np.max(Ft_L_stance)) if len(Ft_L_stance) else 0.0
    max_Ft_R = float(np.max(Ft_R_stance)) if len(Ft_R_stance) else 0.0

    mu_L = np.asarray(mu_L_list)
    mu_R = np.asarray(mu_R_list)

    max_mu_L = float(np.max(mu_L * cL)) if len(mu_L) else 0.0
    max_mu_R = float(np.max(mu_R * cR)) if len(mu_R) else 0.0

    # (Optional) stance ratios can be useful too
    left_contact_ratio  = float(np.mean(cL)) if len(cL) else 0.0
    right_contact_ratio = float(np.mean(cR)) if len(cR) else 0.0

    # =============================
    # Physics sanity check
    # =============================
    m = float(np.sum(model.body_mass))
    gmag = float(abs(model.opt.gravity[2]))
    BW = m * gmag

    Fz_total_timeavg = float(np.mean(cL * Fz_L + cR * Fz_R))

    print("\n==== FORCE SANITY CHECK ====")
    print(f"Mass: {m:.3f} kg")
    print(f"Bodyweight: {BW:.2f} N")
    print(f"Mean total normal force: {Fz_total_timeavg:.2f} N")
    print("============================\n")
    # --- Cost of Transport ---
    mass = np.sum(model.body_mass)
    g = abs(model.opt.gravity[2])

    delta_x = base_x[-1] - base_x[0] if len(base_x) > 1 else 0.0

    cot = (
        total_energy / (mass * g * delta_x)
        if delta_x > 1e-6 else np.nan
    )
    wandb.log({
        "episode/return": ep_return,
        "episode/steps": len(t),
        "episode/time": t[-1] if t else 0.0,
        "episode/CoT": cot,
        "episode/energy_total": total_energy,
        "episode/delta_x": delta_x,
        "stance/mean_Fz_L": mean_Fz_L_stance,
        "stance/mean_Fz_R": mean_Fz_R_stance,
        "stance/mean_Ft_L": mean_Ft_L_stance,
        "stance/mean_Ft_R": mean_Ft_R_stance,
        "stance/contact_ratio_L": left_contact_ratio,
        "stance/contact_ratio_R": right_contact_ratio,
    })
    wandb.finish()

    timeseries_csv = os.path.join(plots_dir, "timeseries.csv")

    with open(timeseries_csv, "w", newline="") as f:
        writer = csv.writer(f)

        header = [
            "time",
            "reward",
            "reward_cumulative",
            "speed",
            "base_x",
            "base_y",
            "base_height",
            "left_contact",
            "right_contact",
            "left_Fz",
            "right_Fz",
            "left_Ft",
            "right_Ft",
            "left_mu",
            "right_mu",

        ]

        for j in ["L_HAA", "L_HFE", "L_KFE", "R_HAA", "R_HFE", "R_KFE"]:
            header += [f"{j}_q", f"{j}_qd", f"{j}_tau"]

        writer.writerow(header)

        for i in range(len(t)):
            row = [
                t[i],
                reward_step[i],
                reward_cum[i],
                speed[i],
                base_x[i],
                base_y[i],
                base_h[i],
                contact["Left"][i],
                contact["Right"][i],
                force["Left"][i],
                force["Right"][i],
                tangent_L_list[i],
                tangent_R_list[i],
                mu_L_list[i],
                mu_R_list[i],
            ]

            for j in ["L_HAA", "L_HFE", "L_KFE", "R_HAA", "R_HFE", "R_KFE"]:
                row += [q[j][i], qd[j][i], tau[j][i]]

            writer.writerow(row)
    SUMMARY_CSV = os.path.join("plots", ENV.replace(":", "_"), "render_summary.csv")
    summary_exists = os.path.exists(SUMMARY_CSV)

    with open(SUMMARY_CSV, "a", newline="") as f:
        writer = csv.writer(f)

        if not summary_exists:
            writer.writerow([
                "run_folder",
                "render_tag",
                "episode_return",
                "episode_steps",
                "episode_time",
                "mean_speed",
                "max_speed",
                "mean_base_height",
                "left_contact_ratio",
                "right_contact_ratio",
                "mean_Fz_L_stance",
                "mean_Fz_R_stance",
                "mean_Ft_L_stance",
                "mean_Ft_R_stance",
                "max_Fz_L",
                "max_Fz_R",
                "max_Ft_L",
                "max_Ft_R",
                "max_mu_L",
                "max_mu_R",
                "total_energy",
                "delta_x",
                "CoT",
            ])

        writer.writerow([
            run_folder,
            render_tag,
            ep_return,
            len(t),
            t[-1] if t else 0.0,
            float(np.mean(speed)) if speed else 0.0,
            float(np.max(speed))  if speed else 0.0,
            float(np.mean(base_h)),
            left_contact_ratio,
            right_contact_ratio,
            mean_Fz_L_stance,
            mean_Fz_R_stance,
            mean_Ft_L_stance,
            mean_Ft_R_stance,
            max_Fz_L,
            max_Fz_R,
            max_Ft_L,
            max_Ft_R,
            max_mu_L,
            max_mu_R,
            total_energy,
            delta_x,
            cot,
        ])        

    # ============================================================
    # Plot helpers
    # ============================================================
    # def save(fig, name):
    #     fig.tight_layout()
    #     fig.savefig(os.path.join(plots_dir, f"{name}.png"), dpi=300)
    #     plt.close(fig)

    # ---------------- Reward ----------------
    fig, ax = plt.subplots()
    ax.plot(t, reward_step)
    ax.set(title="Reward per Timestep", xlabel="Time (s)", ylabel="Reward")
    ax.grid(True)
    save(fig, plots_dir, "reward_step")


    fig, ax = plt.subplots()
    ax.plot(t, reward_cum)
    ax.set(title="Cumulative Reward", xlabel="Time (s)", ylabel="Return")
    ax.grid(True)
    save(fig, plots_dir, "reward_cumulative")


    # ---------------- Base ----------------
    for y, title, name in [
        (speed, "Base Speed", "base_speed"),
        (base_h, "Base Height", "base_height"),
    ]:
        fig, ax = plt.subplots()
        ax.plot(t, y)
        ax.set(title=title, xlabel="Time (s)")
        ax.grid(True)
        save(fig, plots_dir, name)

    fig, ax = plt.subplots()
    ax.plot(base_x, base_y)
    ax.set(title="Base XY Trajectory", xlabel="X", ylabel="Y")
    ax.axis("equal")
    ax.grid(True)
    save(fig, plots_dir, "base_xy")

    # ---------------- Contacts ----------------
    fig, ax = plt.subplots()
    ax.step(t, contact["Left"], where="post")
    ax.set(title="Left Foot Contact", xlabel="Time (s)", ylabel="0/1")
    ax.grid(True)
    save(fig, plots_dir, "left_contact")

    fig, ax = plt.subplots()
    ax.step(t, contact["Right"], where="post")
    ax.set(title="Right Foot Contact", xlabel="Time (s)", ylabel="0/1")
    ax.grid(True)
    save(fig, plots_dir, "right_contact")

    fig, ax = plt.subplots()
    ax.step(t, contact["Left"], where="post", label="Left")
    ax.step(t, contact["Right"], where="post", linestyle="--", label="Right")
    ax.plot(t, force["Left"], label="Left GRF")
    ax.plot(t, force["Right"], linestyle="--", label="Right GRF")
    ax.legend()
    ax.grid(True)
    save(fig, plots_dir, "contact_force_symmetry")

    # ---------------- Joint Angles ----------------
    for side, side_name in [("L", "Left"), ("R", "Right")]:
        fig, ax = plt.subplots()
        for j in ["HAA", "HFE", "KFE"]:
            ax.plot(t, q[f"{side}_{j}"], label=j)
        ax.legend()
        ax.set(title=f"{side_name} Joint Angles", xlabel="Time (s)", ylabel="rad")
        ax.grid(True)
        save(fig, plots_dir, f"{side_name.lower()}_joint_angles")

    fig, ax = plt.subplots()
    for j in ["HAA", "HFE", "KFE"]:
        ax.plot(t, q[f"L_{j}"], label=f"Left {j}")
        ax.plot(t, q[f"R_{j}"], "--", label=f"Right {j}")
    ax.legend(ncol=2)
    ax.grid(True)
    save(fig, plots_dir, "joint_angles_symmetry")

    # ---------------- Joint Velocities ----------------
    fig, ax = plt.subplots()
    for j in ["HAA", "HFE", "KFE"]:
        ax.plot(t, qd[f"L_{j}"], label=f"Left {j}")
        ax.plot(t, qd[f"R_{j}"], "--", label=f"Right {j}")
    ax.legend(ncol=2)
    ax.grid(True)
    save(fig, plots_dir, "joint_velocities")

    # ---------------- Joint Torques ----------------
    fig, ax = plt.subplots()
    for j in ["HAA", "HFE", "KFE"]:
        ax.plot(t, tau[f"L_{j}"], label=f"Left {j}")
        ax.plot(t, tau[f"R_{j}"], "--", label=f"Right {j}")
    ax.legend(ncol=2)
    ax.grid(True)
    save(fig, plots_dir, "joint_torques")


# ============================================================
# Main
# ============================================================
def main():
    run_folders = pick_runs(BASE_LOG_DIR)

    print("\nRuns to render:")
    for r in run_folders:
        print(" ", r)

    for r in run_folders:
        print("\n" + "=" * 60)
        print(f"Rendering BEST model from:\n  {r}")
        print("=" * 60)

        render_best(r)

    print("\n All requested runs rendered.")


if __name__ == "__main__":
    main()
