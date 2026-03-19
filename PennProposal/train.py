# train.py
import os, re, time
import gymnasium as gym
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed
from src.envs import *

ENV          = 'bolt_simplified:Bolt-v3'  # choose environment here
ENV_ID       = f"src.envs.{ENV}" # [for walk - src.envs.bolt_walk:Bolt-v2]
SEED         = 123
N_ENVS       = int(os.getenv("N_ENVS", "4"))      # parallel envs
MAX_STEPS    = int(os.getenv("MAX_STEPS", "1000"))# steps per episode (~10s at 100 Hz)
BASE_LOG_DIR = f"build/traininglogs/{ENV}"
TOTAL_STEPS  = int(os.getenv("TOTAL_STEPS", "5000000"))
DEVICE       = os.getenv("DEVICE", "cuda")        # set to "cpu" if needed

def make_env(rank, log_dir):
    def _thunk():
        runlogs = os.path.join(log_dir, "runlogs")
        worker_log_dir = os.path.join(runlogs, f"w{rank}")
        os.makedirs(worker_log_dir, exist_ok=True)

        env = gym.make(ENV_ID, log_root=worker_log_dir)#, g_min=1.62,g_max=9.81,)#, render_mode="human")                      # no render during training
        env = gym.wrappers.TimeLimit(env, MAX_STEPS)
        
        worker_monitor_dir = os.path.join(runlogs, f"monitor_w{rank}")
        os.makedirs(worker_monitor_dir, exist_ok=True)
        env = Monitor(env, worker_monitor_dir)
        env.reset(seed=SEED + rank)
        return env
    return _thunk


def main():
    os.makedirs(BASE_LOG_DIR, exist_ok=True)
    existing = [d for d in os.listdir(BASE_LOG_DIR) if os.path.isdir(os.path.join(BASE_LOG_DIR, d))]
    run_numbers = [int(m.group(1)) for d in existing if (m := re.match(r"run_(\d+)_", d))]
    run_name = f"run_{(max(run_numbers, default=0)+1)}_{time.strftime('%Y%m%d-%H%M%S')}"
    log_dir  = os.path.join(BASE_LOG_DIR, run_name)
    ckpt_dir = os.path.join(log_dir, "models")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Logging to: {log_dir}")

    wandb.init(project="PennProposalBolt", name=run_name, sync_tensorboard=True, save_code=True)
    set_random_seed(SEED)

    # ----- training envs -----
    venv = SubprocVecEnv([make_env(i, log_dir) for i in range(N_ENVS)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    def make_eval_env():
        eval_log_dir = os.path.join(log_dir, "runlogs", "eval")
        os.makedirs(eval_log_dir, exist_ok=True)
        e = gym.make(ENV_ID, log_root=eval_log_dir)#, g_min=1.62, g_max=9.81,)
        e = gym.wrappers.TimeLimit(e, MAX_STEPS)
        return e

    # ----- eval env (vectorized + share obs stats) -----
    eval_vec = DummyVecEnv([make_eval_env])
    eval_vec = VecNormalize(eval_vec, training=False, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_vec.obs_rms = venv.obs_rms

    checkpoint_cb = CheckpointCallback(save_freq=max(1, 200_000 // N_ENVS),
                                       save_path=ckpt_dir, name_prefix="bolt_ppo")
    eval_cb = EvalCallback(eval_vec, best_model_save_path=ckpt_dir, log_path=log_dir,
                           eval_freq=max(1, 50_000 // N_ENVS), n_eval_episodes=5, deterministic=True)

    model = PPO(
        "MlpPolicy",
        venv,
        n_steps=2048, batch_size=4096, n_epochs=10,
        learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.0, vf_coef=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),
        seed=SEED, verbose=1, tensorboard_log=log_dir, device=DEVICE,
    )

    model.learn(
        total_timesteps=TOTAL_STEPS,
        callback=[checkpoint_cb, eval_cb,
                  WandbCallback(gradient_save_freq=100, model_save_path=ckpt_dir, verbose=2)]
    )

    venv.save(os.path.join(log_dir, "vecnorm.pkl"))
    model.save(os.path.join(log_dir, "ppo_bolt_final"))
    wandb.finish()

if __name__ == "__main__":
    import multiprocessing as mp
    # set env vars BEFORE starting workers
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("WANDB_START_METHOD", "thread")
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
