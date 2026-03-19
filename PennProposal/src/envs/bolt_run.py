import numpy as np
import os
import mujoco
from gymnasium import utils
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class BoltEnvRun(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(
        self,
        forward_reward_weight=2.5,
        ctrl_cost_weight=0.1,
        reset_noise_scale=0.02, 
        exclude_current_positions_from_observation=True,
        alive_bonus=0.1,
        standing_penalty_weight=0.05,
        step_back_penalty_weight=1.0,
        min_height=0.2,
        low_height_steps=5,
        tilt_fail_threshold=0.65,
        energy_cost_weight=0.56,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            forward_reward_weight,
            ctrl_cost_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            alive_bonus,
            step_back_penalty_weight,
            min_height,
            standing_penalty_weight,
            low_height_steps,
            tilt_fail_threshold, # Added tilt fail threshold
            energy_cost_weight,
            **kwargs,
        )

        self._log_root = kwargs.pop("log_root", None)
        self._log_enabled = self._log_root is not None
        self._episode_id = 0
        if self._log_enabled:
            os.makedirs(self._log_root, exist_ok=True)


        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale
        self.alive_bonus = alive_bonus
        self.step_back_penalty_weight = step_back_penalty_weight
        self.min_height = min_height
        self.standing_penalty_weight = standing_penalty_weight
        self.low_height_steps = low_height_steps
        self.tilt_fail_threshold = tilt_fail_threshold # Added tilt fail threshold
        self.low_height_counter = 0
        self.slow_progress_counter = 0
        self.energy_cost_weight = energy_cost_weight
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        # [FL_HAA, FL_HFE, FL_KFE, FR_HAA, FR_HFE, FR_KFE]
        self.offset = np.array([0, 0.40, -0.80,  0, 0.40, -0.80], dtype=np.float64)
        # self.offset = np.array([+0.04, -0.40, +0.80,  -0.04, -0.40, +0.80], dtype=np.float64)
        # self.offset = np.array([+0.04, 0.40, -0.80,  -0.04, 0.40, -0.80], dtype=np.float64)

        

        # Observation space
        if exclude_current_positions_from_observation:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(24,), dtype=np.float64
            )
        else:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(25,), dtype=np.float64
            )

        current_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_path, "..", "models", "bolt_bipedal2.xml")

        MujocoEnv.__init__(
            self,
            model_path,
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        self.action_space = Box(low=-0.5, high=0.5, shape=(6,), dtype=np.float32)
        ####################### Added this ##########################
        # ---- Look up things once (ids & indices) ----
        self._torso_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        # ---- Foot handles (prefer sites) ----
        # Touch sensors (by name)
        self._sid_FL = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "FL_touch")
        self._sid_FR = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "FR_touch")

        # Addresses into sensordata (in doubles)
        self._sadr_FL = self.model.sensor_adr[self._sid_FL] if self._sid_FL != -1 else None
        self._sadr_FR = self.model.sensor_adr[self._sid_FR] if self._sid_FR != -1 else None
        self._sid_FLfoot = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "FL_foot_site")
        self._sid_FRfoot = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "FR_foot_site")

        # Optional fallback to body world positions if sites are renamed later
        self._bid_FLbody = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "FL_LOWER_LEG")
        self._bid_FRbody = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "FR_LOWER_LEG")

        # __init__: cache joint indices for quick access
        self._jid_FL_HAA = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "FL_HAA")
        self._jid_FL_HFE = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "FL_HFE")
        self._jid_FL_KFE = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "FL_KFE")
        self._jid_FR_HAA = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "FR_HAA")
        self._jid_FR_HFE = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "FR_HFE")
        self._jid_FR_KFE = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "FR_KFE")
        self._qadr = self.model.jnt_qposadr.copy()
        
        
        self._t = 0.0
        self._warmup_sec = 0.35        
        self._target_ema = np.zeros(6, dtype=np.float64)
        self._ema_alpha = 0.6

        self._act_low = self.model.actuator_ctrlrange[:, 0].copy()
        self._act_high = self.model.actuator_ctrlrange[:, 1].copy()
        assert self._act_low.shape[0] == 6, "Expecting 6 actuators."

        if (self._sid_FLfoot == -1 or self._sid_FRfoot == -1) and \
        (self._bid_FLbody == -1 or self._bid_FRbody == -1):
            print("[BoltEnvRun] WARNING: could not resolve foot sites or bodies. "
          "Step-width/crossing penalties will be disabled.")

        # Joints you care about (order matters; matches your action order)
        self._joint_names = ["FL_HAA","FL_HFE","FL_KFE","FR_HAA","FR_HFE","FR_KFE"]
        self._joint_ids   = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                             for jn in self._joint_names]
        # DOF addresses (to index qvel/qfrc_actuator etc). Hinge joints have 1 DOF.
        self._dof_adrs = [int(self.model.jnt_dofadr[jid]) for jid in self._joint_ids]

        # Start empty logger
        # self._reset_logger()


    def _upright(self):
        # torso rotation matrix; world z alignment = R[2,2]
        R = self.data.xmat[self._torso_bid].reshape(3, 3)
        return float(R[2, 2])

    def _foot_contacts(self, threshold=1e-3):
        # Touch sensors output normal force magnitude
        fl = float(self.data.sensordata[self._sadr_FL]) if self._sadr_FL is not None else 0.0
        fr = float(self.data.sensordata[self._sadr_FR]) if self._sadr_FR is not None else 0.0
        return (fl > threshold), (fr > threshold), fl, fr

    def _foot_y_positions(self):
        """Return (yL, yR) in world coords using sites if available, else bodies."""
        if self._sid_FLfoot != -1 and self._sid_FRfoot != -1:
            yL = float(self.data.site_xpos[self._sid_FLfoot, 1])
            yR = float(self.data.site_xpos[self._sid_FRfoot, 1])
            return yL, yR
        if self._bid_FLbody != -1 and self._bid_FRbody != -1:
            # modern MuJoCo: xipos is body world position (not xpos)
            yL = float(self.data.xipos[self._bid_FLbody, 1])
            yR = float(self.data.xipos[self._bid_FRbody, 1])
            return yL, yR
        return None, None
    
        # --------- Logging helpers ---------
#     def _reset_logger(self):
#         # one list per column
#         self._ep = {
#             # time
#             "t": [],
#             # Pose (torso/world)
#             "x": [], "y": [], "z": [],
#             "roll": [], "pitch": [], "yaw": [], "upright": [],
#             # Joint state (angles & vels)
#             "q": [],          # shape (6,)
#             "qd": [],         # shape (6,)
#             # Actuation
#             "ctrl": [],       # applied command (after offset/smoothing)
#             "tau": [],        # joint torques from qfrc_actuator
#             "power": [],      # tau * qd
#             # Contacts (GRF world frame)
#             "FL_fx": [], "FL_fy": [], "FL_fz": [],
#             "FR_fx": [], "FR_fy": [], "FR_fz": [],
#             # Touch sensors
#             "FL_touch": [], "FR_touch": [],
#             # Feet positions
#             "FL_pos": [], "FR_pos": [],   # world xyz (3,)
#             # Gait / metrics
#             "step_width": [], "crossing_penalty": [], "stance_L": [], "stance_R": [],
#             # Task/episode signals
#             "reward": [], "forward_reward": [], "ctrl_cost": [],
#             "base_h": [], "terminated_reason": [],"energy_penalty": [],
# "P_abs": [],
#         }

    # def _save_episode(self):
    #     if not self._log_enabled or len(self._ep["t"]) == 0:
    #         return
    #     # Pack ragged lists to arrays where applicable
    #     def arr(x): return np.asarray(x, dtype=np.float32)
    #     out = {k: (np.vstack(v) if (len(v)>0 and isinstance(v[0], (list, tuple, np.ndarray)) and
    #                                 np.asarray(v[0]).ndim>0) else arr(v))
    #            for k, v in self._ep.items()}
    #     fname = os.path.join(self._log_root, f"episode_{self._episode_id:05d}.npz")
    #     np.savez_compressed(fname, **out)
    #     self._episode_id += 1


    def control_cost(self, action):
        return self._ctrl_cost_weight * float(np.sum(np.square(action)))#np.sum(np.square(action))
    
    

    ### Changing all the values to float for compatibility with newer mujoco versions ###
    def step(self, action):
        a_blend = np.asarray(action, np.float64)

        target_raw = a_blend + self.offset
        self._target_ema = self._ema_alpha * self._target_ema + (1.0 - self._ema_alpha) * target_raw
        target = np.clip(self._target_ema, self._act_low, self._act_high)
        # target = a_blend + self.offset
        # self._t += self.dt

        # --- positions before ---
        x_before = float(self.data.qpos[0])
        y_before = float(self.data.qpos[1])

        self.do_simulation(target, self.frame_skip)
        self._t += self.dt

        # --- velocities after ---
        x_after = float(self.data.qpos[0])
        y_after = float(self.data.qpos[1])
        vx = (x_after - x_before) / self.dt
        vy = (y_after - y_before) / self.dt

        # --- torso orientation & heading alignment ---
        R = self.data.xmat[self._torso_bid].reshape(3, 3)
        upright = float(R[2, 2])           # 1.0 = upright
        heading_align = float(R[0, 0])     # body x · world x = cos(yaw_error) in [-1,1]
        heading_gate = max(0.0, heading_align) ** 1.5  # gamma=1.5; tune 1.0..2.0

        yL, yR = self._foot_y_positions()
        if yL is None or yR is None:
            step_width = 0.0
            stepwidth_penalty = 0.0
            crossing_penalty = 0.0
        else:
            step_width = abs(yL - yR)
            w_min = 0.10
            k_sw = 2.0
            k_cross = 4.0
            stepwidth_penalty = k_sw * max(0.0, w_min - step_width)
            crossing_penalty  = k_cross * max(0.0, yR - yL)   # correct sign for +y = left
        if not hasattr(self, "_sw_dbg"):
            self._sw_dbg = 0
        if self._sw_dbg < 8 and yL is not None:
            #print(f"yL={yL:.3f}, yR={yR:.3f}, stepW={step_width:.3f}, crossP={crossing_penalty:.3f}")
            self._sw_dbg += 1

        # --- yaw rate (optional, tiny damping) ---
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        if not hasattr(self, "_prev_yaw"):
            self._prev_yaw = yaw
        dyaw = yaw - self._prev_yaw
        if dyaw > np.pi:  dyaw -= 2*np.pi
        if dyaw < -np.pi: dyaw += 2*np.pi
        yaw_rate = dyaw / self.dt
        self._prev_yaw = yaw

        qpos = self.data.qpos.flat  # contains free joint + hinges

        def get_q(jid, default=0.0):
            return float(qpos[self._qadr[jid]]) if jid != -1 else default

        # nominal angles (radians): tiny abduction, modest hip flexion, slight knee flexion
        q_nom = {
            "FL_HAA": +0.05, "FR_HAA": -0.05,
            "FL_HFE": -0.35, "FR_HFE": -0.35,
            "FL_KFE": +0.25, "FR_KFE": +0.25,
        }
        w_post = {"HAA": 1.0, "HFE": 0.6, "KFE": 0.6}  # weights per joint group (tune small)

        posture_penalty = 0.0
        posture_penalty += w_post["HAA"] * (get_q(self._jid_FL_HAA)-q_nom["FL_HAA"])**2
        posture_penalty += w_post["HAA"] * (get_q(self._jid_FR_HAA)-q_nom["FR_HAA"])**2
        posture_penalty += w_post["HFE"] * (get_q(self._jid_FL_HFE)-q_nom["FL_HFE"])**2
        posture_penalty += w_post["HFE"] * (get_q(self._jid_FR_HFE)-q_nom["FR_HFE"])**2
        posture_penalty += w_post["KFE"] * (get_q(self._jid_FL_KFE)-q_nom["FL_KFE"])**2
        posture_penalty += w_post["KFE"] * (get_q(self._jid_FR_KFE)-q_nom["FR_KFE"])**2

        k_posture = 0.01  # keep this small; raise to 0.1 if joints wander a lot
        posture_penalty *= k_posture * (0.5 + 0.5*max(0.0, heading_align))

        # ------------- Rewards / penalties -----------------
        # Linear running reward, gated by heading alignment
        forward_reward = self._forward_reward_weight * vx * heading_gate

        # Control cost on the CLIPPED command actually applied
        ctrl_cost = self._ctrl_cost_weight * float(np.sum(np.square(target)))
        alive_bonus = self.alive_bonus

        # Standing / step-back (unchanged)
        step_back_penalty = self.step_back_penalty_weight if vx < 0.0 else 0.0
        self.slow_progress_counter = self.slow_progress_counter + 1 if vx < 0.05 else 0
        standing_penalty = self.standing_penalty_weight if self.slow_progress_counter >= 5 else 0.0

        # Uprightness (unchanged)
        tilt_penalty = 2.0 * max(0.0, 1.0 - upright)

        # Foot contacts / alternation (unchanged)
        cL, cR, fL, fR = self._foot_contacts()
        foot_reward = 0.3 if (cL ^ cR) else 0.0
        double_support_penalty = 0.05 if (cL and cR and vx > 0.2) else 0.0
        # k_ds = 0.2   # start in 0.05–0.15 range
        # v_walk = 0.6  # walking–running transition speed
        
        # k_xor = 0.3
        # def sigmoid(x):
        #     return 1.0 / (1.0 + np.exp(-x))
        # foot_reward = k_xor * float(cL ^ cR) * sigmoid(vx - v_walk)
        # double_support_bonus = k_ds * float(cL and cR) * max(0.0, v_walk - vx)

        # if not hasattr(self, "_stance_balance"):
        #     self._stance_balance = 0.0  # positive means more L stance, negative more R stance

        # self._stance_balance += (1.0 if cL else 0.0) - (1.0 if cR else 0.0)

        # k_dom = 5e-4
        # dominance_penalty = k_dom * (self._stance_balance ** 2)


        # Joint speed penalty (unchanged)
        joint_speed_penalty = 0.002 * float(np.sum(self.data.qvel[6:] ** 2))

        # --- NEW: lateral drift & heading/yaw penalties ---
        k_lat  = 0.5   # lateral velocity penalty
        k_head = 0.3   # heading misalignment penalty
        k_yaw  = 0.02  # yaw-rate damping (tiny)
        lateral_penalty = k_lat * (vy ** 2)
        heading_penalty = k_head * max(0.0, 1.0 - heading_align)
        yawrate_penalty = k_yaw * (yaw_rate ** 2)

        # --- NEW: small HAA usage penalty to reduce gratuitous steering ---
        k_haa = 0.01
        haa_penalty = k_haa * float(action[0] ** 2 + action[3] ** 2)  # FL_HAA & FR_HAA

        # (Optional) smooth forward speed
        if not hasattr(self, "_prev_vx"):
            self._prev_vx = vx
        k_dv = 0.05
        dv_penalty = k_dv * (vx - self._prev_vx) ** 2
        self._prev_vx = vx

        # Termination (unchanged)
        base_h = float(self.data.qpos[2])
        self.low_height_counter = self.low_height_counter + 1 if base_h < self.min_height else 0
        terminated = (self.low_height_counter >= self.low_height_steps) or (upright < self.tilt_fail_threshold) 
        # --- NEW: vertical motion penalties ---
        h_nom = 0.44
        k_h = 0.2
        height_penalty = k_h * (base_h - h_nom)**2
        # Base vertical velocity
        vz = float(self.data.qvel[2])    # qvel index 2 is z velocity in MuJoCo default order

        # Contact booleans already: cL, cR
        in_flight = (not cL) and (not cR)

        # Weights (start small and tune)
        k_vz     = 0.5    # penalize vertical bouncing
        k_flight = 0.1    # penalize pure aerial phases
        k_dact   = 0.01   # action-rate smoothing

        # Vertical-motion and flight penalties
        vz_penalty     = k_vz * (vz ** 2)
        flight_penalty = k_flight * float(in_flight)

        # Simple action-rate smoothing
        if not hasattr(self, "_prev_action"):
            self._prev_action = np.asarray(action, dtype=np.float64)
        dact = np.asarray(action, dtype=np.float64) - self._prev_action
        self._prev_action = np.asarray(action, dtype=np.float64)
        dact_penalty = k_dact * float(np.sum(dact ** 2))

        roll  = float(np.arctan2( R[2,1], R[2,2]))

        k_pitch = 20
        k_roll  = 0.2 
        pitch = float(np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2)))
        pitch_penalty = k_pitch * (pitch**2)
        roll_penalty  = k_roll  * (roll**2)
        
        ## Energy cost computation
        qpos_flat = self.data.qpos.flat
        qvel_flat = self.data.qvel.flat
        q_list = []
        qd_list = []
        for jid, dof_adr in zip(self._joint_ids, self._dof_adrs):
            q_list.append(float(qpos_flat[self._qadr[jid]]))
            qd_list.append(float(qvel_flat[dof_adr]))
        tau = np.asarray([self.data.qfrc_actuator[d] for d in self._dof_adrs], dtype=np.float64)
        qd  = np.asarray(qd_list, dtype=np.float64)
        P_abs = float(np.sum(np.abs(tau * qd)))
        E_step = P_abs * self.dt
        energy_penalty = self.energy_cost_weight * E_step
        self._last_q   = q_list
        self._last_qd  = qd_list
        self._last_tau = tau


        # --- Total reward ---
        reward = (
            forward_reward + alive_bonus + foot_reward - energy_penalty
            - ctrl_cost - tilt_penalty
            - step_back_penalty - standing_penalty
            - joint_speed_penalty
            - lateral_penalty - heading_penalty - yawrate_penalty
            - haa_penalty - dv_penalty - stepwidth_penalty - crossing_penalty - vz_penalty
            - flight_penalty - dact_penalty - posture_penalty - roll_penalty
            - height_penalty - double_support_penalty #- dominance_penalty + double_support_bonus 
        )

        observation = self._get_obs()
        info = {
            "x_position": x_after, "y_position": y_after,
            "x_velocity": vx, "y_velocity": vy,
            "heading_align": heading_align, "yaw_rate": yaw_rate,
            "reward_run": forward_reward, "foot_reward": foot_reward,
            "tilt_penalty": tilt_penalty, "lateral_penalty": lateral_penalty,
            "heading_penalty": heading_penalty, "yawrate_penalty": yawrate_penalty,
            "haa_penalty": haa_penalty, "dv_penalty": dv_penalty,
            "reward_ctrl": -ctrl_cost,
            "step_back_penalty": step_back_penalty, "standing_penalty": standing_penalty,
            "foot_force_L": fL, "foot_force_R": fR, "base_h": base_h,
            "terminated_reason": "height" if self.low_height_counter >= self.low_height_steps else ("tilt" if upright < self.tilt_fail_threshold else None), 
            "step_width": step_width,
            "stepwidth_penalty": stepwidth_penalty,
            "crossing_penalty": crossing_penalty,
            "vz": vz,
            "vz_penalty": vz_penalty,
            "flight_penalty": flight_penalty,
            "dact_penalty": dact_penalty,
            "posture_penalty": posture_penalty,
            "pitch": pitch, "roll": roll,
            "pitch_penalty": pitch_penalty, "roll_penalty": roll_penalty,
            "height_penalty": height_penalty,

        }

        # # if self.render_mode == "human":
        # #     self.render()

        # # if self._log_enabled:
        # #     # time
        # #     self._ep["t"].append(float(self.data.time))

        # #     # torso pose (from free joint qpos or xipos/xmat)
        # #     self._ep["x"].append(float(self.data.qpos[0]))
        # #     self._ep["y"].append(float(self.data.qpos[1]))
        # #     self._ep["z"].append(float(self.data.qpos[2]))
        # #     self._ep["roll"].append(roll)
        # #     self._ep["pitch"].append(pitch)
        # #     self._ep["yaw"].append(yaw)
        # #     self._ep["upright"].append(upright)

        # #     # joint angles/velocities for your 6 hinges (use qpos via jnt_qposadr; you already have get_q)
        # #     # qpos_flat = self.data.qpos.flat
        # #     # qvel_flat = self.data.qvel.flat
        # #     # q_list = []
        # #     # qd_list = []
        # #     # for jid, dof_adr in zip(self._joint_ids, self._dof_adrs):
        # #     #     q_list.append(float(qpos_flat[self._qadr[jid]]))
        # #     #     qd_list.append(float(qvel_flat[dof_adr]))
        # #     self._ep["q"].append(q_list)       # (6,)
        # #     self._ep["qd"].append(qd_list)     # (6,)

        # #     # applied control after smoothing/offset (what we actually sent this step)
        # #     # MuJoCo keeps it here:
        # #     ctrl_now = np.asarray(self.data.ctrl).copy()
        # #     self._ep["ctrl"].append(ctrl_now.tolist())

        # #     # joint torques from actuators projected to DOFs (same order as qd)
        # #     # tau = np.asarray([self.data.qfrc_actuator[d] for d in self._dof_adrs], dtype=np.float64)
        # #     self._ep["tau"].append(tau.tolist())

        # #     # joint power
        # #     # qd = np.asarray(qd_list, dtype=np.float64)
        # #     self._ep["power"].append((tau * qd).tolist())
        # #     self._ep["energy_penalty"].append(float(energy_penalty))
        # #     self._ep["P_abs"].append(float(P_abs))


        #     # contact forces on lower-leg bodies (world-frame 3D force)
        #     if self._bid_FLbody != -1:
        #         FLf = self.data.cfrc_ext[self._bid_FLbody, :3]
        #     else:
        #         FLf = np.zeros(3)
        #     if self._bid_FRbody != -1:
        #         FRf = self.data.cfrc_ext[self._bid_FRbody, :3]
        #     else:
        #         FRf = np.zeros(3)
        #     self._ep["FL_fx"].append(float(FLf[0]))
        #     self._ep["FL_fy"].append(float(FLf[1]))
        #     self._ep["FL_fz"].append(float(FLf[2]))
        #     self._ep["FR_fx"].append(float(FRf[0]))
        #     self._ep["FR_fy"].append(float(FRf[1]))
        #     self._ep["FR_fz"].append(float(FRf[2]))

        #     # touch sensors (your scalar normal forces)
        #     self._ep["FL_touch"].append(float(fL))
        #     self._ep["FR_touch"].append(float(fR))

        #     # feet world positions (xyz)
        #     if self._sid_FLfoot != -1:
        #         self._ep["FL_pos"].append(self.data.site_xpos[self._sid_FLfoot].copy())
        #     elif self._bid_FLbody != -1:
        #         self._ep["FL_pos"].append(self.data.xipos[self._bid_FLbody].copy())
        #     else:
        #         self._ep["FL_pos"].append(np.zeros(3))

        #     if self._sid_FRfoot != -1:
        #         self._ep["FR_pos"].append(self.data.site_xpos[self._sid_FRfoot].copy())
        #     elif self._bid_FRbody != -1:
        #         self._ep["FR_pos"].append(self.data.xipos[self._bid_FRbody].copy())
        #     else:
        #         self._ep["FR_pos"].append(np.zeros(3))

        #     # gait & metrics
        #     self._ep["step_width"].append(float(step_width))
        #     self._ep["crossing_penalty"].append(float(crossing_penalty))
        #     self._ep["stance_L"].append(int(cL))
        #     self._ep["stance_R"].append(int(cR))

        #     # task signals
        #     self._ep["reward"].append(float(reward))
        #     self._ep["forward_reward"].append(float(forward_reward))
        #     self._ep["ctrl_cost"].append(float(ctrl_cost))
        #     self._ep["base_h"].append(float(base_h))
        #     self._ep["terminated_reason"].append(
        #         "height" if self.low_height_counter >= self.low_height_steps
        #         else ("tilt" if upright < self.tilt_fail_threshold else "")
        #     )

        return observation, float(reward), bool(terminated), False, info


    def _get_obs(self):
        qpos = self.data.qpos.flat.copy()   # nq=13
        qvel = self.data.qvel.flat.copy()   # nv=12
        if self._exclude_current_positions_from_observation:

            qpos = qpos[1:]

        return np.concatenate([qpos, qvel]).astype(np.float64)

    def reset_model(self):
        # self._save_episode()
        # self._reset_logger()
        # Start near neutral joint pose; small noise
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        

        noise = self._reset_noise_scale
        qpos += self.np_random.uniform(low=-noise, high=noise, size=self.model.nq)
        qvel += noise * self.np_random.standard_normal(self.model.nv)
        qvel[3:6] = 0.0
        self.set_state(qpos, qvel)
        def set_q(jid, val):
            if jid != -1:
                qpos[self._qadr[jid]] = val

    # small outward abduction, slight hip & knee flexion
        set_q(self._jid_FL_HAA, +0.06)
        set_q(self._jid_FR_HAA, -0.06)
        set_q(self._jid_FL_HFE, -0.30)
        set_q(self._jid_FR_HFE, -0.30)
        set_q(self._jid_FL_KFE, +0.20)
        set_q(self._jid_FR_KFE, +0.20) #If you want to change initial pose, do it here
        self.set_state(qpos, qvel)
        self._t = 0.0
        self._target_ema = self.offset.copy()

        for attr in ("_prev_vx", "_prev_yaw", "_prev_action", "_sw_dbg"):
            if hasattr(self, attr):
                delattr(self, attr)
        #self._target_ema = np.zeros(6, dtype=np.float64)
        return self._get_obs()
