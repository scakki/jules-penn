import numpy as np
import os
import mujoco
from gymnasium import utils
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class BoltEnvRunSimplifiedlunar2(MujocoEnv, utils.EzPickle):
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
        energy_cost_weight=0.2,
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
            tilt_fail_threshold,
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
        self.tilt_fail_threshold = tilt_fail_threshold
        self.low_height_counter = 0
        self.slow_progress_counter = 0
        self._prev_stance = 0
        self.energy_cost_weight = energy_cost_weight
        self._contact_hist_L = []
        self._contact_hist_R = []
        self._hist_len = 40
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        self.offset = np.array([0, 0.40, -0.80,  0, 0.40, -0.80], dtype=np.float64)


        

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
        model_path = os.path.join(current_path, "..", "models", "bolt_bipedal_lunar2.xml")

        MujocoEnv.__init__(
            self,
            model_path,
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        self.action_space = Box(low=-0.5, high=0.5, shape=(6,), dtype=np.float32)

        #############################################
        ######### Sensor and body handles ###########
        #############################################

        self._torso_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        # Touch sensors (by name)
        self._sid_FL = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "FL_touch")
        self._sid_FR = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "FR_touch")

        # Addresses into sensordata (in doubles)
        self._sadr_FL = self.model.sensor_adr[self._sid_FL] if self._sid_FL != -1 else None
        self._sadr_FR = self.model.sensor_adr[self._sid_FR] if self._sid_FR != -1 else None
        self._sid_FLfoot = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "FL_foot_site")
        self._sid_FRfoot = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "FR_foot_site")
        self._bid_FLbody = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "FL_LOWER_LEG")
        self._bid_FRbody = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "FR_LOWER_LEG")

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

        self._joint_names = ["FL_HAA","FL_HFE","FL_KFE","FR_HAA","FR_HFE","FR_KFE"]
        self._joint_ids   = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                             for jn in self._joint_names]
        self._dof_adrs = [int(self.model.jnt_dofadr[jid]) for jid in self._joint_ids]


        #############################################
            ######### Helper functions ###########
        #############################################


    def _upright(self):
        R = self.data.xmat[self._torso_bid].reshape(3, 3)
        return float(R[2, 2])

    def _foot_contacts(self, threshold=1e-3):
        # Touch sensors output normal force magnitude
        fl = float(self.data.sensordata[self._sadr_FL]) if self._sadr_FL is not None else 0.0
        fr = float(self.data.sensordata[self._sadr_FR]) if self._sadr_FR is not None else 0.0
        return (fl > threshold), (fr > threshold), fl, fr
    
    def get_foot_forces(self):
        F_L = np.zeros(3)
        F_R = np.zeros(3)

        if self._bid_FLbody != -1:
            F_L = self.data.cfrc_ext[self._bid_FLbody][:3].copy()

        if self._bid_FRbody != -1:
            F_R = self.data.cfrc_ext[self._bid_FRbody][:3].copy()

        return F_L, F_R

    def _foot_y_positions(self):
        """Return (yL, yR) in world coords using sites if available, else bodies."""
        if self._sid_FLfoot != -1 and self._sid_FRfoot != -1:
            yL = float(self.data.site_xpos[self._sid_FLfoot, 1])
            yR = float(self.data.site_xpos[self._sid_FRfoot, 1])
            return yL, yR
        if self._bid_FLbody != -1 and self._bid_FRbody != -1:
            yL = float(self.data.xipos[self._bid_FLbody, 1])
            yR = float(self.data.xipos[self._bid_FRbody, 1])
            return yL, yR
        return None, None

    def control_cost(self, action):
        return self._ctrl_cost_weight * float(np.sum(np.square(action)))
    
    
    def step(self, action):
        a_blend = np.asarray(action, np.float64)

        target_raw = a_blend + self.offset
        self._target_ema = self._ema_alpha * self._target_ema + (1.0 - self._ema_alpha) * target_raw
        target = np.clip(self._target_ema, self._act_low, self._act_high)

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
        heading_gate = max(0.0, heading_align) ** 1.5

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
            crossing_penalty  = k_cross * max(0.0, yR - yL)
        if not hasattr(self, "_sw_dbg"):
            self._sw_dbg = 0
        if self._sw_dbg < 8 and yL is not None:
            self._sw_dbg += 1

        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        if not hasattr(self, "_prev_yaw"):
            self._prev_yaw = yaw
        dyaw = yaw - self._prev_yaw
        if dyaw > np.pi:  dyaw -= 2*np.pi
        if dyaw < -np.pi: dyaw += 2*np.pi
        yaw_rate = dyaw / self.dt
        self._prev_yaw = yaw

        qpos = self.data.qpos.flat

        def get_q(jid, default=0.0):
            return float(qpos[self._qadr[jid]]) if jid != -1 else default
        q_nom = {
            "FL_HAA": +0.05, "FR_HAA": -0.05,
            "FL_HFE": -0.35, "FR_HFE": -0.35,
            "FL_KFE": +0.25, "FR_KFE": +0.25,
        }
        w_post = {"HAA": 1.0, "HFE": 0.6, "KFE": 0.6} 

        posture_penalty = 0.0
        posture_penalty += w_post["HAA"] * (get_q(self._jid_FL_HAA)-q_nom["FL_HAA"])**2
        posture_penalty += w_post["HAA"] * (get_q(self._jid_FR_HAA)-q_nom["FR_HAA"])**2
        posture_penalty += w_post["HFE"] * (get_q(self._jid_FL_HFE)-q_nom["FL_HFE"])**2
        posture_penalty += w_post["HFE"] * (get_q(self._jid_FR_HFE)-q_nom["FR_HFE"])**2
        posture_penalty += w_post["KFE"] * (get_q(self._jid_FL_KFE)-q_nom["FL_KFE"])**2
        posture_penalty += w_post["KFE"] * (get_q(self._jid_FR_KFE)-q_nom["FR_KFE"])**2

        k_posture = 0.01 
        posture_penalty *= k_posture * (0.5 + 0.5*max(0.0, heading_align))

        
        #############################################
        ######### Reward components ###########
        #############################################

        # Linear running reward, gated by heading alignment
        forward_reward = self._forward_reward_weight * vx * heading_gate

        # Control cost on the CLIPPED command actually applied
        ctrl_cost = self._ctrl_cost_weight * float(np.sum(np.square(target)))
        alive_bonus = self.alive_bonus

        # Standing / step-back
        step_back_penalty = self.step_back_penalty_weight if vx < 0.0 else 0.0
        self.slow_progress_counter = self.slow_progress_counter + 1 if vx < 0.05 else 0
        standing_penalty = self.standing_penalty_weight if self.slow_progress_counter >= 5 else 0.0

        # Uprightness
        tilt_penalty = 2.0 * max(0.0, 1.0 - upright)

        # Foot contacts / alternation
        cL, cR, fL, fR = self._foot_contacts(threshold=0.05)
        self._contact_hist_L.append(1 if cL else 0)
        self._contact_hist_R.append(1 if cR else 0)

        if len(self._contact_hist_L) > self._hist_len:
            self._contact_hist_L.pop(0)
            self._contact_hist_R.pop(0)

        pL = np.mean(self._contact_hist_L)
        pR = np.mean(self._contact_hist_R)

        k_dom = 0.1
        dominance_penalty = k_dom * abs(pL - pR)
        foot_reward = 0.3 if (cL ^ cR) else 0.0
        double_support_penalty = 0.05 if (cL and cR and vx > 0.2) else 0.0
        if cL and not cR:
            stance = -1
        elif cR and not cL:
            stance = +1
        else:
            stance = 0

        alternation_reward = 0.0
        if stance != 0 and stance != self._prev_stance:
            alternation_reward = 0.3

        if stance != 0:
            self._prev_stance = stance
        
        # cL, cR, fL, fR = self._foot_contacts()
        single = float(cL ^ cR)
        double = float(cL and cR)
        flight = float((not cL) and (not cR))

        # smooth gate around v0
        v0 = 0.6        # transition speed m/s (tune)
        beta = 10.0     # sharpness (tune)
        g = 1.0 / (1.0 + np.exp(-beta * (vx - v0)))  # 0=slow, 1=fast

        # alternation event (you already compute)
        r_alt = alternation_reward  # e.g. 0.3 on switch else 0

        # weights
        w_d  = 0.10   # slow: reward double support a bit
        w_s  = 0.20   # both: reward single support
        w_f  = 0.30   # slow: penalize flight
        w_a  = 0.30   # fast: reward alternation events
        w_dd = 0.10   # fast: penalize too much double support

        contact_term = (1.0 - g) * (w_d*double + w_s*single - w_f*flight) \
                    + g * (w_a*r_alt + w_s*single - w_dd*double)

        # Joint speed penalty
        joint_speed_penalty = 0.002 * float(np.sum(self.data.qvel[6:] ** 2))

        k_lat  = 0.5   # lateral velocity penalty
        k_head = 0.3   # heading misalignment penalty
        k_yaw  = 0.02  # yaw-rate damping (tiny)
        lateral_penalty = k_lat * (vy ** 2)
        heading_penalty = k_head * max(0.0, 1.0 - heading_align)
        yawrate_penalty = k_yaw * (yaw_rate ** 2)

        k_haa = 0.01
        haa_penalty = k_haa * float(action[0] ** 2 + action[3] ** 2)  # FL_HAA & FR_HAA

        if not hasattr(self, "_prev_vx"):
            self._prev_vx = vx
        k_dv = 0.05
        dv_penalty = k_dv * (vx - self._prev_vx) ** 2
        self._prev_vx = vx

        # Termination
        base_h = float(self.data.qpos[2])
        self.low_height_counter = self.low_height_counter + 1 if base_h < self.min_height else 0
        terminated = (self.low_height_counter >= self.low_height_steps) or (upright < self.tilt_fail_threshold) 
        # vertical motion penalties
        h_nom = 0.44
        k_h = 0.2
        height_penalty = k_h * (base_h - h_nom)**2
        # Base vertical velocity
        vz = float(self.data.qvel[2]) 

        # Contact booleans already: cL, cR
        in_flight = (not cL) and (not cR)

        # Weights (start small and tune)

        k_flight = 0.1    # penalize pure aerial phases
        k_dact   = 0.01   # action-rate smoothing
        flight_penalty = k_flight * float(in_flight)

        # Simple action-rate smoothing
        if not hasattr(self, "_prev_action"):
            self._prev_action = np.asarray(action, dtype=np.float64)
        dact = np.asarray(action, dtype=np.float64) - self._prev_action
        self._prev_action = np.asarray(action, dtype=np.float64)
        dact_penalty = k_dact * float(np.sum(dact ** 2))

        roll  = float(np.arctan2( R[2,1], R[2,2]))

        k_pitch = 0.02
        k_roll  = 0.1 
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
        energy_penalty = self.energy_cost_weight * E_step  ### Uncomment this after testing

        # P_mech = np.abs(tau * qd)

        # # Weight vertical joints more
        # hip_vertical_weight = 1.5
        # knee_vertical_weight = 1.5

        # P_weighted = (
        #     hip_vertical_weight * (P_mech[1] + P_mech[4]) +  # hip flex/ext
        #     knee_vertical_weight * (P_mech[2] + P_mech[5]) + # knees
        #     1.0 * (P_mech[0] + P_mech[3])                    # ab/adduction
        # )

        # # Make vertical motion amplify energetic cost
        # vz = abs(self.data.qvel[2])
        # vertical_factor = 1.0 + 2.0 * vz

        # E_step = P_weighted * vertical_factor * self.dt
        # energy_penalty = self.energy_cost_weight * E_step
        self._last_q   = q_list
        self._last_qd  = qd_list
        self._last_tau = tau

        # Extract joint angles
        q_FL_HAA = get_q(self._jid_FL_HAA)
        q_FR_HAA = get_q(self._jid_FR_HAA)
        q_FL_HFE = get_q(self._jid_FL_HFE)
        q_FR_HFE = get_q(self._jid_FR_HFE)
        q_FL_KFE = get_q(self._jid_FL_KFE)
        q_FR_KFE = get_q(self._jid_FR_KFE)

        # Symmetry penalty
        symmetry_penalty = 0.0

        # HAA must be mirrored (sign flip)
        symmetry_penalty += (q_FL_HAA + q_FR_HAA)**2

        # HFE direct match
        symmetry_penalty += (q_FL_HFE - q_FR_HFE)**2

        # KFE direct match
        symmetry_penalty += (q_FL_KFE - q_FR_KFE)**2

        # scale it
        k_sym = 0.02
        symmetry_penalty *= k_sym

        k_vz = 0.7
        vz_penalty = k_vz * (vz ** 2)


        # --- Total reward ---
        # reward = (
        #     forward_reward + alive_bonus + foot_reward - energy_penalty
        #     - ctrl_cost #- tilt_penalty
        #     # - step_back_penalty - standing_penalty
        #     # - joint_speed_penalty
        #     # - lateral_penalty - heading_penalty - yawrate_penalty
        #     # - haa_penalty - dv_penalty - stepwidth_penalty - crossing_penalty - vz_penalty
        #     - flight_penalty #- dact_penalty - posture_penalty - roll_penalty
        #     - height_penalty# - double_support_penalty #- dominance_penalty + double_support_bonus
        # )

        reward = (
            forward_reward + alive_bonus - energy_penalty - ctrl_cost #- vz_penalty #- dominance_penalty #+ contact_term # - roll_penalty - pitch_penalty - symmetry_penalty
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

        return observation, float(reward), bool(terminated), False, info


    def _get_obs(self):
        qpos = self.data.qpos.flat.copy()   # nq=13
        qvel = self.data.qvel.flat.copy()   # nv=12
        if self._exclude_current_positions_from_observation:

            qpos = qpos[1:]

        return np.concatenate([qpos, qvel]).astype(np.float64)

    def reset_model(self):
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

        set_q(self._jid_FL_HAA, +0.06)
        set_q(self._jid_FR_HAA, -0.06)
        set_q(self._jid_FL_HFE, -0.30)
        set_q(self._jid_FR_HFE, -0.30)
        set_q(self._jid_FL_KFE, +0.20)
        set_q(self._jid_FR_KFE, +0.20)
        self.set_state(qpos, qvel)
        self._t = 0.0
        self._target_ema = self.offset.copy()

        for attr in ("_prev_vx", "_prev_yaw", "_prev_action", "_sw_dbg"):
            if hasattr(self, attr):
                delattr(self, attr)
        return self._get_obs()
