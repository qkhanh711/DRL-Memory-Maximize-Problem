# import gym
# from gym import spaces
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.special import j0


def EnvConfig_v1(envName: str):
    # print(f"Using environment configuration for: {envName}")
    return {
        # System / episode
        "num_users": 10,
        "T": 5,                    # episode length (steps)
        "sys_tau": 5.0,             # system latency budget (s)
        "step_size": 1,           # step size for queue checking (s)
        "Gmax": 5e9,                # FLOPS budget per step
        "Mmax": 48,                 # memory budget (arbitrary units)

        # Penalty weights
        "lambda_qos": 0.5,
        "lambda_latency": 0.5,
        "lambda_mem": 1.0,
        "lambda_flops": 1.0,
        "lambda_price": 1.0,        # NEW: pricing component weight

        # Compute / memory capacity
        "PVM": 1e12,                # 1 TFLOPS (bytes/sec when dividing flops? here used as FLOP/s)
        "Rmem": 2.304e12,           # memory bandwidth (bytes/s)

        # Denoise step bounds
        "max_denoise_steps": 15,
        "min_denoise_steps": 3,

        # Piecewise quadratic memory model (NEW)
        # Lower resolution: a1*px^2 + a2*px + a3
        "mem_a1": 3.661e-3,
        "mem_a2": 2.637e-3,
        "mem_a3": 6807.05,
        # Mid resolution: constant
        "mem_const": 110649.59,
        # High resolution: b1*px^2 + b2*px + b3
        "mem_b1": 1039e-2,
        "mem_b2": -32716,
        "mem_b3": 35948.36,
        "mem_threshold_low": 1024**2,    # pixels
        "mem_threshold_high": 1792**2,   # pixels
        # Old linear model (kept for backward compatibility)
        # "c1": 3.81e-6,
        # "c2": 4.86,

        # Workload model
        "base_image_size": 1024 * 1024,  # 1 MiB reference
        "base_resolution": 512 * 512,    # reference resolution in pixels (NEW)
        "GE0": 1e8,       # base FLOPS encoder
        "GD0": 1e8,       # base FLOPS decoder
        "G_eps": 1e8,     # FLOPS per denoise step
        "G_prompt": 1e7,  # FLOPS for prompt processing

        # Pricing model (NEW: explicit FLOPs pricing)
        "lambda_m": 1e-2,   # memory pricing coefficient
        "lambda_g": 5e-10,  # FLOPS pricing coefficient
        "lambda_c": 2.5e-6, # communication pricing coefficient

        # Latency model (NEW: denoising/overhead latency)
        "t_ldm_overhead": 0.005,  # fixed LDM overhead (s)
        "t_per_denoise": 0.0005,  # time per denoise step (s)

        # Wireless link / geometry
        "sp_pos": np.array([0.0, 0.0, 50.0]),
        "h0": 1.42e-4,
        "path_loss": 2.0,
        "bandwidth": 1e6,           # Hz
        "noise_power": 4.0e-21,     # W/Hz  (~-174 dBm/Hz)
        "upload_power": 0.0501,     # W
        "download_power": 0.5012,   # W

        # Jakes fading model
        "f_c": 2e9,                 # carrier frequency (Hz), 2 GHz
        "c_light": 3e8,             # speed of light (m/s)

        # Reward bonus
        "psi": 100,

        # QoS target (lower PIQUE is better)
        "qos_required": 30,
    }


class User:
    def __init__(self, user_id: int, config: dict, rng: np.random.Generator):
        self.user_id = user_id
        self.config = config
        self.rng = rng
        self.reset(config)

    def reset(self, config=None):
        if config is None:
            config = self.config
        self.position = self.rng.uniform(-500.0, 500.0, size=2)

        self.image_size =  float(self.rng.uniform(1, 5) * 1024.0 * 1024.0)  #1 - 5 MB
        self.prompt_size = float(self.rng.uniform(1, 10) * 1024.0)   # 10–100 KB

        self.direction = float(self.rng.uniform(0.0, 2.0 * np.pi))
        self.qos_required = config["qos_required"]
        self.mobility_speed = float(self.rng.uniform(5, 15))  # m/s, v_i ~ U(0.5, 1.5) per Jakes model
        self.mobility_angle = self.direction

        # Jakes time-correlated small-scale fading state
        f_c = config.get("f_c", 2e9)
        c_light = config.get("c_light", 3e8)
        t_s = config.get("step_size", 1.0)  # sub-slot duration (s)
        f_D = self.mobility_speed * f_c / c_light  # maximum Doppler frequency
        self.rho = float(j0(2 * np.pi * f_D * t_s))  # temporal correlation coefficient
        # Initialize Gamma_{i,0} ~ CN(0, 1)
        self.gamma: complex = (
            self.rng.standard_normal() + 1j * self.rng.standard_normal()
        ) / np.sqrt(2)
        
        self.is_served = False
        self.completion_time = None
        self.assigned_denoise_steps = None
        self.assigned_resolution = None
        self.memory_usage = 0.0

    def update_position(self):
        dx = self.mobility_speed * np.cos(self.mobility_angle)
        dy = self.mobility_speed * np.sin(self.mobility_angle)
        self.position += np.array([dx, dy], dtype=np.float64)
        self.mobility_angle += float(self.rng.uniform(-0.1, 0.1))
        # Jakes fading evolution: Gamma_{i,t} = rho_i * Gamma_{i,t-1} + sqrt(1-rho_i^2) * daleth
        # daleth_{i,t} ~ CN(0, 1)
        daleth = (
            self.rng.standard_normal() + 1j * self.rng.standard_normal()
        ) / np.sqrt(2)
        self.gamma = self.rho * self.gamma + np.sqrt(1.0 - self.rho ** 2) * daleth


class GAIServiceEnv_v1(gym.Env):

    metadata = {"render.modes": []}

    def __init__(self, config, seed):
        super().__init__()
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.users = [User(i, config, self.rng) for i in range(config["num_users"])]

        n = config["num_users"]
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6 * n,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low =0.0,
            high=1.0,
            shape=(3 * n,),  # NEW: denoise_steps (n) + resolutions (n) + serve_decisions (n)
            dtype=np.float32
        )
        
        # NEW: queue management and timing
        self.queue = []  # users waiting due to memory constraints
        self.current_time = 0.0
        self.next_queue_check_time = 0.0
        self.current_memory_usage = 0.0
        self.peaked_memory_usage = 0.0  # NEW: track peak memory usage
        self.total_users_served = 0  # NEW: track total served in entire episode
        self.total_revenue = 0.0  # NEW: track cumulative revenue across episode
        self.total_reward = 0.0  # NEW: track cumulative reward across episode
        self.completed_users = []  # NEW: track indices of users who completed service


    def map_action_to_decisions(self, action: np.ndarray, config: dict):
        num_users = self.config["num_users"]
        denoise_steps = []
        serve_decisions = []
        
        for i in range(num_users):
            a_ds = action[i]
            ds = int(self.config["min_denoise_steps"] + a_ds * (self.config["max_denoise_steps"] - self.config["min_denoise_steps"]))
            denoise_steps.append(ds)

            a_serve = action[2 * num_users + i]
            serve = 1 if a_serve >= 0.5 else 0
            serve_decisions.append(serve)
            

        return denoise_steps, serve_decisions
        
    def reset(self):
        self.time_step = 0
        self.current_time = 0.0
        self.next_queue_check_time = self.config["step_size"]
        self.current_memory_usage = 0.0
        self.peaked_memory_usage = 0.0  # NEW: reset peak memory for new episode
        self.total_flops_accumulated = 0.0
        self.total_mem_accumulated = 0.0
        self.avg_latency_last_step = 0.0
        self.total_users_served = 0  # NEW: reset counter for new episode
        self.total_revenue = 0.0  # NEW: reset cumulative revenue for new episode
        self.total_reward = 0.0  # NEW: reset cumulative reward for new episode
        self.completed_users = []  # NEW: reset completed users list
        self.queue = []
        for u in self.users:
            u.reset()
        return self._get_state()

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        
        self.current_time += self.config["step_size"]
        
        # NEW: Check for any completed users (both in queue and not in queue)
        self._check_completions()
        
        if self.current_time >= self.next_queue_check_time:
            self._process_queue()
            self.next_queue_check_time += self.config["step_size"]
        
        observation, reward, done, info = self._compute_reward(action)

        # Update user mobility so position changes across timesteps
        self._move_users()
        
        # Increment step counter and check termination
        self.time_step += 1
        done = bool(self.time_step >= self.config["T"]) or \
               bool(self.current_time >= self.config["sys_tau"]) or \
               self._all_users_served()
        
        info.update({
            "current_time": self.current_time,
            "current_memory_usage": self.current_memory_usage,
            "queue_size": len(self.queue),
            "total_users_served": self.total_users_served,  # NEW: total served in episode
            "user_positions": self._get_user_positions(),
            "user_status": self._get_user_status(),
        })
        
        if done:
            pass
            # print(f"\n{'='*50}")
            # print(f"EPISODE ENDED - Step {self.time_step}")
            # print(f"Total users served in entire sys_tau: {self.total_users_served}/{self.config['num_users']}")
            # print(f"Users completed service (indices): {sorted(self.completed_users)}")
            # print(f"Peaked memory usage: {self.peaked_memory_usage:.2f}/{self.config['Mmax']} units")
            # print(f"Total time elapsed: {self.current_time:.2f}s / {self.config['sys_tau']}s")
            # print(f"{'='*50}\n")
        
        return observation, float(reward), done, info
    
    def _all_users_served(self) -> bool:
        return all(u.is_served for u in self.users)
    
    def _update_peaked_memory(self):
        """Update peaked memory if current memory exceeds previous peak"""
        if self.current_memory_usage > self.peaked_memory_usage:
            self.peaked_memory_usage = self.current_memory_usage
    
    def _check_completions(self):
        for i, user in enumerate(self.users):
            if user.is_served and user.completion_time is not None:
                if self.current_time >= user.completion_time and i not in self.completed_users:
                    self.completed_users.append(i)
                    # Free up memory if user was holding memory
                    if user.memory_usage > 0:
                        self.current_memory_usage -= user.memory_usage
                        # print(f"✓ User {i} completed service at {self.current_time:.2f}s, freed {user.memory_usage:.2f} memory")
                        user.memory_usage = 0  # Reset to avoid double-freeing
        
        if self.completed_users:
            pass
            # print(f"[Step {self.time_step}] Completed users so far: {sorted(self.completed_users)}")
    
    def _process_queue(self):
        users_to_remove = []
        for user_id in self.queue:
            user = self.users[user_id]
            if user.completion_time is not None and self.current_time >= user.completion_time:
                # User completed (memory already freed in _check_completions)
                users_to_remove.append(user_id)
        
        # Remove completed users from queue
        for user_id in users_to_remove:
            self.queue.remove(user_id)
        
        # Try to serve users from queue
        users_to_serve = []
        for user_id in self.queue[:]:  # Use slice to avoid modifying during iteration
            user = self.users[user_id]
            # Check if user can be served (memory + latency constraints)
            if self.current_memory_usage + user.memory_usage <= self.config["Mmax"] and \
               user.assigned_denoise_steps is not None:
                # Can serve this user
                users_to_serve.append(user_id)
                self.current_memory_usage += user.memory_usage
                self._update_peaked_memory()  # NEW: track peak memory
                # Calculate completion time
                user.completion_time = self.current_time + self._calculate_service_time(user)
                user.is_served = True
                self.total_users_served += 1  # NEW: count users served from queue
                
                # NEW: Add revenue for user served from queue
                if user.assigned_resolution is not None:
                    _, flops = self._compute_latency(user, user.assigned_denoise_steps)
                    price = self._compute_price(
                        user.memory_usage,
                        flops,
                        user.image_size + user.prompt_size
                    )
                    self.total_revenue += float(price)
                
                self.queue.remove(user_id)
                # print(f"User {user_id} started service from queue at {self.current_time:.2f}s, " \
                    #   f"will complete at {user.completion_time:.2f}s")
    
    def _calculate_service_time(self, user: User) -> float:
        if user.assigned_denoise_steps is None:
            return 0.0
        latency, _ = self._compute_latency(user, user.assigned_denoise_steps)
        return latency
    
    def _get_state(self) -> np.ndarray:
       state = []
       for u in self.users:
           state.extend([
               u.position[0],
               u.position[1],
               u.image_size,
               u.prompt_size,
               u.direction,
               u.qos_required,
           ])
       return np.array(state, dtype=np.float32)

    def _get_user_positions(self):
        return [
            {
                "user_id": int(u.user_id),
                "x": float(u.position[0]),
                "y": float(u.position[1]),
            }
            for u in self.users
        ]

    def _get_user_status(self):
        queue_set = set(self.queue)
        completed_set = set(self.completed_users)
        return [
            {
                "user_id": int(u.user_id),
                "x": float(u.position[0]),
                "y": float(u.position[1]),
                "is_served": bool(u.is_served),
                "is_in_queue": bool(u.user_id in queue_set),
                "is_completed": bool(u.user_id in completed_set),
            }
            for u in self.users
        ]

    def _move_users(self):
        for user in self.users:
            user.update_position()

    def _distance(self, user: User) -> float:
        user_pos_3d = np.array([user.position[0], user.position[1], 0.0], dtype=np.float64)
        return float(np.linalg.norm(self.config["sp_pos"] - user_pos_3d))
    
    def _channel_gain_sq(self, user: "User", distance: float) -> float:
        """Returns |h_{i,t}|^2 = varkappa_{i,t} * |Gamma_{i,t}|^2

        Large-scale fading:  varkappa_{i,t} = h0 / d_{i,t}^varrho
        Small-scale fading:  Gamma_{i,t}  (time-correlated Jakes, stored on user)
        Combined gain:       h_{i,t} = sqrt(varkappa_{i,t}) * Gamma_{i,t}
        """
        varkappa = self.config["h0"] / (distance ** self.config["path_loss"])
        return float(varkappa * abs(user.gamma) ** 2)

    def _channel_rate(self, user: "User", distance: float, uplink: bool = True) -> float:
        """Shannon capacity using composite fading channel gain |h_{i,t}|^2."""
        h_sq = self._channel_gain_sq(user, distance)
        B_i = self.config["bandwidth"] / self.config["num_users"]
        pwr = self.config["upload_power"] if uplink else self.config["download_power"]
        snr = pwr * h_sq / (B_i * self.config["noise_power"])
        rate_bits_per_s = B_i * np.log2(1.0 + snr)
        return float(rate_bits_per_s / 8.0)  # bytes/s

    def _compute_flops(self, user: User, denoise_steps: int) -> float:
        pixels = user.image_size / 3.0
        rho = pixels / self.config["base_resolution"]
        return float(rho * self.config["GE0"] + self.config["GD0"]
                            + denoise_steps * self.config["G_eps"]
                            + self.config["G_prompt"])
    
    def _compute_resolutions(self):
        resolutions = []
        for user in self.users:
            pixels = user.image_size / 3.0
            res = int(np.sqrt(pixels))
            resolutions.append(res)
        return resolutions

    def _compute_memory(self, pixels) -> float:
        cfg = self.config
        if pixels <= cfg["mem_threshold_low"]:
            # Lower resolution: quadratic
            mem = cfg["mem_a1"] * pixels**2 + cfg["mem_a2"] * pixels + cfg["mem_a3"]
        elif pixels <= cfg["mem_threshold_high"]:
            # Mid resolution: constant
            mem = cfg["mem_const"]
        else:
            # High resolution: quadratic
            mem = cfg["mem_b1"] * pixels**2 + cfg["mem_b2"] * pixels + cfg["mem_b3"]
        
        return float(max(mem, 0.0))/1024.0  # convert to arbitrary units

    def _compute_qos(self, denoise_steps: int) -> float:
        # Return a synthetic PIQUE-like score (lower is better)
        # Maps to normalized [0,1] internally for paper consistency
        if denoise_steps < 8:
            score = self.rng.uniform(28.0, 36.0)  # poor
        elif denoise_steps < 12:
            score = self.rng.uniform(24.0, 32.0)  # fair
        elif denoise_steps < 18:
            score = self.rng.uniform(12.0, 18.0)  # good
        else:
            score = self.rng.uniform(8.0, 16.0)   # excellent
        
        # NEW: normalize to [0,1] matching paper range
        return float(score)

    def _compute_price(self, mem: float, flops: float, comm_bytes: float) -> float:
        # p_i = λ_m * m_i + λ_g * g_i + λ_c * S_i
        cfg = self.config
        price = (cfg["lambda_m"] * mem + 
                 cfg["lambda_g"] * flops + 
                 cfg["lambda_c"] * comm_bytes)
        return float(price)

    def _compute_latency(self, user: User, denoise_steps: int):
        d = self._distance(user)

        rate_up = self._channel_rate(user, d, uplink=True)
        rate_down = self._channel_rate(user, d, uplink=False)
        mem_rate = self.config["Rmem"]              
        compute_power = self.config["PVM"]          

        flops = self._compute_flops(user, denoise_steps)

        t_up = (user.image_size + user.prompt_size) / max(rate_up, 1e-9) * 1e-1
        t_down = user.image_size / max(rate_down, 1e-9) * 1e-1

        t_mem = (user.image_size + user.prompt_size) / max(mem_rate, 1e-9)

        t_comp = flops / max(compute_power, 1e-9) * 1e2

        t_ldm_overhead = self.config["t_ldm_overhead"]  # fixed LDM overhead
        t_denoise = denoise_steps * self.config["t_per_denoise"]  # per-step overhead
        total_latency = t_up + t_mem + t_comp + t_down + t_ldm_overhead + t_denoise
        return float(total_latency), float(flops)

    def _compute_reward(self, action: np.ndarray):
        cgf = self.config
        N = cgf["num_users"]
        total_served = 0
        
        denoise_steps_list, serve_decisions = self.map_action_to_decisions(action, cgf)
        resolutions = self._compute_resolutions()
        
        idx = np.arange(N)
        lat_flops = np.array([
            self._compute_latency(self.users[i], denoise_steps_list[i])
            for i in idx
        ])
        latencies = lat_flops[:, 0]
        flops_list = lat_flops[:, 1]

        memories = np.array([
            self._compute_memory(resolutions[i])
            for i in idx
        ])

        qos_scores = np.array([
            self._compute_qos(denoise_steps_list[i])
            for i in idx
        ])
        
        prices = np.array([
            self._compute_price(
                memories[i],
                flops_list[i],
                self.users[i].image_size + self.users[i].prompt_size
            )
            for i in idx
        ])
        
        # print("QoS requirements and denoise steps:")
        # print(denoise_steps_list)
        # print(resolutions)
        # print("Computing latencies and memories...")
        # print("Latencies:", latencies)
        # print("Memories:", memories)
        # print("Prices:", prices)
        # print("QoS scores:", qos_scores)

        served_users = []
        queued_users = []
        rejected_users = []
        
        for i in idx:
            user = self.users[i]
            
            if serve_decisions[i] == 0:
                continue
            
            if user.is_served or i in self.queue:
                continue
            
            if latencies[i] > cgf["sys_tau"]:
                # print(f"User {i}: latency {latencies[i]:.3f}s > tau {cgf['sys_tau']}s - rejected")
                rejected_users.append(i)
                continue
            
            if self.current_memory_usage + memories[i] <= cgf["Mmax"]:
                self.current_memory_usage += memories[i]
                self._update_peaked_memory()  # NEW: track peak memory
                user.is_served = True
                user.assigned_denoise_steps = denoise_steps_list[i]
                user.assigned_resolution = resolutions[i]
                user.memory_usage = memories[i]
                user.completion_time = self.current_time + latencies[i]
                served_users.append(i)
                total_served += 1
                # print(f"User {i}: served immediately, memory usage: {self.current_memory_usage:.2f}/{cgf['Mmax']}")
            else:
                if i not in self.queue:
                    self.queue.append(i)
                    user.assigned_denoise_steps = denoise_steps_list[i]
                    user.assigned_resolution = resolutions[i]
                    user.memory_usage = memories[i]
                    queued_users.append(i)
                    # print(f"User {i}: queued due to memory constraint, would use {memories[i]:.2f} memory")
        
        # print(f"Total served: {total_served}, Queued: {len(queued_users)}, Queue size: {len(self.queue)}, Rejected: {len(rejected_users)}")
        # print(f"Current memory usage: {self.current_memory_usage:.2f}/{cgf['Mmax']}")
        # print(f"Serve decisions: {serve_decisions}")  # NEW: show decisions made by agent
        
        self.total_users_served += total_served
        
        # NEW: Accumulate revenue from served users
        # print(served_users)
        step_revenue = float(np.sum([prices[i] for i in served_users])) if len(served_users) > 0 else 0.0
        self.total_revenue += step_revenue
        
        # NEW: Compute reward using the formula: R_t = sum_i [ b_i(t)*p_i - lambda_Q*max(0, Q_out - Q_req) - lambda_L*max(0, t_i - tau) + Phi ]
        reward = 0.0
        relu = lambda x: max(0.0, x)
        
        for i in served_users:
            # b_i(t) * p_i: payment/revenue from user i
            price_component = prices[i]
            
            # - lambda_Q * max(0, Q_out - Q_req): QoS penalty
            qos_penalty = cgf.get("lambda_qos", 0.5) * relu(qos_scores[i] - (cgf["qos_required"] / 100.0))
            
            # - lambda_L * max(0, t_i - tau): Latency penalty
            latency_penalty = cgf.get("lambda_latency", 0.5) * relu(latencies[i] - cgf["sys_tau"])
            
            # + Phi: bonus for serving
            phi_bonus = cgf.get("psi", 100)
            
            # Sum: b_i*p_i - lambda_Q*penalty_qos - lambda_L*penalty_latency + Phi
            # print(i, price_component, qos_penalty, latency_penalty, phi_bonus)
            user_reward = 1.5 * price_component - 1.2 * qos_penalty - 0.9 * latency_penalty + phi_bonus
            reward += user_reward
        
        # print(reward)
        self.total_reward += reward
        # print("Cumulative reward so far:", self.total_reward) 
        # print()
        done = False
        
        info = {
            # "served_users": served_users,
            "reward" : float(self.total_reward),
            "mean_latency": float(np.mean(latencies)) if len(latencies) > 0 else 0.0,
            "peaked_mem": float(self.peaked_memory_usage),
            "total_users_served": self.total_users_served,
            "total_revenue": float(self.total_revenue),  # NEW: cumulative revenue across episode
            "qos_scores": float(np.mean(qos_scores)) if len(qos_scores) > 0 else 0.0,
            "qos_violations": int(np.sum(qos_scores > (cgf["qos_required"] / 100.0))),
        }
        
        return self._get_state(), reward, done, info

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)

if __name__ == "__main__":
    cfg = EnvConfig_v1("GAIServiceEnv")
    env = GAIServiceEnv_v1(cfg, seed=42)



    n_tests = 100
    
    for test_id in range(n_tests):
        print(f"\n########## TEST EPISODE {test_id} ##########")
        state = env.reset()


        action = env.action_space.sample()
        # print("\n=== STEP 0 ===")
        observation, reward, done, info = env.step(action)
        i = 0
        while not done:
            i += 1
            action = env.action_space.sample()
            old_reward = reward
            observation, new_reward, done, info = env.step(action)
            # print(f"\n=== STEP {i} ===")
            print("Reward:", old_reward+new_reward)
            # print("Done:", done)
            print("Info:", info)
            # print("Next observation shape:", observation.shape)