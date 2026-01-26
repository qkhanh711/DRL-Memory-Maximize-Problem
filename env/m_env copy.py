import gymnasium as gym
from gymnasium import spaces
import numpy as np


def EnvConfig_v1(envName: str):
    print(f"Using environment configuration for: {envName}")
    return {
        # System / episode
        "num_users": 10,
        "T": 10,                    # episode length (steps)
        "sys_tau": 4,             # system latency budget (s)
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
        "max_denoise_steps": 25,
        "min_denoise_steps": 3,

        # Piecewise quadratic memory model (NEW)
        # Lower resolution: a1*px^2 + a2*px + a3
        "mem_a1": 1e-8,
        "mem_a2": 2e-4,
        "mem_a3": 1.0,
        # Mid resolution: constant
        "mem_const": 12.0,
        # High resolution: b1*px^2 + b2*px + b3
        "mem_b1": 5e-9,
        "mem_b2": 1e-4,
        "mem_b3": 8.0,
        "mem_threshold_low": 1024**2,    # pixels
        "mem_threshold_high": 1792**2,   # pixels
        # Old linear model (kept for backward compatibility)
        "c1": 3.81e-6,
        "c2": 4.86,

        # Workload model
        "base_image_size": 1024 * 1024,  # 1 MiB reference
        "base_resolution": 512 * 512,    # reference resolution in pixels (NEW)
        "GE0": 1e8,       # base FLOPS encoder
        "GD0": 1e8,       # base FLOPS decoder
        "G_eps": 1e8,     # FLOPS per denoise step
        "G_prompt": 1e7,  # FLOPS for prompt processing

        # Pricing model (NEW: explicit FLOPs pricing)
        "lambda_m": 1e-8,   # memory pricing coefficient
        "lambda_g": 5e-10,  # FLOPS pricing coefficient
        "lambda_c": 2.5e-7, # communication pricing coefficient

        # Latency model (NEW: denoising/overhead latency)
        "t_ldm_overhead": 0.005,  # fixed LDM overhead (s)
        "t_per_denoise": 0.0005,  # time per denoise step (s)

        # Wireless link / geometry
        "sp_pos": np.array([0.0, 0.0, 50.0]),
        "h0": 1.42e-4,
        "path_loss": 2.0,
        "bandwidth": 1e6,           # Hz
        "noise_power": 4.0e-21,     # W/Hz
        "upload_power": 0.0501,     # W
        "download_power": 0.5012,   # W

        # Reward bonus
        "psi": 100,

        # QoS target (lower BRISQUE is better)
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
        # Random position on ground plane (x,y), z=0
        self.position = self.rng.uniform(-500.0, 500.0, size=2)

        # Treat sizes as KB for realism, convert to bytes
        self.image_size = float(self.rng.uniform(100, 350) * 1024.0)  # 100–1000 KB
        self.prompt_size = float(self.rng.uniform(1, 10) * 1024.0)   # 10–100 KB

        self.direction = float(self.rng.uniform(0.0, 2.0 * np.pi))
        self.qos_required = config["qos_required"]
        self.mobility_speed = float(self.rng.uniform(0.5, 2.0))  # m/step
        self.mobility_angle = self.direction

    def update_position(self):
        dx = self.mobility_speed * np.cos(self.mobility_angle)
        dy = self.mobility_speed * np.sin(self.mobility_angle)
        self.position += np.array([dx, dy], dtype=np.float64)
        self.mobility_angle += float(self.rng.uniform(-0.1, 0.1))


class GAIServiceEnv_v1(gym.Env):
    """
    Action: continuous Box in [-1, 1]^(2*num_users)
      For each user i:
        a[2*i]   -> serve switch (>=0 => serve, <0 => skip)
        a[2*i+1] -> normalized denoise in [-1,1] -> mapped to [min_steps, max_steps]

    Observation: concatenated per-user features:
      [x, y, image_size_bytes, prompt_size_bytes, direction_rad, qos_required] for all users
    """
    metadata = {"render.modes": []}

    def __init__(self, config, seed):
        super().__init__()
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.users = [User(i, config, self.rng) for i in range(config["num_users"])]

        n = config["num_users"]
        # NEW: expanded state includes global context (Gmax, Mmax, avg latency)
        # Per-user: [x, y, image_size, prompt_size, direction, qos_required] = 6
        # Global: [remaining_Gmax, remaining_Mmax, avg_latency_last_step] = 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6 * n + 3,), dtype=np.float32
        )
        # Use symmetric action space that matches the internal normalization logic
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2 * n,), dtype=np.float32
        )

        self.time_step = 0
        self.total_flops_accumulated = 0.0  # cumulative FLOPS
        self.total_mem_accumulated = 0.0   # cumulative memory
        self.avg_latency_last_step = 0.0   # for state context

    # ------------- Gym API -------------
    def reset(self):
        self.time_step = 0
        self.total_flops_accumulated = 0.0
        self.total_mem_accumulated = 0.0
        self.avg_latency_last_step = 0.0
        for u in self.users:
            u.reset()
        return self._get_state()

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        assert action.shape == (2 * self.config["num_users"],)

        reward, info = self._compute_reward(action)
        self._move_users()
        # NEW: accumulate cumulative resources for state context
        self.total_flops_accumulated += info["total_flops"]
        self.total_mem_accumulated += info["total_mem"]
        self.avg_latency_last_step = info["total_latency"]
        self.time_step += 1
        done = bool(self.time_step >= self.config["T"])
        return self._get_state(), float(reward), done, info

    # ------------- Dynamics & helpers -------------
    def _get_state(self) -> np.ndarray:
        state = []
        # Per-user features
        for user in self.users:
            state.extend([
                float(user.position[0]),
                float(user.position[1]),
                float(user.image_size),
                float(user.prompt_size),
                float(user.direction),
                float(user.qos_required),
            ])
        # NEW: global context
        remaining_gmax = max(0.0, self.config["Gmax"] - self.total_flops_accumulated)
        remaining_mmax = max(0.0, self.config["Mmax"] - self.total_mem_accumulated)
        state.extend([
            float(remaining_gmax),
            float(remaining_mmax),
            float(self.avg_latency_last_step),
        ])
        return np.array(state, dtype=np.float32)

    def _move_users(self):
        for user in self.users:
            user.update_position()

    def _distance(self, user: User) -> float:
        user_pos_3d = np.array([user.position[0], user.position[1], 0.0], dtype=np.float64)
        return float(np.linalg.norm(self.config["sp_pos"] - user_pos_3d))

    def _channel_rate(self, distance: float, uplink: bool = True) -> float:
        # Simple pathloss + Shannon capacity model
        h_i = self.config["h0"] / (distance ** self.config["path_loss"])
        B_i = self.config["bandwidth"] / self.config["num_users"]
        pwr = self.config["upload_power"] if uplink else self.config["download_power"]
        snr = pwr * h_i / (B_i * self.config["noise_power"])
        rate_bits_per_s = B_i * np.log2(1.0 + snr)
        return float(rate_bits_per_s / 8.0)  # bytes/s

    def _compute_flops(self, user: User, denoise_steps: int) -> float:
        # NEW: tie FLOPs to resolution instead of bytes
        # Approximate: pixels ≈ image_bytes / 3 (rough RGB mapping)
        pixels = user.image_size / 3.0
        rho = pixels / self.config["base_resolution"]
        return float(rho * (self.config["GE0"] + self.config["GD0"]
                            + denoise_steps * self.config["G_eps"]
                            + self.config["G_prompt"]))

    def _compute_memory(self, user: User) -> float:
        # NEW: piecewise quadratic memory model matching paper Eq. (13)
        # Approximate pixels from bytes: pixels ≈ bytes / 3
        pixels = user.image_size / 3.0
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
        
        return float(max(mem, 0.0))

    def _compute_qos(self, denoise_steps: int) -> float:
        # Return a synthetic BRISQUE-like score (lower is better)
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
        return float(score / 100.0)

    def _compute_price(self, mem: float, flops: float, comm_bytes: float) -> float:
        # NEW: add FLOPs pricing explicitly (was missing)
        # p_i = λ_m * m_i + λ_g * g_i + λ_c * S_i
        cfg = self.config
        price = (cfg["lambda_m"] * mem + 
                 cfg["lambda_g"] * flops + 
                 cfg["lambda_c"] * comm_bytes)
        return float(price)

    def _served_penalty(self, served_count: int) -> float:
        # NEW: replace hard thresholds with smooth exponential penalty
        ratio = served_count / self.config["num_users"]
        # Smooth penalty: α * exp(-ratio/κ)
        alpha, kappa = 30.0, 0.3
        penalty = alpha * np.exp(-ratio / kappa)
        return float(penalty)

    def _compute_latency(self, user: User, denoise_steps: int):
        d = self._distance(user)

        rate_up = self._channel_rate(d, uplink=True)
        rate_down = self._channel_rate(d, uplink=False)
        mem_rate = self.config["Rmem"]              # bytes/s
        compute_power = self.config["PVM"]          # FLOP/s

        flops = self._compute_flops(user, denoise_steps)

        # Communication latencies (s)
        t_up = (user.image_size + user.prompt_size) / max(rate_up, 1e-9)
        t_down = user.image_size / max(rate_down, 1e-9)

        # Memory access latency (s)
        t_mem = (user.image_size + user.prompt_size) / max(mem_rate, 1e-9)

        # Compute latency (s)
        t_comp = flops / max(compute_power, 1e-9)

        # NEW: restore denoising and LDM overhead latency (key coupling!)
        t_ldm_overhead = self.config["t_ldm_overhead"]  # fixed LDM overhead
        t_denoise = denoise_steps * self.config["t_per_denoise"]  # per-step overhead
        
        total_latency = t_up + t_mem + t_comp + t_down + t_ldm_overhead + t_denoise
        return float(total_latency), float(flops)

    def _map_action_to_decision(self, a_serve: float, a_steps: float):
        # serve switch
        serve = 1 if a_serve >= 0.0 else 0

        # normalized [-1,1] -> [0,1]
        t = (np.clip(a_steps, -1.0, 1.0) + 1.0) * 0.5
        min_s = self.config["min_denoise_steps"]
        max_s = self.config["max_denoise_steps"]
        denoise_steps = int(np.floor(min_s + t * (max_s - min_s)))
        denoise_steps = int(np.clip(denoise_steps, min_s, max_s))
        return serve, denoise_steps

    def _compute_reward(self, action: np.ndarray):
        cfg = self.config
        N = cfg["num_users"]

        total_reward = 0.0
        latencies = []
        total_flops = 0.0
        total_mem = 0.0
        total_penalty = 0.0
        served = 0

        # For richer logging
        per_user = {
            "serve": [],
            "steps": [],
            "latency": [],
            "flops": [],
            "mem": [],
            "qos": [],
            "price": [],
            "pen_qos": [],
            "pen_lat": [],
        }

        relu = lambda x: x if x > 0 else 0

        for i, user in enumerate(self.users):
            a_serve = float(action[2 * i])
            a_steps = float(action[2 * i + 1])
            serve, steps = self._map_action_to_decision(a_serve, a_steps)

            if serve:
                latency, flops = self._compute_latency(user, steps)
                mem = self._compute_memory(user)
                qos = self._compute_qos(steps)
                # NEW: pass FLOPs instead of latency to pricing function
                price = self._compute_price(mem, flops, user.image_size + user.prompt_size)

                # NEW: normalize QoS comparison (user.qos_required was already /100 in reset)
                qos_required_norm = user.qos_required / 100.0
                pen_q = cfg["lambda_qos"] * relu(qos - qos_required_norm)
                print("pen Q", pen_q, "=", qos, "-", qos_required_norm)
                pen_l = cfg["lambda_latency"] * relu(latency - cfg["sys_tau"])
                print("pen L", pen_l, "=", latency, "-", cfg["sys_tau"])
                total_reward += price
                latencies.append(latency)
                total_flops += flops
                total_mem += mem
                total_penalty += (pen_q + pen_l)
                print(f"""
                      Total penalty  = {total_penalty} =
                      Penalty QoS   : {pen_q} +
                      Penalty Latency: {pen_l}""")
                # print("pen Q + pen L",total_penalty)
                served += 1
                print(f"""
                      User {i} served:
                      Steps: {steps}
                      Latency: {latency:.6f}
                      FLOservedPs: {flops:.0f}
                      Memory: {mem:.3f}
                      QoS: {qos:.3f}
                      Price: {price:.6f}
                      Penalty QoS: {pen_q:.6f}
                      Penalty Latency: {pen_l:.6f}
                      """)

                per_user["serve"].append(1)
                per_user["steps"].append(steps)
                per_user["latency"].append(latency)
                per_user["flops"].append(flops)
                per_user["mem"].append(mem)
                per_user["qos"].append(qos)
                per_user["price"].append(price)
                per_user["pen_qos"].append(pen_q)
                per_user["pen_lat"].append(pen_l)
            else:
                per_user["serve"].append(0)
                per_user["steps"].append(0)
                per_user["latency"].append(0.0)
                per_user["flops"].append(0.0)
                per_user["mem"].append(0.0)
                per_user["qos"].append(0.0)
                per_user["price"].append(0.0)
                per_user["pen_qos"].append(0.0)
                per_user["pen_lat"].append(0.0)
        
        if latencies:  
            total_latency = max(latencies)
        else:
            total_latency = 0.0

        total_penalty += cfg["lambda_latency"] * relu(total_latency - cfg["sys_tau"])
        # print(total_penalty)
        def normalize_flops(flops):
            return flops / cfg["Gmax"] * 100
        total_penalty += cfg["lambda_flops"] * relu(normalize_flops(total_flops))
        # print(total_penalty)
        total_penalty += cfg["lambda_mem"] * relu(total_mem - cfg["Mmax"])
        # print(total_penalty)
        total_penalty += self._served_penalty(served)
        # print(total_penalty)

        # Bonus if all constraints satisfied
        bonus = 0.0
        if (total_latency <= cfg["sys_tau"]
                and total_flops <= cfg["Gmax"]
                and total_mem <= cfg["Mmax"]):
            bonus = float(cfg["psi"])

        info = dict(
            total_served=served,
            total_latency=total_latency,
            total_flops=total_flops,
            total_mem=total_mem,
            penalty=total_penalty,
            bonus=bonus,
            per_user=per_user,
        )
        # print(f"""
            #   Total Reward : {total_reward}
            #   Total Penalty: {total_penalty}
            #   Total Bonus  : {bonus}
            #   """)
        return total_reward - total_penalty + bonus, info


if __name__ == "__main__":
    cfg = EnvConfig_v1("GAIServiceEnv")
    env = GAIServiceEnv_v1(cfg, seed=42)
    state = env.reset()

    print("Initial state shape:", state.shape)

    # Random policy demo
    action = env.action_space.sample()
    print("\n=== ONE STEP ===")
    obs, reward, done, info = env.step(action)
    print("Reward:", reward)
    print("Done:", done)
    print("Served:", info["total_served"])
    print("Total latency:", f"{info['total_latency']:.6f}")
    print("Total FLOPS:", f"{info['total_flops']:.0f}")
    print("Total memory:", f"{info['total_mem']:.3f}")
    print("Bonus:", info["bonus"])
    print("Penalty:", f"{info['penalty']:.6f}")
