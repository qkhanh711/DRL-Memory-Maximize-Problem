import gym
from gym import spaces
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

        # Compute / memory capacity
        "PVM": 1e12,                # 1 TFLOPS (bytes/sec when dividing flops? here used as FLOP/s)
        "Rmem": 2.304e12,           # memory bandwidth (bytes/s)

        # Denoise step bounds
        "max_denoise_steps": 25,
        "min_denoise_steps": 3,

        # Memory model
        "c1": 3.81e-6,
        "c2": 4.86,

        # Workload model
        "base_image_size": 1024 * 1024,  # 1 MiB reference
        "GE0": 1e8,       # base FLOPS encoder
        "GD0": 1e8,       # base FLOPS decoder
        "G_eps": 1e8,     # FLOPS per denoise step
        "G_prompt": 1e7,  # FLOPS for prompt processing

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
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6 * n,), dtype=np.float32
        )
        # Use symmetric action space that matches the internal normalization logic
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2 * n,), dtype=np.float32
        )

        self.time_step = 0

    # ------------- Gym API -------------
    def reset(self):
        self.time_step = 0
        for u in self.users:
            u.reset()
        return self._get_state()

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        assert action.shape == (2 * self.config["num_users"],)

        reward, info = self._compute_reward(action)
        self._move_users()
        self.time_step += 1
        done = bool(self.time_step >= self.config["T"])
        return self._get_state(), float(reward), done, info

    # ------------- Dynamics & helpers -------------
    def _get_state(self) -> np.ndarray:
        state = []
        for user in self.users:
            state.extend([
                float(user.position[0]),
                float(user.position[1]),
                float(user.image_size),
                float(user.prompt_size),
                float(user.direction),
                float(user.qos_required),
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
        rho = user.image_size / self.config["base_image_size"]
        return float(rho * (self.config["GE0"] + self.config["GD0"]
                            + denoise_steps * self.config["G_eps"]
                            + self.config["G_prompt"]))

    def _compute_memory(self, user: User) -> float:
        # Simple affine model: memory ~ c1 * image_bytes + c2
        return float(self.config["c1"] * user.image_size + self.config["c2"])

    def _compute_qos(self, denoise_steps: int) -> float:
        # Return a synthetic BRISQUE-like score (lower is better)
        if denoise_steps < 8:
            return float(self.rng.uniform(28.0, 36.0))  # poor
        elif denoise_steps < 12:
            return float(self.rng.uniform(24.0, 32.0))  # fair
        elif denoise_steps < 18:
            return float(self.rng.uniform(12.0, 18.0))  # good
        else:
            return float(self.rng.uniform(8.0, 16.0))   # excellent

    def _compute_price(self, mem: float, latency: float, comm_bytes: float) -> float:
        # print(f"""
        #       {1e-2 * mem}
        #       {1e-1* latency}
        #       {2.5e-7 * comm_bytes}
        #       """)
        return float(1e-8 * mem + 1e-1 * latency + 2.5e-7 * comm_bytes)

    def _served_penalty(self, served_count: int) -> float:
        ratio = served_count / self.config["num_users"]
        if ratio == 0:
            return 50
        elif ratio < 0.3:
            return 20.0
        elif ratio < 0.6:
            return 10.0
        elif ratio < 0.8:
            return 5.0
        else:
            return -10.0

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

        # Keep comments aligned with values: these are small overheads (milliseconds)
        t_ldm_overhead = 0.005    # ~5 ms fixed overhead
        t_denoise = denoise_steps * 0.0005  # ~0.5 ms per step
        t_ldm_overhead = 0   # ~5 ms fixed overhead
        t_denoise = denoise_steps * 0 # ~0.5 m
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
                price = self._compute_price(mem, latency, user.image_size + user.prompt_size)

                pen_q = cfg["lambda_qos"] * relu(qos - user.qos_required)
                pen_l = cfg["lambda_latency"] * relu(latency - cfg["sys_tau"])

                total_reward += price
                latencies.append(latency)
                total_flops += flops
                total_mem += mem
                total_penalty += (pen_q + pen_l)
                # print("pen Q + pen L",total_penalty)
                served += 1

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
