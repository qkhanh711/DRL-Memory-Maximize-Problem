# import gym
# import numpy as np
# from gym import spaces


# def compute_energy(ue, uav_pos, bs_pos, D_i):
#     # Giả sử năng lượng = khoảng cách truyền * hệ số + năng lượng xử lý
#     d_ue_uav = np.linalg.norm(uav_pos - getattr(ue, 'pos', np.zeros(3)))
#     d_uav_bs = np.linalg.norm(uav_pos - bs_pos)
#     # Hệ số năng lượng truyền tải
#     phi = ue.config.get('phi', 1.5e-9)
#     # Năng lượng xử lý (giả sử tỉ lệ với D_i nếu là GenAI, hoặc K nếu là DNN)
#     if ue.w == 1:
#         E_proc = 0.1 * D_i
#     else:
#         E_proc = 0.05 * ue.K
#     E = phi * (d_ue_uav + d_uav_bs) + E_proc
#     return E

# def compute_latency(ue, uav_pos, bs_pos, D_i):
#     # Latency = latency truyền + latency xử lý
#     d_ue_uav = np.linalg.norm(uav_pos - getattr(ue, 'pos', np.zeros(3)))
#     d_uav_bs = np.linalg.norm(uav_pos - bs_pos)
#     # Tốc độ truyền (giả sử tỉ lệ với channel gain)
#     rate_ue_uav = 1e6 * ue.g
#     rate_uav_bs = 1e6 * ue.gUAV_BS
#     # Latency truyền tải
#     L_comm = (ue.Simg + ue.Sprompt) / rate_ue_uav + (ue.Simg + ue.Sprompt) / rate_uav_bs
#     # Latency xử lý
#     if ue.w == 1:
#         L_proc = 0.01 * D_i
#     else:
#         L_proc = 0.005 * ue.K
#     L = L_comm + L_proc
#     return L

# def compute_qos(ue, D_i):
#     # QoS (BRISQUE): càng nhỏ càng tốt, tỉ lệ nghịch với D_i (nhiều bước denoise thì chất lượng tốt hơn)
#     if ue.w == 1:
#         # Q = np.random.randint(7, 45)
#         if D_i <= 10:
#             Q = np.random.randint(38, 45)
#         elif D_i <= 20:
#             Q = np.random.randint(25, 38)
#         elif D_i <= 30:
#             Q = np.random.randint(15, 25)
#         elif D_i <= 40:
#             Q = np.random.randint(10, 15)
#         else:
#             Q = np.random.randint(7, 10)
#     else:
#         Q = np.random.randint(80, 93)
#     return Q 


# class UE:
#     def __init__(self, idx, config):
#         self.idx = idx
#         self.config = config
#         self.reset()

#     def reset(self):
#         # Loại dịch vụ: 0 (DNN) hoặc 1 (GenAI)
#         self.w = np.random.choice([0, 1])
#         # Kích thước input image và prompt (giả sử random trong khoảng hợp lý)
#         self.Simg = np.random.randint(128, 1024)  # ví dụ: 128-1024 KB
#         self.Sprompt = np.random.randint(1, 10)   # ví dụ: 1-10 KB
#         # Năng lực tính toán của BS (giả sử random quanh giá trị chuẩn)
#         self.f = self.config.get('f_inf_BS', 1e9) * np.random.uniform(0.8, 1.2)
#         # Channel gain UE-UAV (random)
#         self.g = np.random.uniform(1e-5, 1e-3)
#         # Channel gain UAV-BS (random, sẽ cập nhật ngoài env)
#         self.gUAV_BS = np.random.uniform(1e-5, 1e-3)
#         # Số lớp DNN nếu là DNN
#         self.K = np.random.randint(5, 20) if self.w == 0 else 0

#     def update(self):
#         # Có thể random lại các thông số cho mỗi bước
#         self.w = np.random.choice([0, 1])
#         self.Simg = np.random.randint(128, 1024)
#         self.Sprompt = np.random.randint(1, 10)
#         self.f = self.config.get('f_inf_BS', 1e9) * np.random.uniform(0.8, 1.2)
#         self.g = np.random.uniform(1e-5, 1e-3)
#         self.gUAV_BS = np.random.uniform(1e-5, 1e-3)
#         self.K = np.random.randint(5, 20) if self.w == 0 else 0 

# class UAVGenAIEnv(gym.Env):
#     def __init__(self, config):
#         self.num_users = config["num_users"]
#         self.Dmax = config["Dmax"]
#         self.tau = config["tau"]  
#         self.area = config["area"]
#         self.z_range = config["z_range"]
#         self.BS_position = np.array(config["BS_position"])
#         self.lam_Q = config["lambda_Q"]
#         self.lam_L = config["lambda_L"]
#         self.lam_E = config["lambda_E"]
#         self.psi = config["psi"]
#         self.Qreq = config.get("Qreq", [30.0]*self.num_users)
#         self.T = config.get("T", 100)
#         self.time = 0
#         self.ues = [UE(i, config) for i in range(self.num_users)]
        
#         # Define observation and action spaces
#         # State: [Simg_1, Sprompt_1, w_1, f_1, g_1, gUAV_BS_1, K_1, ..., UAV_x, UAV_y, UAV_z]
#         state_dim = self.num_users * 7 + 3  # 7 features per UE + 3 UAV position
#         self.observation_space = spaces.Box(
#             low=-np.inf, 
#             high=np.inf, 
#             shape=(state_dim,), 
#             dtype=np.float32
#         )
        
#         # Action: [D_1, D_2, ..., D_n, UAV_x, UAV_y, UAV_z]
#         action_dim = self.num_users + 3
#         self.action_space = spaces.Box(
#             low=np.array([0]*self.num_users + [self.area[0], self.area[2], self.z_range[0]]),
#             high=np.array([self.Dmax]*self.num_users + [self.area[1], self.area[3], self.z_range[1]]),
#             dtype=np.float32
#         )
        
#         self.reset()

#     def reset(self):
#         self.time = 0
#         for ue in self.ues:
#             ue.reset()
#         self.uav_pos = self._random_uav_pos()
#         self.state = self._get_state()
#         return self.state

#     def step(self, action):
#         D = action[:self.num_users]
#         uav_pos = action[self.num_users:self.num_users+3]
#         self.uav_pos = np.array(uav_pos)
#         rewards, Qs, Ls, Es = [], [], [], []
#         for i, ue in enumerate(self.ues):
#             if ue.w == 1:
#                 D_i = int(np.clip(D[i], 1, self.Dmax))
#             else:
#                 D_i = 0
#             Q_i = compute_qos(ue, D_i)
#             L_i = compute_latency(ue, self.uav_pos, self.BS_position, D_i)
#             E_i = compute_energy(ue, self.uav_pos, self.BS_position, D_i)
#             Qs.append(Q_i)
#             Ls.append(L_i)
#             Es.append(E_i)
#             r = -self.lam_Q * max(0, Q_i - self.Qreq[i]) \
#                 -self.lam_L * max(0, L_i - self.tau) \
#                 -self.lam_E * E_i
#             rewards.append(r)
#         # Bonus nếu thỏa mọi ràng buộc
#         if all(Qs[i] <= self.Qreq[i] and Ls[i] <= self.tau for i in range(self.num_users)):
#             reward = sum(rewards) + self.psi
#         else:
#             reward = sum(rewards)
#         self.time += 1
#         self._update_state()
#         done = self.time >= self.T
#         return self.state, reward, done, {}

#     def _get_state(self):
#         # Trả về vector trạng thái (numpy array):
#         # [Simg_1, Sprompt_1, w_1, f_1, g_1, gUAV_BS_1, K_1, ..., UAV_x, UAV_y, UAV_z]
#         state = []
#         for ue in self.ues:
#             state.extend([
#                 ue.Simg, ue.Sprompt, ue.w, ue.f, ue.g, ue.gUAV_BS, ue.K
#             ])
#         state.extend(self.uav_pos.tolist())
#         return np.array(state, dtype=np.float32)

#     def _update_state(self):
#         for ue in self.ues:
#             ue.update()
#         # UAV giữ nguyên vị trí nếu không có action di chuyển
#         self.state = self._get_state()

#     def _random_uav_pos(self):
#         x = np.random.uniform(self.area[0], self.area[1])
#         y = np.random.uniform(self.area[2], self.area[3])
#         z = np.random.uniform(self.z_range[0], self.z_range[1])
#         return np.array([x, y, z])
    
#     def seed(self, seed=None):
#         """Set the random seed for the environment"""
#         np.random.seed(seed)
#         return [seed] 

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import math
import os
import yaml
from typing import Dict, List, Tuple
# Mock implementations for BRISQUE to avoid import issues
class MockBRISQUE:
    def score(self, image):
        # Return a realistic BRISQUE score (lower is better, typical range 0-100)
        return np.random.uniform(20, 60)

class MockBRISQUEWrapper:
    def __init__(self):
        self.brisque = MockBRISQUE()
    
    def calculate_quality_score(self, image_data=None, image_size=None):
        # Return mock quality score based on size
        if image_size:
            base_score = 45.0
            size_factor = min(1.0, image_size / 262144)  # Normalize by base size
            return base_score - (size_factor * 10)  # Better quality for larger images
        return 40.0

def mock_bitsize(data_size):
    """Mock implementation of Bitsize function"""
    return data_size * 8  # Convert bytes to bits

# Use mock implementations
BRISQUE = MockBRISQUE()
BRISQUEWrapper = MockBRISQUEWrapper()
Bitsize = mock_bitsize

print("Using mock implementations for BRISQUE and utils")

# Global config variable
CONFIG = {}

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    global CONFIG
    try:
        # Try different possible paths for the config file
        possible_paths = [
            config_path,
            os.path.join(os.path.dirname(__file__), '..', '..', config_path),
            os.path.join(os.getcwd(), config_path)
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                with open(path, 'r') as file:
                    loaded_config = yaml.safe_load(file)
                CONFIG = loaded_config if loaded_config else {}
                return CONFIG
        
        print(f"Warning: Config file {config_path} not found in any of the expected locations. Using default values.")
        CONFIG = {}
        return CONFIG
    except Exception as e:
        print(f"Warning: Error loading config file {config_path}: {e}. Using default values.")
        CONFIG = {}
        return CONFIG

def get_config_value(path, default_value):
    """Safely get a config value with a default fallback"""
    try:
        keys = path.split('.')
        value = CONFIG
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default_value
        return value if value is not None else default_value
    except:
        return default_value

# NOTE: Global config loading removed - now done in constructor for centralized config management
# All constants now loaded in UAVEnv.__init__() method to support config parameter passing

# UAV Flight Velocity Constraints (improved with config support)
def get_flight_mode_constraints(flight_mode, config=None):
    """
    Get velocity constraints for a specific flight mode
    
    Args:
        flight_mode: Flight mode string ('p_mode' or 's_mode')
        config: Configuration dictionary (if None, uses default values)
    
    Returns:
        dict: Flight mode constraints
    """
    default_flight_modes = {
        'p_mode': {
            'max_horizontal_speed': 17.0,
            'max_ascent_speed': 5.0,
            'max_descent_speed': 4.0,
            'max_tilt_descent_speed': 4.0
        },
        's_mode': {
            'max_horizontal_speed': 23.0,
            'max_ascent_speed': 6.0,
            'max_descent_speed': 5.0,
            'max_tilt_descent_speed': 7.0
        }
    }
    
    if config is None:
        # Use default values
        mode_config = default_flight_modes.get(flight_mode, default_flight_modes.get('p_mode', {}))
    else:
        # Use provided config
        flight_modes = config.get('uav', {}).get('flight_modes', default_flight_modes)
        mode_config = flight_modes.get(flight_mode, flight_modes.get('p_mode', {}))
    
    return {
        'max_horizontal_speed': mode_config.get('max_horizontal_speed', 17.0),
        'max_ascent_speed': mode_config.get('max_ascent_speed', 5.0),
        'max_descent_speed': mode_config.get('max_descent_speed', 4.0),
        'max_tilt_descent_speed': mode_config.get('max_tilt_descent_speed', 4.0)
    }

def calculate_uav_position_after_step(current_pos, action, flight_mode, step_duration, config=None):
    """
    Calculate UAV position after one environment step with velocity constraints.
    
    Args:
        current_pos: Current UAV position [x, y, z] in meters
        action: Desired velocity [vx, vy, vz] in m/s (NOT delta position)
        flight_mode: UAV flight mode ('p_mode' or 's_mode')
        step_duration: Duration of environment step in seconds
        config: Configuration dictionary (if None, uses default position limits)
    
    Returns:
        new_position: New UAV position [x, y, z] in meters
        velocity: Actual velocity used [vx, vy, vz] in m/s (after constraints)
    """
    constraints = get_flight_mode_constraints(flight_mode, config)
    
    # Get position limits from config or use defaults
    if config:
        x_min = get_config_value_from_config(config, 'uav.position_limits.x_min', -500)
        x_max = get_config_value_from_config(config, 'uav.position_limits.x_max', 500)
        y_min = get_config_value_from_config(config, 'uav.position_limits.y_min', -500)
        y_max = get_config_value_from_config(config, 'uav.position_limits.y_max', 500)
        z_min = get_config_value_from_config(config, 'uav.position_limits.z_min', 100)
        z_max = get_config_value_from_config(config, 'uav.position_limits.z_max', 1000)
    else:
        x_min, x_max = -500, 500
        y_min, y_max = -500, 500
        z_min, z_max = 100, 1000
    
    # Convert action to desired velocity (action represents velocity in m/s)
    desired_velocity = np.array(action, dtype=np.float32)
    
    # Apply velocity constraints
    vx, vy, vz = desired_velocity
    
    # Horizontal velocity constraint (combined x,y)
    horizontal_speed = np.sqrt(vx**2 + vy**2)
    if horizontal_speed > constraints['max_horizontal_speed']:
        scale_factor = constraints['max_horizontal_speed'] / horizontal_speed
        vx *= scale_factor
        vy *= scale_factor
    
    # Vertical velocity constraints
    if vz > 0:  # Ascending
        vz = min(vz, constraints['max_ascent_speed'])
    else:  # Descending
        # Use appropriate descent constraint based on flight mode
        max_descent = constraints['max_descent_speed']
        vz = max(vz, -max_descent)
    
    # Calculate actual velocity vector
    actual_velocity = np.array([vx, vy, vz])
    
    # Calculate new position: pos_new = pos_old + velocity * time
    new_position = current_pos + actual_velocity * step_duration
    
    # Apply position bounds
    new_position[0] = np.clip(new_position[0], x_min, x_max)
    new_position[1] = np.clip(new_position[1], y_min, y_max)
    new_position[2] = np.clip(new_position[2], z_min, z_max)
    
    return new_position, actual_velocity

# Helper function for config access
def get_config_value_from_config(config, path, default_value):
    """Safely get a config value from provided config with a default fallback"""
    try:
        keys = path.split('.')
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default_value
        return value if value is not None else default_value
    except:
        return default_value


# Computational functions for GenAI processing (moved here to avoid global constant issues)
def calculate_image_scaling_factor(image_size_pixels, base_size=262144):
    """Calculate computational scaling factor based on image size"""
    # FLOPs scale approximately quadratically with image resolution
    scaling_factor = (image_size_pixels / base_size) ** 1.5
    return max(0.25, scaling_factor)  # Minimum 25% of base computation

def calculate_genai_processing_flops(denoising_steps, image_size_pixels, config=None):
    """Calculate total FLOPs required for GenAI processing"""
    # Get FLOP values from config or use defaults
    if config:
        encoder_flops = float(get_config_value_from_config(config, 'genai.flops.encoder', 1.0e12))
        decoder_flops = float(get_config_value_from_config(config, 'genai.flops.decoder', 1.0e12))
        unet_flops = float(get_config_value_from_config(config, 'genai.flops.unet_base_per_step', 5.0e12))
        text_flops = float(get_config_value_from_config(config, 'genai.flops.text_conditioning', 0.5e12))
        base_size = float(get_config_value_from_config(config, 'genai.base_image_size', 262144))
    else:
        encoder_flops = 1.0e12
        decoder_flops = 1.0e12
        unet_flops = 5.0e12
        text_flops = 0.5e12
        base_size = 262144.0
    
    scaling_factor = calculate_image_scaling_factor(image_size_pixels, base_size)
    
    # Ensure all values are float for computation
    denoising_steps = float(denoising_steps)
    scaling_factor = float(scaling_factor)
    
    # Total FLOPs = Encoder + (Denoising steps * UNet) + Text Conditioning + Decoder
    total_flops = (
        encoder_flops * scaling_factor +
        (denoising_steps * unet_flops * scaling_factor) +
        text_flops * scaling_factor +
        decoder_flops * scaling_factor
    )
    
    return total_flops

def calculate_bs_processing_time(denoising_steps, image_size_pixels, config=None):
    """Calculate actual BS processing time in seconds"""
    total_flops = calculate_genai_processing_flops(denoising_steps, image_size_pixels, config)
    
    # Get BS capacity from config or use default
    if config:
        bs_capacity_tflops = get_config_value_from_config(config, 'base_station.computing.capacity_tflops', 500.0)
    else:
        bs_capacity_tflops = 500.0
        
    bs_capacity_flops = bs_capacity_tflops * 1e12
    processing_time = total_flops / bs_capacity_flops
    
    return processing_time  # seconds


class UAVGenAIEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, config=None, num_ues=3, total_timesteps=50, output_gen_path="./data/output/generated_images/", config_path="config.yaml"):
        """
        Initialize UAV Environment with improved config management
        
        Args:
            config: Configuration dictionary (if provided, takes precedence over config_path)
            num_ues: Number of UEs (overrides config if specified)
            total_timesteps: Episode length (overrides config if specified)  
            output_gen_path: Output path (overrides config if specified)
            config_path: Path to config file (used only if config=None)
        """
        super(UAVGenAIEnv, self).__init__()

        # IMPROVED CONFIG LOADING: Support both centralized config and backward compatibility
        global CONFIG
        
        if config is not None:
            # Use provided config (centralized approach)
            CONFIG = config
            print("✅ Using provided config (centralized loading)")
        elif not CONFIG:
            # Fallback to loading from file (backward compatibility)
            load_config(config_path)
            print(f"⚠️  Loading config from file: {config_path} (consider using centralized config)")
        
        # Store config reference for helper functions
        self.config = CONFIG

        # LOAD ALL CONFIG-DEPENDENT CONSTANTS in constructor (centralized approach)
        # Communication parameters
        self.path_loss_exponent = get_config_value_from_config(CONFIG, 'communication.path_loss_exponent', 2.0)
        self.g0_channel_power_ref = get_config_value_from_config(CONFIG, 'communication.g0_channel_power_ref', 1.42e-4)
        self.noise_power_n0 = get_config_value_from_config(CONFIG, 'communication.noise_power_n0', 1e-9)
        
        # Bandwidth
        self.bw_ue_uav_subchannel = get_config_value_from_config(CONFIG, 'communication.bw_ue_uav_subchannel', 512000.0)
        self.bw_uav_bs_subchannel = get_config_value_from_config(CONFIG, 'communication.bw_uav_bs_subchannel', 1000000.0)
        
        # Transmit Powers (converted from dBm to W)
        p_ue_up_dbm = get_config_value_from_config(CONFIG, 'communication.power_ue_up_dbm', 17)
        p_uav_up_dbm = get_config_value_from_config(CONFIG, 'communication.power_uav_up_dbm', 17)
        p_uav_down_dbm = get_config_value_from_config(CONFIG, 'communication.power_uav_down_dbm', 17)
        p_bs_down_dbm = get_config_value_from_config(CONFIG, 'communication.power_bs_down_dbm', 40)
        
        self.p_ue_up_w = 10**((p_ue_up_dbm - 30) / 10)
        self.p_uav_up_w = 10**((p_uav_up_dbm - 30) / 10)
        self.p_uav_down_w = 10**((p_uav_down_dbm - 30) / 10)
        self.p_bs_down_w = 10**((p_bs_down_dbm - 30) / 10)
        
        # UAV position limits
        self.x_min_uav = get_config_value_from_config(CONFIG, 'uav.position_limits.x_min', -500)
        self.x_max_uav = get_config_value_from_config(CONFIG, 'uav.position_limits.x_max', 500)
        self.y_min_uav = get_config_value_from_config(CONFIG, 'uav.position_limits.y_min', -500)
        self.y_max_uav = get_config_value_from_config(CONFIG, 'uav.position_limits.y_max', 500)
        self.z_min_uav = get_config_value_from_config(CONFIG, 'uav.position_limits.z_min', 100)
        self.z_max_uav = get_config_value_from_config(CONFIG, 'uav.position_limits.z_max', 1000)
        
        # UAV initial position
        self.uav_initial_x = get_config_value_from_config(CONFIG, 'uav.initial_position.x', 0.0)
        self.uav_initial_y = get_config_value_from_config(CONFIG, 'uav.initial_position.y', 0.0)
        self.uav_initial_z = get_config_value_from_config(CONFIG, 'uav.initial_position.z', 300.0)
        
        # Base Station location
        self.bs_x = get_config_value_from_config(CONFIG, 'base_station.position.x', 0)
        self.bs_y = get_config_value_from_config(CONFIG, 'base_station.position.y', 50)
        self.bs_z = get_config_value_from_config(CONFIG, 'base_station.position.z', 50)
        self.bs_location = np.array([self.bs_x, self.bs_y, self.bs_z], dtype=np.float32)
        
        # Communication parameters  
        self.max_comm_range = get_config_value_from_config(CONFIG, 'communication.max_communication_range', 1000.0)
        self.min_snr_threshold = get_config_value_from_config(CONFIG, 'communication.min_snr_threshold', 5.0)
        
        # UE Configuration
        self.data_size_per_ue = get_config_value_from_config(CONFIG, 'ue.data_size_per_ue', 1000000)

        # Environment parameters (constructor args override config)
        self.num_ues = num_ues if num_ues != 3 else get_config_value_from_config(CONFIG, 'environment.num_ues', 3)
        self.total_timesteps = total_timesteps if total_timesteps != 50 else get_config_value_from_config(CONFIG, 'environment.total_timesteps', 50)
        self.output_gen_path = output_gen_path if output_gen_path != "./data/output/generated_images/" else get_config_value_from_config(CONFIG, 'environment.output_gen_path', "./data/output/generated_images/")
        
        self.current_step = 0
        
        # TIMING PARAMETERS: Load with clear documentation
        self.env_step_duration = get_config_value_from_config(CONFIG, 'environment.env_step_duration', 1.0)  # Physics simulation timestep (seconds)
        self.tau_slot_duration = get_config_value_from_config(CONFIG, 'communication.tau_slot_duration', 3.0)  # Max communication time per slot (seconds)
        
        # Check for transmission_timeout and warn if present
        comm_config = CONFIG.get('communication', {})
        if 'transmission_timeout' in comm_config:
            self.transmission_timeout = comm_config['transmission_timeout']
            episode_duration = self.total_timesteps * self.env_step_duration
            if self.transmission_timeout > episode_duration * 2:
                print(f"⚠️  WARNING: transmission_timeout ({self.transmission_timeout}s) >> episode_duration ({episode_duration}s)")
                print("   Consider removing transmission_timeout from config - it's redundant with tau_slot_duration")
        else:
            # transmission_timeout removed - use tau_slot_duration for per-transmission limits
            self.transmission_timeout = None

        # UAV flight configuration - use config parametermax_horizontal_velocity
        self.flight_mode = get_config_value_from_config(CONFIG, 'uav.default_flight_mode', 'p_mode')
        self.flight_constraints = get_flight_mode_constraints(self.flight_mode, CONFIG)

        # Action space configuration from config - VELOCITY BASED
        # Get velocity limits from flight constraints for consistency
        flight_constraints = get_flight_mode_constraints(self.flight_mode, CONFIG)
        max_horizontal_velocity = get_config_value_from_config(CONFIG, 'action_space.uav_movement.max_horizontal_velocity', flight_constraints['max_horizontal_speed'])
        max_ascent_velocity = get_config_value_from_config(CONFIG, 'action_space.uav_movement.max_ascent_velocity', flight_constraints['max_ascent_speed'])
        max_descent_velocity = get_config_value_from_config(CONFIG, 'action_space.uav_movement.max_descent_velocity', flight_constraints['max_descent_speed'])
        
        min_steps = get_config_value_from_config(CONFIG, 'action_space.denoising_steps.min_steps', 1)
        max_steps = get_config_value_from_config(CONFIG, 'action_space.denoising_steps.max_steps', 20)
        
        # --- Define Action Space --- [cite: 69]
        # Mixed action space: UAV movement velocities + Denoising steps optimization
        self.action_space = spaces.Dict({
            # 3D UAV velocity: (vx, vy, vz) in m/s - CONSISTENT WITH PHYSICS
            "uav_movement": spaces.Box(
                low=np.array([-max_horizontal_velocity, -max_horizontal_velocity, -max_descent_velocity]), 
                high=np.array([max_horizontal_velocity, max_horizontal_velocity, max_ascent_velocity]), 
                dtype=np.float32
            ),
            # Denoising steps for each UE (joint optimization)
            "denoising_steps": spaces.Box(
                low=np.array([min_steps] * self.num_ues),
                high=np.array([max_steps] * self.num_ues),
                dtype=np.int32
            )
        })

        # --- Define Observation Space (State Space) --- [cite: 67]
        # UAV position (3) + UAV velocity (3) + UE positions (3 * num_ues) + BS position (3)
        # Data states: sent, processed, received (3 * num_ues)  
        # Processing info: current_ue_id, remaining_time (2)
        # UE requests: image_size, quality_requirement per UE (2 * num_ues)
        # Current denoising steps per UE (num_ues)
        # Timestep info (1)
        # BS processing: queue_length(1) + estimated_times(num_ues) + job_progress(1)
        # Flight mode info (1)
        obs_dim = 3 + 3 + 3 * self.num_ues + 3 + 3 * self.num_ues + 2 + 2 * self.num_ues + self.num_ues + 1 + 1 + self.num_ues + 1 + 1
        
        self.observation_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(obs_dim,), 
            dtype=np.float32
        )

        # Initialize state variables for relay system
        self.ue_locs = None  # Will be set in reset()
        self.uav_current_loc = None
        self.uav_velocity = np.zeros(3, dtype=np.float32)  # Current velocity [vx, vy, vz]
        self.bs_loc = self.bs_location.copy()  # Use instance variable
        
        # Data flow states
        self.ue_data_sent = None      # UE -> UAV -> BS completed
        self.ue_data_processed = None # BS processing completed
        self.ue_data_received = None  # BS -> UAV -> UE completed
        
        # Generative AI states
        self.current_denoising_steps = None  # Current denoising steps per UE
        self.ue_image_sizes = None           # Image size requirements per UE
        self.ue_quality_requirements = None  # Quality requirements per UE
        self.target_quality_scores = None    # Target BRISQUE scores per UE
        
        # BS processing state
        self.current_processing_ue = -1  # -1 means no UE being processed
        self.processing_remaining_time = 0.0
        self.processing_start_time = 0.0
        self.processing_queue = []  # Sequential queue (FIFO)
        self.bs_total_processing_time = 0.0  # Track total processing time for rewards

        # Reward normalization
        self.reward_history = []  # Store rewards for normalization
        self.reward_mean = 0.0
        self.reward_std = 1.0  # Start with std=1 to avoid division by zero
        self.use_log_rewards = get_config_value_from_config(CONFIG, 'environment.use_log_rewards', False)
        
        # Load reward weights from config
        self.reward_weights = self._load_reward_weights(CONFIG)
        
        # Paper reward configuration
        self.use_paper_reward = get_config_value_from_config(CONFIG, 'rewards.use_paper_reward', True)
        print(f"🎯 Reward function: {'Paper-compliant' if self.use_paper_reward else 'Legacy'}")
        
        # Paper reward hyperparameters
        if self.use_paper_reward:
            self.lambda_quality = get_config_value_from_config(CONFIG, 'rewards.lambda_quality', 1.0)
            self.lambda_latency = get_config_value_from_config(CONFIG, 'rewards.lambda_latency', 0.1) 
            self.lambda_energy = get_config_value_from_config(CONFIG, 'rewards.lambda_energy', 1e-6)
            self.lambda_positioning = get_config_value_from_config(CONFIG, 'rewards.lambda_positioning', 0.5)
            self.lambda_connectivity = get_config_value_from_config(CONFIG, 'rewards.lambda_connectivity', 0.2)
            self.psi_bonus = get_config_value_from_config(CONFIG, 'rewards.psi_bonus', 100.0)
            self.tau_deadline = get_config_value_from_config(CONFIG, 'rewards.tau_deadline', 60.0)
            
            # Ensure all lambda parameters are float (fix for config parsing issues)
            self.lambda_quality = float(self.lambda_quality) if not isinstance(self.lambda_quality, (list, tuple)) else float(self.lambda_quality[0] if self.lambda_quality else 1.0)
            self.lambda_latency = float(self.lambda_latency) if not isinstance(self.lambda_latency, (list, tuple)) else float(self.lambda_latency[0] if self.lambda_latency else 0.1)
            self.lambda_energy = float(self.lambda_energy) if not isinstance(self.lambda_energy, (list, tuple)) else float(self.lambda_energy[0] if self.lambda_energy else 1e-6)
            self.lambda_positioning = float(self.lambda_positioning) if not isinstance(self.lambda_positioning, (list, tuple)) else float(self.lambda_positioning[0] if self.lambda_positioning else 0.5)
            self.lambda_connectivity = float(self.lambda_connectivity) if not isinstance(self.lambda_connectivity, (list, tuple)) else float(self.lambda_connectivity[0] if self.lambda_connectivity else 0.2)
            self.psi_bonus = float(self.psi_bonus) if not isinstance(self.psi_bonus, (list, tuple)) else float(self.psi_bonus[0] if self.psi_bonus else 100.0)
            self.tau_deadline = float(self.tau_deadline) if not isinstance(self.tau_deadline, (list, tuple)) else float(self.tau_deadline[0] if self.tau_deadline else 60.0)
        
    def _load_reward_weights(self, config):
        """Load reward weights from config with defaults"""
        rewards_config = get_config_value_from_config(config, 'rewards', {})
        
        return {
            # Primary objective weights
            'completion_weight': float(rewards_config.get('completion_weight', 100.0)),
            'upload_weight': float(rewards_config.get('upload_weight', 20.0)),
            'processing_weight': float(rewards_config.get('processing_weight', 50.0)),
            
            # Quality and efficiency weights
            'quality_bonus': float(rewards_config.get('quality_bonus', 30.0)),
            'quality_penalty': float(rewards_config.get('quality_penalty', 2.0)),
            'efficiency_weight': float(rewards_config.get('efficiency_weight', 25.0)),
            'quality_bonus_weight': float(rewards_config.get('quality_bonus_weight', 2.0)),
            
            # System optimization weights
            'bs_utilization_penalty': float(rewards_config.get('bs_utilization_penalty', 0.1)),
            'connectivity_weight': float(rewards_config.get('connectivity_weight', 10.0)),
            'positioning_penalty': float(rewards_config.get('positioning_penalty', 50.0)),
            
            # Time and completion weights
            'time_penalty': float(rewards_config.get('time_penalty', 0.5)),
            'completion_bonus': float(rewards_config.get('completion_bonus', 500.0)),
            'completion_step_penalty': float(rewards_config.get('completion_step_penalty', 2.0)),
            'efficiency_bonus': float(rewards_config.get('efficiency_bonus', 5.0)),
            'efficiency_threshold': float(rewards_config.get('efficiency_threshold', 60.0))
        }
    
    def update_reward_weights(self, new_weights):
        """Update reward weights during runtime"""
        self.reward_weights.update(new_weights)
        print(f"🎯 Updated reward weights: {new_weights}")
    
    def get_reward_weights(self):
        """Get current reward weights"""
        return self.reward_weights.copy()
    
    def _get_obs(self):
        """Get normalized state observation"""
        state = []
        
        # Normalize UAV position (3 values)
        uav_norm = self.uav_current_loc / 1000.0  # Normalize to [-1,1] range
        state.extend(uav_norm)
        
        # Normalize UAV velocity (3 values)
        max_velocity = max(self.flight_constraints['max_horizontal_speed'], 
                          self.flight_constraints['max_ascent_speed'])
        velocity_norm = self.uav_velocity / max_velocity
        state.extend(velocity_norm)
        
        # Normalize UE positions (3 * num_ues values)
        ue_norm = self.ue_locs / 1000.0
        state.extend(ue_norm.flatten())
        
        # Normalize BS position (3 values)
        bs_norm = self.bs_loc / 1000.0
        state.extend(bs_norm)
        
        # Data states (3 * num_ues values)
        state.extend(self.ue_data_sent.astype(float))
        state.extend(self.ue_data_processed.astype(float))
        state.extend(self.ue_data_received.astype(float))
        
        # Processing info (2 values)
        state.append(self.current_processing_ue / self.num_ues if self.current_processing_ue >= 0 else -1)
        state.append(self.processing_remaining_time / 60.0)  # Normalize to max 60 seconds
        
        # UE requests info (2 * num_ues values)
        # Normalize image sizes to [0,1] range (assuming max 2048x2048)
        normalized_sizes = self.ue_image_sizes / (2048.0 * 2048.0)
        state.extend(normalized_sizes)
        # Normalize quality requirements to [0,1] range (BRISQUE 0-100)
        normalized_quality = self.ue_quality_requirements / 100.0
        state.extend(normalized_quality)
        
        # Current denoising steps (num_ues values)
        # Normalize to [0,1] range (1-20 steps)
        normalized_denoising = (self.current_denoising_steps - 1.0) / 19.0
        state.extend(normalized_denoising)
        
        # Timestep info (1 value)
        state.append(self.current_step / self.total_timesteps)
        
        # BS processing information
        # Processing queue length (1 value)
        queue_length_norm = len(self.processing_queue) / self.num_ues
        state.append(queue_length_norm)
        
        # Estimated processing times for each UE (num_ues values)
        for ue_idx in range(self.num_ues):
            if not self.ue_data_processed[ue_idx] and self.ue_data_sent[ue_idx]:
                # Calculate estimated processing time
                denoising_steps = self.current_denoising_steps[ue_idx]
                image_size = self.ue_image_sizes[ue_idx]
                estimated_time = calculate_bs_processing_time(denoising_steps, image_size)
                normalized_time = min(estimated_time / 120.0, 1.0)  # Cap at 2 minutes
            else:
                normalized_time = 0.0
            state.append(normalized_time)
        
        # Current job progress (1 value)
        if self.current_processing_ue >= 0:
            denoising_steps = self.current_denoising_steps[self.current_processing_ue]
            image_size = self.ue_image_sizes[self.current_processing_ue]
            total_time = calculate_bs_processing_time(denoising_steps, image_size)
            progress = 1.0 - (self.processing_remaining_time / max(total_time, 0.1))
            progress = max(0.0, min(1.0, progress))
        else:
            progress = 0.0
        state.append(progress)
        
        # Flight mode info (1 value)
        # Encode flight mode: P mode = 0.0, S mode = 1.0
        flight_mode_encoding = 1.0 if self.flight_mode == 's_mode' else 0.0
        state.append(flight_mode_encoding)
        
        return np.array(state, dtype=np.float32)

    def _get_info(self):
        """Get comprehensive environment info including QoS, Energy, Latency metrics"""
        # Basic completion metrics
        completion_rate = np.sum(self.ue_data_received) / self.num_ues
        
        # Calculate separate completion rates for GenAI and DNN tasks
        genai_mask = self.ue_task_types == 'genai'
        dnn_mask = self.ue_task_types == 'dnn'
        
        if np.any(genai_mask):
            genai_completion_rate = np.sum(self.ue_data_received[genai_mask]) / np.sum(genai_mask)
        else:
            genai_completion_rate = 0.0
            
        if np.any(dnn_mask):
            dnn_completion_rate = np.sum(self.ue_data_received[dnn_mask]) / np.sum(dnn_mask)
        else:
            dnn_completion_rate = 0.0
        
        # Calculate QoS metrics
        quality_scores = []
        quality_requirements_met = 0
        quality_violations = 0
        total_denoising_efficiency = 0.0
        
        for ue_idx in range(self.num_ues):
            if self.ue_data_processed[ue_idx]:
                if self.ue_task_types[ue_idx] == 'genai':
                    # GenAI quality based on denoising steps (BRISQUE score)
                    denoising_steps = self.current_denoising_steps[ue_idx]
                    # Better quality with more denoising steps (lower BRISQUE score)
                    achieved_quality = max(15.0, 50.0 - (denoising_steps * 2.0))
                    quality_scores.append(achieved_quality)
                    
                    # Check if requirements met (lower BRISQUE = better)
                    if achieved_quality <= self.ue_quality_requirements[ue_idx]:
                        quality_requirements_met += 1
                    else:
                        quality_violations += 1
                    
                    # Calculate denoising efficiency
                    efficiency = min(1.0, self.ue_quality_requirements[ue_idx] / achieved_quality)
                    total_denoising_efficiency += efficiency
                    
                elif self.ue_task_types[ue_idx] == 'dnn':
                    # DNN quality based on model accuracy (mock implementation)
                    # Simulate accuracy based on processing complexity
                    achieved_quality = min(0.99, 0.7 + np.random.uniform(0, 0.2))
                    quality_scores.append(achieved_quality)
                    
                    # Check if requirements met (higher accuracy = better)
                    if achieved_quality >= self.ue_quality_requirements[ue_idx]:
                        quality_requirements_met += 1
                    else:
                        quality_violations += 1
                    
                    # DNN efficiency based on accuracy
                    efficiency = min(1.0, achieved_quality / max(self.ue_quality_requirements[ue_idx], 0.01))
                    total_denoising_efficiency += efficiency
        
        avg_quality_score = np.mean(quality_scores) if quality_scores else 0.0
        genai_quality_score = avg_quality_score  # Same as average for now
        qos_denoising_efficiency = total_denoising_efficiency / max(1, len(quality_scores))
        
        # Calculate Energy metrics (in Joules)
        # Energy = Power * Time
        transmission_energy = 0.0
        processing_energy = 0.0
        genai_processing_energy = 0.0
        dnn_processing_energy = 0.0
        
        for ue_idx in range(self.num_ues):
            if self.ue_data_sent[ue_idx]:
                # UE transmission energy (23 dBm = ~0.2W)
                ue_power_w = 0.2  # 23 dBm in Watts
                # UAV transmission energy (23 dBm = ~0.2W)  
                uav_power_w = 0.2
                # Estimate transmission time based on distance and data size
                distance = np.linalg.norm(self.ue_locs[ue_idx] - self.uav_current_loc)
                transmission_time = max(0.1, distance / 1000.0)  # Basic time estimate
                transmission_energy += (ue_power_w + uav_power_w) * transmission_time
            
            if self.ue_data_processed[ue_idx]:
                # Processing energy based on denoising steps
                denoising_steps = self.current_denoising_steps[ue_idx]
                # Assume 10W processing power, 0.1s per denoising step
                processing_time = denoising_steps * 0.1
                processing_power_w = 10.0
                step_processing_energy = processing_power_w * processing_time
                processing_energy += step_processing_energy
                genai_processing_energy += step_processing_energy  # All GenAI for now
        
        total_energy = transmission_energy + processing_energy
        energy_per_ue = total_energy / max(1, self.num_ues)
        
        # Calculate Latency metrics (in seconds)
        upload_latencies = []
        processing_latencies = []
        download_latencies = []
        total_latencies = []
        deadline_violations = 0
        
        for ue_idx in range(self.num_ues):
            if self.ue_data_sent[ue_idx]:
                # Calculate latencies based on distance and processing
                distance = np.linalg.norm(self.ue_locs[ue_idx] - self.uav_current_loc)
                
                # Upload latency (UE -> UAV -> BS)
                upload_latency = max(0.1, distance / 10000.0)  # Speed of light factor
                upload_latencies.append(upload_latency)
                
                # Processing latency
                if self.ue_data_processed[ue_idx]:
                    denoising_steps = self.current_denoising_steps[ue_idx]
                    processing_latency = denoising_steps * 0.1  # 0.1s per step
                    processing_latencies.append(processing_latency)
                    
                    # Download latency (BS -> UAV -> UE)
                    download_latency = upload_latency * 0.5  # Assume faster download
                    download_latencies.append(download_latency)
                    
                    # Total latency
                    total_latency = upload_latency + processing_latency + download_latency
                    total_latencies.append(total_latency)
                    
                    # Check deadline violation (assume 5s deadline)
                    if total_latency > 5.0:
                        deadline_violations += 1
        
        avg_upload_latency = np.mean(upload_latencies) if upload_latencies else 0.0
        avg_processing_latency = np.mean(processing_latencies) if processing_latencies else 0.0
        avg_download_latency = np.mean(download_latencies) if download_latencies else 0.0
        avg_total_latency = np.mean(total_latencies) if total_latencies else 0.0
        
        # Processing type determination
        if self.current_processing_ue >= 0:
            current_processing_type = 'processing'
        elif np.any(self.ue_data_sent & ~self.ue_data_processed):
            current_processing_type = 'upload'
        elif np.any(self.ue_data_processed & ~self.ue_data_received):
            current_processing_type = 'download'
        else:
            current_processing_type = 'idle'
        
        return {
            # Basic metrics
            "current_step": self.current_step,
            "ue_data_sent": self.ue_data_sent.copy(),
            "ue_data_processed": self.ue_data_processed.copy(),
            "ue_data_received": self.ue_data_received.copy(),
            "current_processing_ue": self.current_processing_ue,
            "completion_rate": completion_rate,
            "genai_completion_rate": genai_completion_rate,
            "dnn_completion_rate": dnn_completion_rate,
            "uav_velocity": self.uav_velocity.copy(),
            "flight_mode": self.flight_mode,
            "env_step_duration": self.env_step_duration,
            
            # Processing metrics
            "current_processing_type": current_processing_type,
            "total_genai_ues": np.sum(genai_mask),
            "total_dnn_ues": np.sum(dnn_mask),
            
            # QoS (Quality of Service) Metrics
            "qos_quality_score_avg": avg_quality_score,
            "qos_quality_score_genai": genai_quality_score,
            "qos_quality_requirements_met": quality_requirements_met,
            "qos_quality_violations": quality_violations,
            "qos_denoising_efficiency": qos_denoising_efficiency,
            
            # Energy Consumption Metrics (in Joules)
            "energy_total_consumption": total_energy,
            "energy_transmission": transmission_energy,
            "energy_processing": processing_energy,
            "energy_genai_processing": genai_processing_energy,
            "energy_dnn_processing": dnn_processing_energy,
            "energy_efficiency_per_ue": energy_per_ue,
            
            # Latency Metrics (in seconds)
            "latency_total_avg": avg_total_latency,
            "latency_upload_avg": avg_upload_latency,
            "latency_processing_avg": avg_processing_latency,
            "latency_download_avg": avg_download_latency,
            "latency_deadline_violations": deadline_violations
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        # Randomize UE positions (3D) - use instance variables
        self.ue_locs = np.random.uniform(
            low=[self.x_min_uav, self.y_min_uav, 0], 
            high=[self.x_max_uav, self.y_max_uav, 50],  # UEs at ground level to 50m height
            size=(self.num_ues, 3)
        )

        # Initialize UAV position (from config) - use instance variables
        self.uav_current_loc = np.array([self.uav_initial_x, self.uav_initial_y, self.uav_initial_z], dtype=np.float32)
        
        # Initialize UAV velocity (starts at rest)
        self.uav_velocity = np.zeros(3, dtype=np.float32)
        
        # Reset flight mode to default
        self.flight_constraints = get_flight_mode_constraints(self.flight_mode, self.config)
        
        # Reset data states
        self.ue_data_sent = np.zeros(self.num_ues, dtype=bool)
        self.ue_data_processed = np.zeros(self.num_ues, dtype=bool)
        self.ue_data_received = np.zeros(self.num_ues, dtype=bool)
        
        # Initialize task types based on service distribution
        service_distribution = CONFIG.get('ue', {}).get('service_distribution', {'genai': 1.0, 'dnn': 0.0})
        genai_prob = service_distribution.get('genai', 0.7)
        dnn_prob = service_distribution.get('dnn', 0.3)
        
        # Assign task types to UEs based on probabilities
        self.ue_task_types = np.random.choice(
            ['genai', 'dnn'], 
            size=self.num_ues, 
            p=[genai_prob, dnn_prob]
        )
        
        # Initialize states for both GenAI and DNN tasks
        self.ue_image_sizes = np.zeros(self.num_ues)
        self.ue_quality_requirements = np.zeros(self.num_ues)
        self.current_denoising_steps = np.zeros(self.num_ues, dtype=np.float32)
        self.target_quality_scores = np.zeros(self.num_ues)
        
        # Initialize GenAI UEs
        genai_mask = self.ue_task_types == 'genai'
        if np.any(genai_mask):
            # Randomize image sizes (in pixels^2) for GenAI UEs
            supported_sizes = CONFIG.get('genai', {}).get('supported_image_sizes', [65536, 262144, 589824, 1048576])
            self.ue_image_sizes[genai_mask] = np.random.choice(supported_sizes, size=np.sum(genai_mask))
            
            # Randomize quality requirements (target BRISQUE scores) for GenAI UEs
            quality_range = CONFIG.get('genai', {}).get('quality_range', {})
            min_brisque = quality_range.get('min_brisque', 15.0)
            max_brisque = quality_range.get('max_brisque', 35.0)
            self.ue_quality_requirements[genai_mask] = np.random.uniform(
                low=min_brisque, high=max_brisque, size=np.sum(genai_mask)
            )
            # Initialize denoising steps for GenAI UEs
            self.current_denoising_steps[genai_mask] = 10  # Default 10 steps
            self.target_quality_scores[genai_mask] = self.ue_quality_requirements[genai_mask]
        
        # Initialize DNN UEs
        dnn_mask = self.ue_task_types == 'dnn'
        if np.any(dnn_mask):
            # DNN tasks don't use image sizes or denoising steps
            # Use different quality metrics (e.g., accuracy instead of BRISQUE)
            self.ue_quality_requirements[dnn_mask] = np.random.uniform(
                low=0.8, high=0.99, size=np.sum(dnn_mask)  # Accuracy requirements for DNN
            )
            self.current_denoising_steps[dnn_mask] = 0  # No denoising for DNN
            self.target_quality_scores[dnn_mask] = self.ue_quality_requirements[dnn_mask]
        
        # Reset BS processing state
        self.current_processing_ue = -1
        self.processing_remaining_time = 0.0
        self.processing_start_time = 0.0
        self.processing_queue = []
        self.bs_total_processing_time = 0.0
        
        # Reset paper reward latency tracking
        self.ue_total_latencies = np.zeros(self.num_ues)  # Total latency per UE
        self.ue_upload_start_times = np.full(self.num_ues, -1.0)  # Upload start times
        self.ue_processing_start_times = np.full(self.num_ues, -1.0)  # Processing start times
        self.ue_download_start_times = np.full(self.num_ues, -1.0)  # Download start times

        observation = self._get_obs()
        info = self._get_info() if hasattr(self, '_get_info') else {}
        return observation, info

    def _calculate_channel_gain(self, loc1, loc2):
        """Calculate channel gain based on paper formulation [cite: 26, 30]"""
        # loc1: (x1, y1, z1) e.g. UAV
        # loc2: (x2, y2, z2) e.g. UE or BS 
        distance_sq = np.sum((loc1 - loc2)**2)
        if distance_sq == 0: # Avoid division by zero if coincident
            return self.g0_channel_power_ref # Use instance variable
        
        # Path loss model: g = g0 * d^(-β) where β is path loss exponent
        distance = np.sqrt(distance_sq)
        gain = self.g0_channel_power_ref * (distance ** (-self.path_loss_exponent))  # Use instance variables
        return gain

    def _calculate_rate(self, bandwidth, power, channel_gain, noise_power):
        """Calculate transmission rate using Shannon-Hartley theorem"""
        # Shannon-Hartley theorem: R = B * log2(1 + SNR)
        # SNR = P * g / N0
        
        # Ensure all values are float for computation
        bandwidth = float(bandwidth)
        power = float(power)
        channel_gain = float(channel_gain)
        noise_power = float(noise_power)
        
        snr = (power * channel_gain) / noise_power
        rate = bandwidth * math.log2(1 + snr) if snr > 0 else 0 # bps
        return rate

    def _calculate_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Calculate Euclidean distance between two positions"""
        return np.linalg.norm(pos1 - pos2)
    
    def _calculate_channel_quality(self, distance: float) -> float:
        """Calculate channel quality based on distance - improved connectivity"""
        if distance > self.max_comm_range:
            return 0.0
        
        # Improved path loss model with more permissive thresholds
        if distance < 100.0:  # Very close range
            return 1.0
        elif distance < 500.0:  # Medium range  
            return 0.8
        elif distance < 1000.0:  # Long range
            return 0.5
        elif distance < 1500.0:  # Extended range
            return 0.3
        else:  # Maximum range
            return 0.1
    
    def _can_communicate(self, pos1: np.ndarray, pos2: np.ndarray) -> bool:
        """Check if two positions can communicate - improved connectivity"""
        distance = self._calculate_distance(pos1, pos2)
        
        # More permissive communication check
        if distance > self.max_comm_range:
            return False
        
        quality = self._calculate_channel_quality(distance)
        return quality > 0.05  # Very low threshold (reduced from 0.1)
    
    def _calculate_transmission_time(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Calculate transmission time based on Shannon rate and data size"""
        # Use proper channel model and rate calculation
        channel_gain = self._calculate_channel_gain(pos1, pos2)
        
        # Determine power and bandwidth based on link type
        distance = self._calculate_distance(pos1, pos2)
        
        # Assume UE-UAV link if one position is below 100m (ground level)
        if min(pos1[2], pos2[2]) < 100:
            # UE-UAV link - use instance variables
            power = self.p_ue_up_w
            bandwidth = self.bw_ue_uav_subchannel
        else:
            # UAV-BS link - use instance variables
            power = self.p_uav_up_w
            bandwidth = self.bw_uav_bs_subchannel
        
        # Calculate transmission rate - use instance variable for noise power
        rate = self._calculate_rate(bandwidth, power, channel_gain, self.noise_power_n0)
        
        if rate <= 0:
            return float('inf')
        
        # Transmission time = Data size / Rate - use instance variable
        return self.data_size_per_ue / rate

    def _calculate_quality_score(self, ue_idx):
        """Calculate quality score for a specific UE based on denoising steps"""
        denoising_steps = self.current_denoising_steps[ue_idx]
        
        # Quality vs denoising steps relationship - CORRECTED
        # More denoising steps = Better quality (lower BRISQUE score)
        if denoising_steps < 5:
            return np.random.uniform(40, 50)  # Poor quality (high BRISQUE)
        elif denoising_steps < 10:
            return np.random.uniform(30, 40)  # Fair quality  
        elif denoising_steps < 15:
            return np.random.uniform(20, 30)  # Good quality
        else:
            return np.random.uniform(10, 20)  # Excellent quality (low BRISQUE)
    
    def _mock_brisque_score(self, denoising_steps):
        """Legacy function for backward compatibility - CORRECTED"""
        # More denoising steps = Better quality (lower BRISQUE score)
        if denoising_steps < 10:
            return np.random.uniform(35, 45)  # Poor quality (high BRISQUE)
        elif denoising_steps >= 10 and denoising_steps < 20:
            return np.random.uniform(20, 30)  # Good quality
        elif denoising_steps >= 20:
            return np.random.uniform(10, 20)  # Excellent quality (low BRISQUE) 
        

    def step(self, action):
        self.current_step += 1

        # Process mixed action space
        if isinstance(action, dict):
            # Extract UAV movement and denoising steps
            uav_movement = action["uav_movement"]
            denoising_steps = action["denoising_steps"]
        else:
            # Backward compatibility with simple action space
            uav_movement = action
            denoising_steps = np.ones(self.num_ues) * 10  # Default 10 steps
        
        # Apply velocity constraints and calculate new UAV position
        # uav_movement now represents desired velocity in m/s
        new_position, actual_velocity = calculate_uav_position_after_step(
            self.uav_current_loc, 
            uav_movement, 
            self.flight_mode, 
            self.env_step_duration,
            self.config
        )
        
        # Update UAV state
        self.uav_current_loc = new_position
        self.uav_velocity = actual_velocity
        
        # Update current denoising steps
        min_steps = get_config_value_from_config(CONFIG, 'action_space.denoising_steps.min_steps', 1)
        max_steps = get_config_value_from_config(CONFIG, 'action_space.denoising_steps.max_steps', 20)
        self.current_denoising_steps = np.clip(denoising_steps, min_steps, max_steps)
        
        # Process data transmission and processing
        self._process_data_flow()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = self._is_done()
        truncated = False
        
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, done, truncated, info

    def _process_data_flow(self):
        """Data flow processing with TFLOPs-based BS computation"""
        
        # Step 1: UE to UAV to BS (upload)
        for ue_idx in range(self.num_ues):
            if not self.ue_data_sent[ue_idx]:
                # Check if UE can send data to UAV and UAV can relay to BS
                ue_to_uav = self._can_communicate(self.ue_locs[ue_idx], self.uav_current_loc)
                uav_to_bs = self._can_communicate(self.uav_current_loc, self.bs_loc)
                
                if ue_to_uav and uav_to_bs:
                    # Calculate transmission time
                    upload_time = (self._calculate_transmission_time(self.ue_locs[ue_idx], self.uav_current_loc) +
                                 self._calculate_transmission_time(self.uav_current_loc, self.bs_loc))
                    
                    # Use tau_slot_duration as transmission timeout if transmission_timeout not set
                    timeout = self.transmission_timeout if self.transmission_timeout else self.tau_slot_duration
                    if upload_time < timeout:  # Reasonable transmission time
                        self.ue_data_sent[ue_idx] = True
                        # Add to processing queue if not already there
                        if ue_idx not in self.processing_queue:
                            self.processing_queue.append(ue_idx)
        
        # Step 2: BS processing (sequential GenAI processing)
        if self.current_processing_ue == -1:
            # Start processing next UE in queue
            if self.processing_queue:
                ue_idx = self.processing_queue.pop(0)  # FIFO
                if self.ue_data_sent[ue_idx] and not self.ue_data_processed[ue_idx]:
                    self.current_processing_ue = ue_idx
                    
                    # Calculate realistic processing time based on TFLOPs - pass config
                    denoising_steps = self.current_denoising_steps[ue_idx]
                    image_size = self.ue_image_sizes[ue_idx]
                    self.processing_remaining_time = calculate_bs_processing_time(denoising_steps, image_size, self.config)
                    self.processing_start_time = self.processing_remaining_time
                    
                    # Track total processing time for rewards
                    self.bs_total_processing_time += self.processing_remaining_time
        else:
            # Continue processing current UE
            time_step = self.env_step_duration  # Use environment step duration
            self.processing_remaining_time -= time_step
            
            if self.processing_remaining_time <= 0:
                # Processing completed
                self.ue_data_processed[self.current_processing_ue] = True
                self.current_processing_ue = -1
                self.processing_remaining_time = 0.0
        
        # Step 3: BS to UAV to UE (download)
        for ue_idx in range(self.num_ues):
            if (self.ue_data_processed[ue_idx] and 
                not self.ue_data_received[ue_idx]):
                
                # Check if BS can send to UAV and UAV can relay to UE
                bs_to_uav = self._can_communicate(self.bs_loc, self.uav_current_loc)
                uav_to_ue = self._can_communicate(self.uav_current_loc, self.ue_locs[ue_idx])
                
                if bs_to_uav and uav_to_ue:
                    # Calculate transmission time
                    download_time = (self._calculate_transmission_time(self.bs_loc, self.uav_current_loc) +
                                   self._calculate_transmission_time(self.uav_current_loc, self.ue_locs[ue_idx]))
                    
                    # Use tau_slot_duration as transmission timeout if transmission_timeout not set
                    timeout = self.transmission_timeout if self.transmission_timeout else self.tau_slot_duration
                    if download_time < timeout:  # Reasonable transmission time
                        self.ue_data_received[ue_idx] = True
    
    def _signed_log(self, x):
        """Signed logarithm transformation"""
        if x >= 0:
            return np.log(x + 1)
        else:
            return -np.log(-x + 1)
        
    def _calculate_raw_reward(self) -> float:
        """Reward calculation with configurable weights"""
        # Use paper reward if enabled, otherwise use legacy reward
        if self.use_paper_reward:
            return self._calculate_paper_reward()
        
        # Legacy reward calculation
        weights = self.reward_weights
        
        # 1. Completion rewards (primary objective)
        completion_reward = np.sum(self.ue_data_received) * weights['completion_weight']
        
        # 2. Upload and processing progress rewards
        upload_reward = np.sum(self.ue_data_sent) * weights['upload_weight']
        processing_reward = np.sum(self.ue_data_processed) * weights['processing_weight']
        
        # 3. Quality rewards - reward for achieving target quality with optimal denoising
        quality_reward = 0.0
        for ue_idx in range(self.num_ues):
            if self.ue_data_received[ue_idx]:
                # Calculate quality score based on denoising steps
                achieved_quality = self._calculate_quality_score(ue_idx)
                target_quality = self.target_quality_scores[ue_idx]
                
                # Reward for meeting quality requirements (lower BRISQUE = better)
                if achieved_quality <= target_quality:
                    quality_reward += weights['quality_bonus']  # Bonus for meeting quality
                else:
                    quality_reward -= abs(achieved_quality - target_quality) * weights['quality_penalty']  # Penalty for poor quality
        
        # 4. Efficiency reward - TFLOPs-based processing time optimization
        processing_efficiency_reward = 0.0
        for ue_idx in range(self.num_ues):
            denoising_steps = self.current_denoising_steps[ue_idx]
            image_size = self.ue_image_sizes[ue_idx]
            
            # Calculate processing time for current configuration
            processing_time = calculate_bs_processing_time(denoising_steps, image_size)
            
            # Quality-Processing Time Trade-off Reward
            achieved_quality = self._calculate_quality_score(ue_idx)
            target_quality = self.target_quality_scores[ue_idx]
            
            # Reward for meeting quality requirements (lower BRISQUE = better)
            quality_met = achieved_quality <= target_quality
            
            if quality_met:
                # Reward for meeting quality with fewer steps (more efficient)
                steps_efficiency = (20 - denoising_steps) / 19.0  # Normalize to [0,1]
                processing_efficiency_reward += steps_efficiency * weights['efficiency_weight']
                
                # Additional reward for good quality
                quality_bonus = max(0, (target_quality - achieved_quality) * weights['quality_bonus_weight'])
                processing_efficiency_reward += quality_bonus
            else:
                # Penalty for not meeting quality requirements
                quality_penalty = abs(achieved_quality - target_quality) * (weights['quality_penalty'] * 1.5)
                processing_efficiency_reward -= quality_penalty
                
                # Encourage more denoising steps if quality is not met
                if denoising_steps < 15:
                    processing_efficiency_reward -= (15 - denoising_steps) * weights['quality_penalty']
        
        # 5. BS utilization reward - penalize total processing time accumulation
        bs_utilization_penalty = -self.bs_total_processing_time * weights['bs_utilization_penalty']
        
        # 6. Positioning and connectivity
        positioning_penalty = 0.0
        if not self._can_communicate(self.uav_current_loc, self.bs_loc):
            positioning_penalty -= weights['positioning_penalty']
        
        connected_ues = 0
        for ue_idx in range(self.num_ues):
            if self._can_communicate(self.uav_current_loc, self.ue_locs[ue_idx]):
                connected_ues += 1
        connectivity_reward = connected_ues * weights['connectivity_weight']
        
        # 7. Time penalty (encourage episode efficiency)
        time_penalty = -self.current_step * weights['time_penalty']
        
        # 8. Completion bonus with processing efficiency
        completion_bonus = 0.0
        if np.all(self.ue_data_received):
            # Base completion bonus
            completion_bonus = weights['completion_bonus'] - self.current_step * weights['completion_step_penalty']
            
            # Additional bonus for good quality
            if quality_reward > 0:
                completion_bonus += quality_reward * weights['quality_bonus_weight']
            
            # Additional bonus for efficient BS usage
            avg_processing_time = self.bs_total_processing_time / max(self.num_ues, 1)
            if avg_processing_time < weights['efficiency_threshold']:
                efficiency_bonus = (weights['efficiency_threshold'] - avg_processing_time) * weights['efficiency_bonus']
                completion_bonus += efficiency_bonus
        
        total_reward = (completion_reward + upload_reward + processing_reward + 
                       quality_reward + processing_efficiency_reward + bs_utilization_penalty +
                       connectivity_reward + positioning_penalty + time_penalty + completion_bonus)
        
        return total_reward
    
    def _calculate_reward(self) -> float:
        raw_reward = self._calculate_raw_reward()
        
        if self.use_log_rewards:
            # Apply log transformation
            log_reward = self._signed_log(raw_reward)
            
            # Running normalization in log space
            self.reward_history.append(log_reward)
            if len(self.reward_history) > 1000:
                self.reward_history.pop(0)
            
            if len(self.reward_history) > 20:
                self.reward_mean = np.mean(self.reward_history)
                self.reward_std = np.std(self.reward_history) + 1e-8
                normalized_reward = (log_reward - self.reward_mean) / self.reward_std
            else:
                normalized_reward = log_reward / 10.0  # Simple scaling initially
            
            return normalized_reward
        else:
            return raw_reward
    
    def _is_done(self) -> bool:
        """Check if episode is finished"""
        # Done if all UEs received their data or max timesteps reached
        return (np.all(self.ue_data_received) or 
                self.current_step >= self.total_timesteps)

    def render(self):
        # Implement visualization if needed (e.g., Pygame or Matplotlib)
        pass

    def close(self):
        pass
    
    def _calculate_paper_reward(self) -> float:
        """
        Calculate paper-compliant reward with positioning and connectivity as rewards
        Based on paper formulation: R = ψ·I - λ_q·Q - λ_l·L - λ_e·E + λ_p·P + λ_c·C
        
        Modified from original paper:
        - λ_p: Positioning reward for good UAV placement (NOW POSITIVE)
        - λ_c: Connectivity reward for good communication links (NOW POSITIVE)
        """
        total_reward = 0.0
        
        # Completion indicator I: 1 if all quality requirements met, 0 otherwise
        completion_indicator = 0.0
        all_quality_met = True
        
        # Quality penalty Q: Accumulated quality violations
        quality_penalty = 0.0
        
        # Latency penalty L: Accumulated latency violations  
        latency_penalty = 0.0
        
        # Energy penalty E: Total energy consumption
        energy_penalty = 0.0
        
        # Process each UE for QoS metrics
        for ue_idx in range(self.num_ues):
            if self.ue_data_received[ue_idx]:
                # Quality assessment
                achieved_quality = self._calculate_quality_score(ue_idx)
                target_quality = self.target_quality_scores[ue_idx]
                
                # Quality penalty (BRISQUE: lower is better)
                if achieved_quality > target_quality:
                    quality_penalty += (achieved_quality - target_quality)
                    all_quality_met = False
                
                # Latency penalty
                if hasattr(self, 'ue_total_latencies') and len(self.ue_total_latencies) > ue_idx:
                    total_latency = self.ue_total_latencies[ue_idx]
                    if total_latency > self.tau_deadline:
                        latency_penalty += (total_latency - self.tau_deadline)
                        all_quality_met = False
                
                # Energy consumption
                denoising_steps = self.current_denoising_steps[ue_idx]
                image_size = self.ue_image_sizes[ue_idx]
                processing_time = calculate_bs_processing_time(denoising_steps, image_size)
                
                # Processing energy (simplified model)
                processing_energy = processing_time * 100.0  # Watts * seconds = Joules
                energy_penalty += processing_energy
                
                # Transmission energy (UE->UAV->BS->UAV->UE)
                uav_pos = self.uav_current_loc
                ue_pos = self.ue_locs[ue_idx]
                bs_pos = self.bs_loc
                
                # UE to UAV transmission
                ue_to_uav_time = self._calculate_transmission_time(ue_pos, uav_pos)
                if ue_to_uav_time != float('inf'):
                    transmission_energy = self.p_ue_up_w * ue_to_uav_time
                    energy_penalty += transmission_energy
                
                # UAV to BS transmission  
                uav_to_bs_time = self._calculate_transmission_time(uav_pos, bs_pos)
                if uav_to_bs_time != float('inf'):
                    transmission_energy = self.p_uav_up_w * uav_to_bs_time
                    energy_penalty += transmission_energy
                
                # BS to UAV transmission (result)
                bs_to_uav_time = self._calculate_transmission_time(bs_pos, uav_pos)
                if bs_to_uav_time != float('inf'):
                    transmission_energy = self.p_bs_down_w * bs_to_uav_time
                    energy_penalty += transmission_energy
                
                # UAV to UE transmission (final result)
                uav_to_ue_time = self._calculate_transmission_time(uav_pos, ue_pos)
                if uav_to_ue_time != float('inf'):
                    transmission_energy = self.p_uav_down_w * uav_to_ue_time
                    energy_penalty += transmission_energy
        
        # Positioning Reward (P): Good UAV placement gets positive reward
        positioning_reward = 0.0
        
        # Base positioning reward for maintaining BS connectivity
        if self._can_communicate(self.uav_current_loc, self.bs_loc):
            positioning_reward += 10.0  # Base bonus for BS connectivity
        
        # Connectivity Reward (C): Good communication links get positive reward  
        connectivity_reward = 0.0
        
        # Count connected UEs and give reward for each connection
        connected_ues = 0
        for ue_idx in range(self.num_ues):
            if self._can_communicate(self.uav_current_loc, self.ue_locs[ue_idx]):
                connected_ues += 1
        
        # Connectivity reward proportional to connected UEs
        connectivity_reward = connected_ues * 5.0  # 5.0 reward per connected UE
        
        # Completion bonus
        if all_quality_met and np.all(self.ue_data_received):
            completion_indicator = 1.0
        
        # Calculate final paper reward
        # R = ψ·I - λ_q·Q - λ_l·L - λ_e·E + λ_p·P + λ_c·C (positioning & connectivity as REWARDS)
        total_reward = (
            self.psi_bonus * completion_indicator
            - self.lambda_quality * quality_penalty
            - self.lambda_latency * latency_penalty  
            - self.lambda_energy * energy_penalty
            + self.lambda_positioning * positioning_reward  # NOW POSITIVE (reward)
            + self.lambda_connectivity * connectivity_reward  # NOW POSITIVE (reward)
        )
        
        return total_reward
    
    def get_action_space_info(self):
        """Get detailed information about action space design"""
        flight_constraints = get_flight_mode_constraints(self.flight_mode, self.config)
        
        info = {
            "design_approach": "velocity_based",
            "uav_movement": {
                "description": "Direct velocity control in m/s",
                "bounds": {
                    "horizontal_velocity": f"±{flight_constraints['max_horizontal_speed']} m/s",
                    "ascent_velocity": f"0 to {flight_constraints['max_ascent_speed']} m/s", 
                    "descent_velocity": f"-{flight_constraints['max_descent_speed']} to 0 m/s"
                },
                "max_displacement_per_step": {
                    "horizontal": f"{flight_constraints['max_horizontal_speed'] * self.env_step_duration} m",
                    "vertical_up": f"{flight_constraints['max_ascent_speed'] * self.env_step_duration} m",
                    "vertical_down": f"{flight_constraints['max_descent_speed'] * self.env_step_duration} m"
                }
            },
            "denoising_steps": {
                "description": "Number of denoising steps per UE",
                "range": f"{self.action_space['denoising_steps'].low[0]} to {self.action_space['denoising_steps'].high[0]} steps"
            },
            "advantages": [
                "Consistent with physics (velocity -> position)",
                "Respects realistic UAV flight constraints",  
                "No artificial 50m/step limitation",
                "Action space matches actual vehicle capabilities"
            ]
        }
        return info
    
    def print_action_space_comparison(self):
        """Print comparison between old delta-based and new velocity-based approach"""
        flight_constraints = get_flight_mode_constraints(self.flight_mode, self.config)
        
        print("="*80)
        print("🚁 UAV ACTION SPACE DESIGN COMPARISON")
        print("="*80)
        
        print("\n❌ OLD APPROACH (Delta-based):")
        print(f"   - Action: [Δx, Δy, Δz] with max ±50m, ±50m, ±20m per step")
        print(f"   - Issues:")
        print(f"     • Unrealistic: 50m/step = 50m/s with 1s step_duration")
        print(f"     • Ignores UAV physics constraints (max 17m/s horizontal)")
        print(f"     • Inconsistent with flight_mode speed limits")
        print(f"     • Action space bounds don't match vehicle capabilities")
        
        print("\n✅ NEW APPROACH (Velocity-based):")
        print(f"   - Action: [vx, vy, vz] in m/s")
        print(f"   - Bounds respect UAV constraints:")
        print(f"     • Horizontal: ±{flight_constraints['max_horizontal_speed']} m/s")
        print(f"     • Ascent: 0 to {flight_constraints['max_ascent_speed']} m/s")  
        print(f"     • Descent: -{flight_constraints['max_descent_speed']} to 0 m/s")
        print(f"   - Max displacement per step ({self.env_step_duration}s):")
        print(f"     • Horizontal: {flight_constraints['max_horizontal_speed'] * self.env_step_duration}m")
        print(f"     • Vertical: +{flight_constraints['max_ascent_speed'] * self.env_step_duration}m / -{flight_constraints['max_descent_speed'] * self.env_step_duration}m")
        
        print(f"\n🎯 BENEFITS:")
        print(f"   • Physically realistic and consistent")
        print(f"   • Velocity -> Position integration is standard physics")
        print(f"   • Action bounds match actual UAV capabilities")
        print(f"   • Training agents learn proper velocity control")
        print(f"   • No arbitrary displacement limits")
        
        print("="*80)
    
    def seed(self, seed=None):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)