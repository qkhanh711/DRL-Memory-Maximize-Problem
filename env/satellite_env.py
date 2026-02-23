
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SystemParameters:
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        logger.info(f"Initializing SystemParameters with seed={seed}")
        # ========== Satellite and Channel Parameters ==========
        self.num_mgu = 50  
        self.nu_sat = 20 
        self.nu_bs = 15   
        self.B_sat = 20e6  
        self.B_bs = 5e6   
        
        # ========== Physical Constants ==========
        self.c = 3e8     
        self.mu = 3.986e14 
        self.R_e = 6.371e6
        self.H_e = 765e3 
        
        # ========== Transmission Power ==========
        self.p_k = 10 ** (23 / 10) / 1000  
        self.p_sat = 10 ** (30 / 10) / 1000  
        self.p_bs = 10 ** (46 / 10) / 1000   
        
        # ========== Antenna Gains ==========
        self.G_sat_q = 10 ** (30 / 10)      
        self.G_user_kq = self.rng.uniform(5, 8, self.num_mgu)  
        self.G_user_kq = 10 ** (self.G_user_kq / 10)  
        self.G_gate = 10 ** (15 / 10)       
        self.sigma2_db = -134  
        self.sigma2 = 10 ** (self.sigma2_db / 10) / 1000  
        
        # ========== Frequency Parameters ==========
        self.Psi_sat = 2e9  
        self.Psi_gate = 2e9 
        
        # ========== Computation Parameters ==========
        self.f_tot_sat = 20e9  
        self.f_tot_bs = 50e9   
        self.C_sat = 1/32  
        self.C_bs = 1/32   
        self.kappa_sat = 1e-9  
        self.kappa_bs = 1e-9
        
        # ========== Data Parameters ==========
        self.D_up_k = self.rng.uniform(2000, 6000, self.num_mgu) * 1e3  
        self.chi_k = self.rng.uniform(300, 600, self.num_mgu)  
        self.tau_up = self.rng.uniform(1, 10, self.num_mgu)  
        self.tau_down = self.rng.uniform(1, 100, self.num_mgu)  
        
        # ========== Image Compression Parameters ==========
        self.M = 100  
        self.B_z = self.rng.uniform(1500, 4500, self.M) * 1e3  
        self.varsigma = 4  
        
        # ========== Resolution Parameters ==========
        self.r_min_pixels = 1280 * 720  
        self.r_max_pixels = 7680 * 4320 
        self.r_min = 0.0     
        self.r_max = 1.0    
        
        # ========== Satellite Geometry ==========
        self.zeta_k = np.deg2rad(self.rng.uniform(5.63, 85.94, self.num_mgu))
        self.zeta_gateway = np.deg2rad(self.rng.uniform(5.63, 85.94))
        
        # ========== Baseline Channel Gains ==========
        self.q_bs_k = 10 ** (self.rng.uniform(-105, -80, self.num_mgu) / 10)  
        self.d_bs_k = self.rng.uniform(500, 1000, self.num_mgu)  
        
        # ========== Constraint Parameters ==========
        self.R_min = 1e6   
        self.theta_max = 3
        # ========== Optimization Parameters ==========
        self.eta_earn = 1.0  
        self.eta_cons = 1.0 
        self.vartheta = 1.0
        self.omega_U = 0.5
        self.omega_E = 0.5   
        
        self.earning_types = ['rho1', 'rho2', 'rho3']
        self.earning_choice = self.rng.choice(self.earning_types, self.num_mgu)
        self.alpha_1 = 4.268
        self.beta_1 = 0.2714
        self.alpha_2 = 1.159
        self.beta_2 = 91.92
        self.alpha_3 = 89.95  
        self.beta_3 = 4.732  


class SatelliteGeometry:
    def calculate_satellite_distance(params: SystemParameters, user_idx: int) -> float:
        zeta = params.zeta_k[user_idx]
        R_e = params.R_e
        H_e = params.H_e
        cos_zeta = np.cos(zeta)
        xi = np.arccos((R_e / (R_e + H_e)) * cos_zeta) - zeta
        return np.sqrt(
            R_e**2 + (R_e + H_e)**2 - 
            2 * R_e * (R_e + H_e) * np.cos(xi)
        )
        
    def calculate_max_communication_time(params: SystemParameters, user_idx: int) -> float:
        zeta = params.zeta_k[user_idx]
        R_e = params.R_e
        H_e = params.H_e
        mu = params.mu
        cos_zeta = np.cos(zeta)
        xi = np.arccos((R_e / (R_e + H_e)) * cos_zeta) - zeta
        G_k = 2 * (R_e + H_e) * xi
        omega = np.sqrt(mu / (R_e + H_e))
        T_max = G_k / omega
        return T_max
    
    def channel_gain_user_to_satellite(params: SystemParameters, user_idx: int, 
                                      channel_idx: int) -> float:
        d_sat = SatelliteGeometry.calculate_satellite_distance(params, user_idx)
        c = params.c
        Psi = params.Psi_sat
        L = (c / (4 * np.pi * d_sat * Psi)) ** 2
        mu_fading = 1.0 + np.abs(params.rng.randn()) * 0.1 
        q_sat = L * params.G_sat_q * params.G_user_kq[user_idx] * (mu_fading ** 2)
        return q_sat


class CommunicationModel:
    def uplink_rate_satellite(params, user_idx, channel_idx, num_users_on_channel):
        if num_users_on_channel == 0:
            return 1e-9  
        
        B = params.B_sat
        p = params.p_k
        sigma2 = params.sigma2
        q_sat = SatelliteGeometry.channel_gain_user_to_satellite(params, user_idx, channel_idx)
        snr = (q_sat * p) / (B * sigma2 / num_users_on_channel)
        rate = (B / num_users_on_channel) * np.log2(1 + snr)
        return rate
    
    def downlink_rate_satellite(params, user_idx, channel_idx, num_users_on_channel):
        if num_users_on_channel == 0:
            return 1e-9  
        
        B = params.B_sat
        p = params.p_sat
        sigma2 = params.sigma2
        q_sat = SatelliteGeometry.channel_gain_user_to_satellite(params, user_idx, channel_idx)
        snr = (q_sat * p) / (B * sigma2 / num_users_on_channel)
        rate = (B / num_users_on_channel) * np.log2(1 + snr)
        return rate
    
    def uplink_rate_bs(params, user_idx, num_users_on_channel):
        if num_users_on_channel == 0:
            return 1e-9  
        
        B = params.B_bs
        p = params.p_k
        sigma2 = params.sigma2
        q_bs = params.q_bs_k[user_idx]
        snr = (q_bs * p) / (B * sigma2 / num_users_on_channel)
        rate = (B / num_users_on_channel) * np.log2(1 + snr)
        return rate
    
    def downlink_rate_bs(params, user_idx, num_users_on_channel):
        if num_users_on_channel == 0:
            return 1e-9  
        
        B = params.B_bs
        p = params.p_bs
        sigma2 = params.sigma2
        q_bs = params.q_bs_k[user_idx]
        snr = (q_bs * p) / (B * sigma2 / num_users_on_channel)
        rate = (B / num_users_on_channel) * np.log2(1 + snr)
        return rate
    
    def gateway_transmission_rate(params, theta):
        # d_gateway = SatelliteGeometry.calculate_satellite_distance(params, -1)  
        
        # d_gateway = np.sqrt(params.R_e**2 + (params.R_e + params.H_e)**2 - 
        #                    2 * params.R_e * (params.R_e + params.H_e) * 
        #                    np.cos(params.zeta_gateway))

        xi = np.arccos((params.R_e / (params.R_e + params.H_e)) * np.cos(params.zeta_gateway)) - params.zeta_gateway
        d_gateway = np.sqrt(
        params.R_e**2 + (params.R_e + params.H_e)**2 - 2*params.R_e*(params.R_e + params.H_e)*np.cos(xi)
        )
        L_gate = (params.c / (4 * np.pi * d_gateway * params.Psi_gate)) ** 2 * np.random.uniform(0.01, 1)**2
        q_gate = L_gate * params.G_sat_q * params.G_gate
        B_tilde = params.B_sat  
        p_sat = params.p_sat
        sigma2 = params.sigma2
        snr = (q_gate * p_sat) / (B_tilde * sigma2)
        rate = B_tilde * np.log2(1 + snr)
        return rate


class ComputationModel:
    def downlink_data_size(params, user_idx, resolution):
        r_k_pixels = params.r_min_pixels + resolution * (params.r_max_pixels - params.r_min_pixels)
        bits_total = 24 * 2 * r_k_pixels
        D_down = bits_total / params.chi_k[user_idx]
        return D_down
    
    def satellite_computation_latency(params, user_idx, resolution, allocated_capacity):
        D_up = params.D_up_k[user_idx]
        D_down = ComputationModel.downlink_data_size(params, user_idx, resolution)
        tau_up = params.tau_up[user_idx]
        tau_down = params.tau_down[user_idx]
        C_sat = params.C_sat
        cycles_needed = tau_up * D_up + tau_down * D_down
        T_comp = cycles_needed / (C_sat * allocated_capacity)
        return T_comp
    
    def satellite_computation_energy(params, user_idx, resolution, allocated_capacity):
        D_up = params.D_up_k[user_idx]
        D_down = ComputationModel.downlink_data_size(params, user_idx, resolution)
        tau_up = params.tau_up[user_idx]
        tau_down = params.tau_down[user_idx]
        C_sat = params.C_sat
        kappa = params.kappa_sat
        f_GHz = allocated_capacity / 1e9
        cycles_needed = tau_up * D_up + tau_down * D_down
        E_comp = kappa * (f_GHz ** 2) * cycles_needed * C_sat
        return E_comp
    
    def bs_computation_cost(params, user_idx, resolution, allocated_capacity):
        D_up = params.D_up_k[user_idx]
        D_down = ComputationModel.downlink_data_size(params, user_idx, resolution)
        tau_up = params.tau_up[user_idx]
        tau_down = params.tau_down[user_idx]
        C_bs = params.C_bs
        cycles_needed = tau_up * D_up + tau_down * D_down
        T_comp = cycles_needed / (C_bs * allocated_capacity)
        kappa = params.kappa_bs
        f_GHz = allocated_capacity / 1e9
        E_comp = kappa * (f_GHz ** 2) * cycles_needed * C_bs

        return T_comp, E_comp


class CompressionModel:
    def cpu_cycles_per_bit(theta, varsigma ):
        return np.exp(varsigma * theta) - np.exp(varsigma)
    
    def compression_latency(params, theta, allocated_capacity):
        
        F = CompressionModel.cpu_cycles_per_bit(theta, params.varsigma)
        total_cycles = np.sum(params.B_z) * F
        
        T_cpr = total_cycles / (allocated_capacity)
        
        return T_cpr
    
    def compression_energy(params, theta, allocated_capacity):
        F = CompressionModel.cpu_cycles_per_bit(theta, params.varsigma)
        total_bits = np.sum(params.B_z)
        f_GHz = allocated_capacity / 1e9
        E_cpr = params.kappa_sat * (f_GHz ** 2) * params.C_sat * total_bits * F
        
        return E_cpr
    
    def gateway_transmission_cost(params, theta):
        compressed_size = np.sum(params.B_z) / theta
        
        tilde_R = CommunicationModel.gateway_transmission_rate(params, theta)
        # print(f"Gateway transmission rate: {tilde_R}")
        
        tilde_T = compressed_size / tilde_R
        tilde_E = params.p_sat * (compressed_size / tilde_R)
        
        return tilde_T, tilde_E


class EarningModel:
    
    def earning_rho1(params, user_idx, resolution, downlink_rate):
        r_norm = (resolution * 0.5)  
        rate_max = 1e8  
        rate_norm = (downlink_rate / rate_max) * 0.5
        rate_norm = np.clip(rate_norm, 0, 0.5)
        earning = params.alpha_1 * (r_norm + rate_norm) ** params.beta_1
        return earning
    
    def earning_rho2(params, user_idx, resolution, downlink_rate):
        r_norm = (resolution * 0.5)
        rate_max = 1e8
        rate_norm = (downlink_rate / rate_max) * 0.5
        rate_norm = np.clip(rate_norm, 0, 0.5)
        earning = params.alpha_2 * np.log(1 + params.beta_2 * (r_norm + rate_norm))
        return earning
    
    def earning_rho3(params, user_idx, resolution, downlink_rate):
        r_norm = (resolution * 0.5)
        rate_max = 1e8
        rate_norm = (downlink_rate / rate_max) * 0.5
        rate_norm = np.clip(rate_norm, 0, 0.5)
        earning = params.alpha_3 * (1 - np.exp(-params.beta_3 * (r_norm + rate_norm)))
        return earning
    
    def calculate_earning(params, user_idx, resolution, downlink_rate):
        earning_type = params.earning_choice[user_idx]
        if earning_type == 'rho1':
            return EarningModel.earning_rho1(params, user_idx, resolution, downlink_rate)
        elif earning_type == 'rho2':
            return EarningModel.earning_rho2(params, user_idx, resolution, downlink_rate)
        elif earning_type == 'rho3':
            return EarningModel.earning_rho3(params, user_idx, resolution, downlink_rate)
        else:
            return 0.0


class SatelliteMECEnvironment:
    def __init__(self, seed = 42):
        self.params = SystemParameters(seed)
        self.step_count = 0
        self.offloading_decisions = np.zeros(self.params.num_mgu, dtype=int)  # 0=satellite, 1=BS
        self.channel_assignments = np.zeros(self.params.num_mgu, dtype=int)  
        self.resolutions = np.ones(self.params.num_mgu) * 0.5  
        self.compression_ratio = 1.0
        
        self._cache_sat_distances = None
        self._cache_channel_gains = None
        logger.info(f"Initialized Satellite-MEC Environment with {self.params.num_mgu} MGUs")
        logger.debug(f"System parameters: nu_sat={self.params.nu_sat}, nu_bs={self.params.nu_bs}")
    
    def set_offloading_decisions(self, decisions, channels):
        self.offloading_decisions = decisions
        self.channel_assignments = channels
        self._cache_sat_distances = None
        self._cache_channel_gains = None
        logger.debug(f"Set offloading: {np.sum(decisions==0)} satellite, {np.sum(decisions==1)} BS users")
    
    def set_resolutions(self, resolutions):
        self.resolutions = np.clip(resolutions, self.params.r_min, self.params.r_max)
        logger.debug(f"Set resolutions: min={np.min(resolutions):.3f}, max={np.max(resolutions):.3f}, avg={np.mean(resolutions):.3f}")
    
    def set_compression_ratio(self, theta):
        self.compression_ratio = np.clip(theta, 1.0, self.params.theta_max)
        logger.debug(f"Set compression ratio: {self.compression_ratio:.3f}")
    
    def get_users_on_satellite_channel(self, channel_idx):
        return np.where((self.offloading_decisions == 0) & 
                       (self.channel_assignments == channel_idx))[0].tolist()
    
    def get_users_on_bs_channel(self, channel_idx):
        return np.where((self.offloading_decisions == 1) & 
                       (self.channel_assignments == channel_idx))[0].tolist()
    
    def calculate_satellite_offloading_cost(self, user_idx, 
                                          channel_idx):
        resolution = self.resolutions[user_idx]
        users_on_channel = self.get_users_on_satellite_channel(channel_idx)
        num_users = len(users_on_channel)
        
        
        D_up = self.params.D_up_k[user_idx]
        D_down = ComputationModel.downlink_data_size(self.params, user_idx, resolution)
        
        R_up = CommunicationModel.uplink_rate_satellite(self.params, user_idx, 
                                                       channel_idx, num_users)
        R_down = CommunicationModel.downlink_rate_satellite(self.params, user_idx,
                                                           channel_idx, num_users)
        
        # print(f"User {user_idx} Satellite Offloading - R_up: {R_up}, R_down: {R_down}")
        
        T_up = D_up / R_up
        T_down = D_down / R_down
        E_up = self.params.p_k * (D_up / R_up)
        E_down = self.params.p_sat * (D_down / R_down)
        
        d_sat = SatelliteGeometry.calculate_satellite_distance(self.params, user_idx)
        T_prop = 2 * d_sat / self.params.c
        
        N_sat_users = np.sum(self.offloading_decisions == 0)
        allocated_capacity = self.params.f_tot_sat / max(N_sat_users, 1)

        T_comp = ComputationModel.satellite_computation_latency(self.params, user_idx,
                                                               resolution, allocated_capacity)
        E_comp = ComputationModel.satellite_computation_energy(self.params, user_idx,
                                                              resolution, allocated_capacity)
        
        total_energy = E_up + E_down + E_comp
        total_latency = T_up + T_down + T_comp + T_prop
        # print(f"Satellite Offloading - User {user_idx}: E_up={E_up:.4f}, E_down={E_down:.4f}, E_comp={E_comp:.4f}, Total_E={total_energy:.4f}")
        
        return total_energy, total_latency
    
    def calculate_bs_offloading_cost(self, user_idx, channel_idx):
        resolution = self.resolutions[user_idx]
        users_on_channel = self.get_users_on_bs_channel(channel_idx)
        num_users = len(users_on_channel)
        
        # print(f"Number of users on BS channel {channel_idx}: {num_users}")
        
        D_up = self.params.D_up_k[user_idx]
        D_down = ComputationModel.downlink_data_size(self.params, user_idx, resolution)
        
        R_up = CommunicationModel.uplink_rate_bs(self.params, user_idx, num_users)
        R_down = CommunicationModel.downlink_rate_bs(self.params, user_idx, num_users)
        
        if R_up < self.params.R_min or R_down < self.params.R_min:
            violation = max(
            self.params.R_min - R_up,
            self.params.R_min - R_down
            )
            penalty = 1e3 * violation / self.params.R_min
            return penalty, penalty
        
        T_up = D_up / R_up
        T_down = D_down / R_down
        E_up = self.params.p_k * (D_up / R_up)
        E_down = self.params.p_bs * (D_down / R_down)
        
        allocated_capacity = self.params.f_tot_bs / (self.params.nu_bs * 10)  
        T_comp, E_comp = ComputationModel.bs_computation_cost(self.params, user_idx,
                                                             resolution, allocated_capacity)
        
        total_energy = E_up + E_down + E_comp
        total_latency = T_up + T_down + T_comp
        # print(f"BS Offloading - User {user_idx}: E_up={E_up:.4f}, E_down={E_down:.4f}, E_comp={E_comp:.4f}, Total_E={total_energy:.4f}") 
        return total_energy, total_latency
    
    def calculate_utility(self, user_idx):
        resolution = self.resolutions[user_idx]
        offload_type = self.offloading_decisions[user_idx]
        # print(f"Calculating utility for User {user_idx}, Offload Type: {'Satellite' if offload_type == 0 else 'BS'}")
        # print()
        channel_idx = self.channel_assignments[user_idx]
        if offload_type == 0:  # Satellite
            offload_cost, _ = self.calculate_satellite_offloading_cost(user_idx, channel_idx)
            R_down = CommunicationModel.downlink_rate_satellite(self.params, user_idx,
                                                               channel_idx, len(self.get_users_on_satellite_channel(channel_idx)))
        elif offload_type == 1:  # BS
            offload_cost, _ = self.calculate_bs_offloading_cost(user_idx, channel_idx)
            R_down = CommunicationModel.downlink_rate_bs(self.params, user_idx,
                                                        len(self.get_users_on_bs_channel(channel_idx)))
        
        earning = EarningModel.calculate_earning(self.params, user_idx, resolution, R_down)
        
        earnings_norm = earning / 5.5
        energy_norm = offload_cost / 10.0
        utility = earnings_norm - self.params.vartheta * energy_norm
        
        return utility
    
    def calculate_all_user_utilities(self):
        logger.debug("Computing all user utilities (vectorized)")
        
        utilities = np.zeros(self.params.num_mgu)
        sat_mask = (self.offloading_decisions == 0)
        bs_mask = (self.offloading_decisions == 1)
        
        if np.any(sat_mask):
            sat_indices = np.where(sat_mask)[0]
            for user_idx in sat_indices:
                channel_idx = self.channel_assignments[user_idx]
                offload_cost, _ = self.calculate_satellite_offloading_cost(user_idx, channel_idx)
                R_down = CommunicationModel.downlink_rate_satellite(
                    self.params, user_idx, channel_idx, 
                    len(self.get_users_on_satellite_channel(channel_idx))
                )
                earning = EarningModel.calculate_earning(
                    self.params, user_idx, self.resolutions[user_idx], R_down
                )
                utilities[user_idx] = earning / 5.5 - self.params.vartheta * offload_cost / 10.0
        
        if np.any(bs_mask):
            bs_indices = np.where(bs_mask)[0]
            for user_idx in bs_indices:
                channel_idx = self.channel_assignments[user_idx]
                offload_cost, _ = self.calculate_bs_offloading_cost(user_idx, channel_idx)
                R_down = CommunicationModel.downlink_rate_bs(
                    self.params, user_idx,
                    len(self.get_users_on_bs_channel(channel_idx))
                )
                earning = EarningModel.calculate_earning(
                    self.params, user_idx, self.resolutions[user_idx], R_down
                )
                utilities[user_idx] = earning / 5.5 - self.params.vartheta * offload_cost / 10.0
        
        logger.debug(f"Utilities: min={np.min(utilities):.4f}, max={np.max(utilities):.4f}, mean={np.mean(utilities):.4f}")
        return utilities
    
    def calculate_total_utility(self):
        logger.debug("Calculating total utility (vectorized)")
        
        user_utilities = self.calculate_all_user_utilities()
        total_utility = np.sum(user_utilities)
        
        E_cpr = CompressionModel.compression_energy(self.params, self.compression_ratio, 
                                                   self.params.f_tot_sat * 0.1)
        _, E_trans = CompressionModel.gateway_transmission_cost(self.params, 
                                                               self.compression_ratio)
        
        compression_energy_total = E_cpr + E_trans
        compression_cost_normalized = compression_energy_total / 1000.0  
        
        avg_user_utility = total_utility / self.params.num_mgu
        total_utility = (self.params.omega_U * avg_user_utility - 
                        self.params.omega_E * compression_cost_normalized)
        
        logger.debug(f"Total utility: {total_utility:.4f} (avg_user={avg_user_utility:.4f}, compression_cost={compression_cost_normalized:.4f})")
        return total_utility
    
    def get_state(self):
        return {
            'offloading_decisions': self.offloading_decisions.copy(),
            'channel_assignments': self.channel_assignments.copy(),
            'resolutions': self.resolutions.copy(),
            'compression_ratio': self.compression_ratio,
            'step': self.step_count
        }
    
    def get_statistics(self):
        """Get system statistics (vectorized)"""
        logger.debug("Computing statistics")
        
        stats = {
            'total_utility': self.calculate_total_utility(),
            'num_satellite_users': int(np.sum(self.offloading_decisions == 0)),
            'num_bs_users': int(np.sum(self.offloading_decisions == 1)),
            'avg_resolution': float(np.mean(self.resolutions)),
            'compression_ratio': float(self.compression_ratio)
        }
        
        user_utilities = self.calculate_all_user_utilities()
        
        stats['per_user_utilities'] = user_utilities.tolist()
        stats['avg_user_utility'] = float(np.mean(user_utilities))
        
        logger.debug(f"Stats: total_utility={stats['total_utility']:.4f}, sat_users={stats['num_satellite_users']}, bs_users={stats['num_bs_users']}")
        return stats
    
    def step(self, action):
        self.step_count += 1
        # self.apply_action(action)
        reward = self.calculate_total_utility()
        obs = self.get_state()
        done = False
        info = self.get_statistics()
        return obs, reward, done, info

    
    def reset(self):
        self.step_count = 0
        self.offloading_decisions = np.zeros(self.params.num_mgu, dtype=int)
        self.channel_assignments = np.zeros(self.params.num_mgu, dtype=int)
        self.resolutions = np.ones(self.params.num_mgu) * 0.5
        self.compression_ratio = 1.0
        
        self._cache_sat_distances = None
        self._cache_channel_gains = None
        
        return self.get_state()


if __name__ == "__main__":
    env = SatelliteMECEnvironment(seed=42)
    
    decisions = np.random.randint(0, 2, env.params.num_mgu)
    print(f"Offloading decisions: {decisions}")
    channels = np.random.randint(0, 20, env.params.num_mgu)
    env.set_offloading_decisions(decisions, channels)
    
    env.set_resolutions(np.random.uniform(0.2, 0.8, env.params.num_mgu)) 
    env.set_compression_ratio(1.5)
    
    stats = env.get_statistics()
    print(f"Total System Utility: {stats['total_utility']:.4f}")
    print(f"Satellite Users: {stats['num_satellite_users']}/{env.params.num_mgu}")
    print(f"Average Resolution: {stats['avg_resolution']:.4f}")
    print(f"Average User Utility: {stats['avg_user_utility']:.4f}")