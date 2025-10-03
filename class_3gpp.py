import numpy as np
import random
# from plot_distributions import PlotChannelModel

LIGHT_SPEED = 299792458 #m/s

class Channel3GPP:
    def __init__(self, 
             scenario: str, 
             frequency_ghz: float, 
             n_paths: int, 
             rx_velocity_mps: float, 
             rx_azimuth_deg: float = 90.0, 
             rx_elevation_deg: float = 0.0):
        """
        Initializes the channel object with the basic simulation parameters.
        """
        if scenario not in ["umi_los", "umi_nlos", "uma_los", "uma_nlos"]:
            raise ValueError(f"Invalid scenario: '{scenario}'")

        self.scenario = scenario
        self.frequency_ghz = frequency_ghz
        self.n_paths = n_paths
        self.rx_velocity_mps = rx_velocity_mps
        self.rx_azimuth_deg = rx_azimuth_deg
        self.rx_elevation_deg = rx_elevation_deg

        # Attributes to be calculated during the simulation
        # Large-scale parameters
        self.sigma_tau = None
        self.kr_factor = None
        self.sigma_theta = None
        # Small-scale parameters
        self.multipath_delays = None
        self.multipath_powers = None
        self.multipath_azimuth_angles = None
        self.multipath_elevation_angles = None

        self.arrival_directions = None

        # Define statistical parameters based on the scenario
        self._define_statistical_parameters()

    def generate_channel(self):
        """
        Performs a single channel realization, calculating all its parameters.
        """
        # 1. Generate large-scale parameters
        self.sigma_tau = self._generate_gaussian_distribution(self.mean_sigma_tau, self.std_dev_sigma_tau, 1, "log_to_linear")
        if self.scenario in {"umi_nlos", "uma_nlos"}:
            self.kr_factor = 0.0
        else:
            self.kr_factor = self._generate_gaussian_distribution(self.mean_kr, self.std_dev_kr, 1, "db_to_linear")[0]
        self.sigma_theta = self._generate_gaussian_distribution(self.mean_sigma_theta, self.std_dev_sigma_theta, 1, "log_deg_to_linear_rad")
        self.sigma_phi = self._generate_gaussian_distribution(self.mean_sigma_phi, self.std_dev_sigma_phi, 1, "log_deg_to_linear_rad")

        # 2. Generate small-scale parameters
        self.multipath_delays = self._calculate_multipath_delays()
        self.multipath_powers = self._calculate_multipath_powers()
        self.multipath_azimuth_angles = self._calculate_multipath_azimuth_angles()
        self.multipath_elevation_angles = self._calculate_multipath_elevation_angles()

        self.doppler_shifts = self._calculate_doppler_shift()

        print(f"--> Channel realization generated successfully.")
        print(f"    - Delay Spread (σ_τ): {self.sigma_tau[0]*1e9:.2f} ns")
        print(f"    - Rice Factor (K_R): {self.kr_factor} (linear)")
        print(f"    - LoS Power [% of Total Power]: {self.multipath_powers[0]*100:.2f}%")

    def _define_statistical_parameters(self):
        """Private method to load the channel statistics."""
        # This logic is the same as your define_parametros_cenario function
        # but stores the values as object attributes (self.*)
        if self.scenario == "umi_los":
            # delay spread statistics (στ) [log]
            self.mean_sigma_tau = -0.24 * np.log10(1 + self.frequency_ghz) - 7.14
            self.std_dev_sigma_tau = 0.38
            self.r_tau = 3.0
            # shadowing statistics (σξ) [dB]
            self.std_dev_sigma_xi = 4.0
            self.mean_kr = 9.0
            self.std_dev_kr = 5.0
            # azimuth angle spread statistics (σθ) [log]
            self.mean_sigma_theta = - 0.08 * np.log10(1 + self.frequency_ghz) + 1.73
            self.std_dev_sigma_theta = 0.014 * np.log10(1 + self.frequency_ghz) + 0.28
            # elevation angle spread statistics (σφ) [log]
            self.mean_sigma_phi = - 0.1 * np.log10(1 + self.frequency_ghz) + 0.73
            self.std_dev_sigma_phi = - 0.04 * np.log10(1 + self.frequency_ghz) + 0.34

        elif self.scenario == "umi_nlos":
            # delay spread statistics (στ) [log]
            self.mean_sigma_tau = -0.24 * np.log10(1 + self.frequency_ghz) - 6.83
            self.std_dev_sigma_tau = -0.16 * np.log10(1 + self.frequency_ghz) + 0.28
            self.r_tau = 2.1
            # shadowing statistics (σξ) [dB]
            self.std_dev_sigma_xi = 7.82
            self.mean_kr = 0.0
            self.std_dev_kr = 0.0
            # azimuth angle spread statistics (σθ) [log]
            self.mean_sigma_theta = - 0.08 * np.log10(1 + self.frequency_ghz) + 1.81
            self.std_dev_sigma_theta = 0.05 * np.log10(1 + self.frequency_ghz) + 0.3
            # elevation angle spread statistics (σφ) [log]
            self.mean_sigma_phi = - 0.04 * np.log10(1 + self.frequency_ghz) + 0.92
            self.std_dev_sigma_phi = - 0.07 * np.log10(1 + self.frequency_ghz) + 0.41

        elif self.scenario == "uma_los":
            # delay spread statistics (στ) [log]
            self.mean_sigma_tau = -0.0963 * np.log10(1 + self.frequency_ghz) - 6.955
            self.std_dev_sigma_tau = 0.66
            self.r_tau = 2.5
            # shadowing statistics (σξ) [dB]
            self.std_dev_sigma_xi = 4.0
            self.mean_kr = 9.0
            self.std_dev_kr = 3.5
            # azimuth angle spread statistics (σθ) [log]
            self.mean_sigma_theta = 1.81
            self.std_dev_sigma_theta = 0.2
            # elevation angle spread statistics (σφ) [log]
            self.mean_sigma_phi = 0.95
            self.std_dev_sigma_phi = 0.16

        elif self.scenario == "uma_nlos":
            # delay spread statistics (στ) [log]
            self.mean_sigma_tau = -0.204 * np.log10(1 + self.frequency_ghz) - 6.28
            self.std_dev_sigma_tau = 0.39
            self.r_tau = 2.3
            # shadowing statistics (σξ) [dB]
            self.std_dev_sigma_xi = 6.0
            self.mean_kr = 0.0
            self.std_dev_kr = 0.0
            # azimuth angle spread statistics (σθ) [log]
            self.mean_sigma_theta = - 0.27 * np.log10(self.frequency_ghz) + 2.08
            self.std_dev_sigma_theta = 0.11
            # elevation angle spread statistics (σφ) [log]
            self.mean_sigma_phi = - 0.3236 * np.log10(self.frequency_ghz) + 1.512
            self.std_dev_sigma_phi = 0.16

    def _generate_gaussian_distribution(self, mean: float, std_dev: float, size: int, scale: str | None = None) -> np.ndarray:
        g = np.random.normal(mean, std_dev, size)
        if scale == "log_to_linear":
            return 10**g
        elif scale == "db_to_linear":
            return 10**(g / 10)
        elif scale == "log_deg_to_linear_rad":
            return (10**g) * (np.pi / 180)
        else:
            return g

    def _calculate_multipath_delays(self):
        mean_tau_n = self.r_tau * self.sigma_tau
        raw_delays = np.random.exponential(mean_tau_n, self.n_paths)
        norm_multipath_delays = raw_delays - np.min(raw_delays)
        return np.sort(norm_multipath_delays)

    def _calculate_multipath_powers(self):
        xi_n = np.random.normal(0, self.std_dev_sigma_xi, self.n_paths) # db
        preliminary_powers = np.exp(-self.multipath_delays * ((self.r_tau - 1) / (self.r_tau * self.sigma_tau))) * 10**(-xi_n / 10)

        nlos_preliminary_powers = preliminary_powers[1:]
        total_dispersed_power = np.sum(nlos_preliminary_powers)

        los_power = self.kr_factor / (self.kr_factor + 1)

        norm_dispersed_power = (1 / (self.kr_factor + 1)) * (nlos_preliminary_powers / total_dispersed_power)

        multipath_powers = np.concatenate(([los_power], norm_dispersed_power))
        return multipath_powers

    def _generate_initial_azimuth_angles(self):
        power_ratio = self.multipath_powers / np.max(self.multipath_powers)
        return 1.42 * self.sigma_theta * np.sqrt(-np.log(power_ratio))

    def _generate_initial_elevation_angles(self):
        power_ratio = self.multipath_powers / np.max(self.multipath_powers)
        return - self.sigma_phi * np.log(power_ratio)

    def _generate_random_signals(self):
        return np.random.choice([-1, 1], size=self.n_paths)

    def _calculate_multipath_azimuth_angles(self):
        final_theta = np.zeros(self.n_paths)
        un_signals = self._generate_random_signals()
        initial_theta = self._generate_initial_azimuth_angles()
        yn_fluctuations = self._generate_gaussian_distribution(0, self.sigma_theta / 7, self.n_paths, "log_deg_to_linear_rad")
        final_theta = (un_signals * initial_theta) + yn_fluctuations
        if self.scenario in {"umi_los", "uma_los"}:
            final_theta = final_theta - initial_theta[0]
        return final_theta * (180 / np.pi)  # Convert to degrees for output
    
    def _calculate_multipath_elevation_angles(self):
        un_signals = self._generate_random_signals()
        initial_phi = self._generate_initial_elevation_angles()
        yn_fluctuations = self._generate_gaussian_distribution(0, self.sigma_phi / 7, self.n_paths, "log_deg_to_linear_rad")
        mean_phi = np.deg2rad(45)
        final_phi = (un_signals * initial_phi) + yn_fluctuations
        if self.scenario in {"umi_los", "uma_los"}:
            final_phi = final_phi - initial_phi[0] + mean_phi
        return final_phi * (180 / np.pi)  # Convert to degrees for output

    def _calculate_arrival_directions(self):
        azimuth_rad = np.deg2rad(self.multipath_azimuth_angles)
        elevation_rad = np.deg2rad(self.multipath_elevation_angles)

        x = np.cos(azimuth_rad) * np.sin(elevation_rad)
        y = np.sin(azimuth_rad) * np.sin(elevation_rad)
        z = np.cos(elevation_rad)

        #TODO: garantir que estas coordenadas tenham módulo = 1

        return np.array([x, y, z])

    def _calculate_doppler_shift(self):
        """
        Calcula o desvio Doppler para cada percurso (multipath component).
        """
        # 1. Converter a frequência da portadora de GHz para Hz (fator 1e9)
        frequency_hz = self.frequency_ghz * 1e9

        # 2. Calcular o comprimento de onda (lambda = c / f)
        wave_len = LIGHT_SPEED / frequency_hz

        # 3. Obter o vetor de direção unitário do movimento do receptor
        rx_azimuth_rad = np.deg2rad(self.rx_azimuth_deg)
        rx_elevation_rad = np.deg2rad(self.rx_elevation_deg)
        
        x = np.cos(rx_azimuth_rad) * np.sin(rx_elevation_rad)
        y = np.sin(rx_azimuth_rad) * np.sin(rx_elevation_rad)
        z = np.cos(rx_elevation_rad)
        rx_direction_vector = np.array([x, y, z])

        # 4. Obter os vetores de direção de chegada para todos os percursos (matriz 3xN)
        multipath_arrival_vectors = self._calculate_arrival_directions()

        # 5. Calcular o termo v/lambda (componente máxima do desvio Doppler)
        max_doppler_shift = self.rx_velocity_mps / wave_len
        
        # 6. Calcular o cosseno do ângulo entre a direção do receptor e cada percurso
        # O resultado é um array com N elementos, um para cada percurso.
        cos_angles = np.dot(rx_direction_vector, multipath_arrival_vectors)
        
        # 7. Calcular o desvio Doppler final para cada percurso
        # O resultado é um array com n_paths elementos.
        doppler_shifts = max_doppler_shift * cos_angles
        
        return doppler_shifts
# if __name__ == '__main__':
# TODO: create unit tests for all functions