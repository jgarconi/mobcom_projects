import numpy as np
import random
from plot_distributions import PlotChannelModel

# TODO:
# - [ ] verify if statistical parameters are correctly defined
# - [ ] verify if multipath angles are correctly calculated (are they in degrees or radians? is the theta[0] the right reference for LoS?)
# - [ ] verify if "log_deg_to_linear_rad" is correctly implemented at _generate_gaussian_distribution
# - [ ] verify if the random seed for mean_phi should be set for reproducibility
# - [ ] verify if the kr_factor is correctly calculated when mean_kr and std_dev_kr are zero (nLoS scenarios)


class Channel3GPP:
    def __init__(self, scenario: str, frequency_ghz: float, n_paths: int):
        """
        Initializes the channel object with the basic simulation parameters.
        """
        if scenario not in ["umi_los", "umi_nlos", "uma_los", "uma_nlos"]:
            raise ValueError(f"Invalid scenario: '{scenario}'")

        self.scenario = scenario
        self.frequency_ghz = frequency_ghz
        self.n_paths = n_paths

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

        # Define statistical parameters based on the scenario
        self._define_statistical_parameters()

    def _define_statistical_parameters(self):
        """Private method to load the channel statistics."""
        # This logic is the same as your define_parametros_cenario function
        # but stores the values as object attributes (self.*)
        if self.scenario == "umi_los":
            # delay spread statistics (στ) [log]
            self.mean_sigma_tau = -0.24 * np.log10(1 + self.frequency_ghz) - 7.14
            self.std_dev_sigma_tau = 0.38
            self.r_tau = 3
            # shadowing statistics (σξ) [dB]
            self.std_dev_sigma_xi = 4
            self.mean_kr = 9
            self.std_dev_kr = 5
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
            self.mean_kr = 0
            self.std_dev_kr = 0
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
            self.std_dev_sigma_xi = 4
            self.mean_kr = 9
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
            self.std_dev_sigma_xi = 6
            self.mean_kr = 0
            self.std_dev_kr = 0
            # azimuth angle spread statistics (σθ) [log]
            self.mean_sigma_theta = - 0.27 * np.log10(self.frequency_ghz) + 2.08
            self.std_dev_sigma_theta = 0.11
            # elevation angle spread statistics (σφ) [log]
            self.mean_sigma_phi = - 0.3236 * np.log10(self.frequency_ghz) + 1.512
            self.std_dev_sigma_phi = 0.16


    def generate_channel(self):
        """
        Performs a single channel realization, calculating all its parameters.
        """
        # 1. Generate large-scale parameters
        self.sigma_tau = self._generate_gaussian_distribution(self.mean_sigma_tau, self.std_dev_sigma_tau, 1, "log_to_linear")
        self.kr_factor = self._generate_gaussian_distribution(self.mean_kr, self.std_dev_kr, 1, "db_to_linear")
        self.sigma_theta = self._generate_gaussian_distribution(self.mean_sigma_theta, self.std_dev_sigma_theta, 1, "log_deg_to_linear_rad")
        self.sigma_phi = self._generate_gaussian_distribution(self.mean_sigma_phi, self.std_dev_sigma_phi, 1, "log_deg_to_linear_rad")

        # 2. Generate small-scale parameters
        self.multipath_delays = self._calculate_multipath_delays()
        self.multipath_powers = self._calculate_multipath_powers()
        self.multipath_azimuth_angles = self._calculate_multipath_azimuth_angles()
        self.multipath_elevation_angles, self.mean_elevation_angle = self._calculate_multipath_elevation_angles()

        print(f"--> Channel realization generated successfully.")
        print(f"    - Delay Spread (σ_τ): {self.sigma_tau[0]*1e9:.2f} ns")
        print(f"    - Rice Factor (K_R): {self.kr_factor[0]:.2f} (linear)")
        print(f"    - Total Power: {self.multipath_powers[0]:.4f}")
        print(f"    - Mean Elevation Angle: {self.mean_elevation_angle:.2f}°")

    def _generate_gaussian_distribution(self, mean: float, std_dev: float, size: int, scale: str) -> np.ndarray:
        g = np.random.normal(mean, std_dev, size)
        if scale == "log_to_linear":
            return 10**g
        elif scale == "db_to_linear":
            return 10**(g / 10)
        elif scale == "log_deg_to_linear_rad":
            return 10**g * (np.pi / 180)
        else:
            raise ValueError("Unknown distribution type.")

    def _calculate_multipath_delays(self):
        mean_tau_n = self.r_tau * self.sigma_tau
        raw_delays = np.random.exponential(mean_tau_n, self.n_paths)
        norm_multipath_delays = raw_delays - np.min(raw_delays)
        return np.sort(norm_multipath_delays)

    def _calculate_multipath_powers(self):
        xi_n = np.random.normal(0, self.std_dev_sigma_xi, self.n_paths)
        preliminary_powers = np.exp(-self.multipath_delays * (self.r_tau - 1) / (self.r_tau * self.sigma_tau)) * 10**(-xi_n / 10)

        if self.mean_kr == 0 and self.std_dev_kr == 0:
            self.kr_factor = np.array([0])
            return preliminary_powers / np.sum(preliminary_powers)

        multipath_powers = np.zeros(self.n_paths)
        total_dispersed_power = np.sum(preliminary_powers[1:])
        multipath_powers[1:] = (1 / (self.kr_factor[0] + 1)) * (preliminary_powers[1:] / total_dispersed_power)
        multipath_powers[0] = self.kr_factor[0] / (self.kr_factor[0] + 1)
        return multipath_powers

    def _generate_initial_azimuth_angle(self):
        initial_theta_n = 1.42 * self.sigma_theta * np.sqrt(-np.log(self.multipath_powers / np.max(self.multipath_powers)))
        return initial_theta_n

    def _generate_initial_elevation_angle(self):
        initial_phi_n = - self.sigma_phi * np.log(self.multipath_powers / np.max(self.multipath_powers))
        return initial_phi_n

    def _generate_random_signals(self):
        return random.choices([-1, 1], k=self.n_paths)

    def _calculate_multipath_azimuth_angles(self):
        final_theta = np.zeros(self.n_paths)
        un_signals = self._generate_random_signals()
        initial_theta = self._generate_initial_azimuth_angle()
        yn_fluctuations = self._generate_gaussian_distribution(0, self.std_dev_sigma_theta / 7, self.n_paths, "log_deg_to_linear_rad")
        final_theta = (un_signals * initial_theta) + yn_fluctuations
        if self.scenario == "umi_los" or self.scenario == "uma_los":
            final_theta = final_theta - initial_theta[0]
        return final_theta * (180 / np.pi)  # Convert to degrees for output
    
    def _calculate_multipath_elevation_angles(self):
        final_phi = np.zeros(self.n_paths)
        un_signals = self._generate_random_signals()
        initial_phi = self._generate_initial_elevation_angle()
        yn_fluctuations = self._generate_gaussian_distribution(0, self.std_dev_sigma_phi / 7, self.n_paths, "log_deg_to_linear_rad")
        rng = np.random.RandomState(55)
        mean_phi = rng.rand() * 2 * np.pi
        final_phi = (un_signals * initial_phi) + yn_fluctuations
        if self.scenario == "umi_los" or self.scenario == "uma_los":
            final_phi = final_phi - initial_phi[0] + mean_phi
        return final_phi * (180 / np.pi) , mean_phi * (180 / np.pi)  # Convert to degrees for output

if __name__ == '__main__':
    # The main code is much cleaner and more semantic
    my_channel = Channel3GPP(scenario="uma_nlos", frequency_ghz=3, n_paths=100)
    my_channel.generate_channel()
    plot = PlotChannelModel(my_channel)
    plot.plot_pdp()
