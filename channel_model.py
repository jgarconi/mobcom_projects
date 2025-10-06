import numpy as np

LIGHT_SPEED = 299792458
ANGULAR_FLUCTUATION_FACTOR = 7.0
MEAN_ELEVATION_DEG = 45

class Channel3GPP:
    """
    Models a wireless communication channel based on 3GPP 38.901 specifications.
    """

    # Dictionary of statistical parameters per scenario, following 3GPP TR 38.901 Tables 7.5-6.
    # For frequency-dependent values, a tuple (A, B) represents the formula: A * log10(fc_ghz) + B
    _SCENARIO_PARAMS = {
        "umi_los": {
            "mean_sigma_tau_log": (-0.24, -7.14), "std_dev_sigma_tau_log": 0.38, "r_tau": 3.0,       # Delay Spread (σ_τ, r_τ)
            "std_dev_sigma_xi_db": 4.0, "mean_kr_db": 9.0, "std_dev_kr_db": 5.0,                     # Power (K_R, σ_ξ)
            "mean_sigma_theta_log_deg": (-0.08, 1.73), "std_dev_sigma_theta_log_deg": (0.014, 0.28), # Azimuth Angle (σ_θ)
            "mean_sigma_phi_log_deg": (-0.1, 0.73), "std_dev_sigma_phi_log_deg": (-0.04, 0.34),      # Elevation Angle (σ_φ)
        },
        "umi_nlos": {
            "mean_sigma_tau_log": (-0.24, -6.83), "std_dev_sigma_tau_log": (-0.16, 0.28), "r_tau": 2.1, # Delay Spread (σ_τ, r_τ)
            "std_dev_sigma_xi_db": 7.82, "mean_kr_db": 0.0, "std_dev_kr_db": 0.0,                       # Power (K_R, σ_ξ)
            "mean_sigma_theta_log_deg": (-0.08, 1.81), "std_dev_sigma_theta_log_deg": (0.05, 0.3),      # Azimuth Angle (σ_θ)
            "mean_sigma_phi_log_deg": (-0.04, 0.92), "std_dev_sigma_phi_log_deg": (-0.07, 0.41),        # Elevation Angle (σ_φ)
        },
        "uma_los": {
            "mean_sigma_tau_log": (-0.0963, -6.955), "std_dev_sigma_tau_log": 0.66, "r_tau": 2.5, # Delay Spread (σ_τ, r_τ)
            "std_dev_sigma_xi_db": 4.0, "mean_kr_db": 9.0, "std_dev_kr_db": 3.5,                  # Power (K_R, σ_ξ)
            "mean_sigma_theta_log_deg": 1.81, "std_dev_sigma_theta_log_deg": 0.2,                 # Azimuth Angle (σ_θ)
            "mean_sigma_phi_log_deg": 0.95, "std_dev_sigma_phi_log_deg": 0.16,                    # Elevation Angle (σ_φ)
        },
        "uma_nlos": {
            "mean_sigma_tau_log": (-0.204, -6.28), "std_dev_sigma_tau_log": 0.39, "r_tau": 2.3,   # Delay Spread (σ_τ, r_τ)
            "std_dev_sigma_xi_db": 6.0, "mean_kr_db": 0.0, "std_dev_kr_db": 0.0,                  # Power (K_R, σ_ξ)
            "mean_sigma_theta_log_deg": (-0.27, 2.08, True), "std_dev_sigma_theta_log_deg": 0.11, # Azimuth Angle (σ_θ)
            "mean_sigma_phi_log_deg": (-0.3236, 1.512, True), "std_dev_sigma_phi_log_deg": 0.16,  # Elevation Angle (σ_φ)
        }
    }

    def __init__(self, 
             scenario: str, 
             frequency_ghz: float, 
             n_paths: int, 
             rx_velocity_mps: float, 
             rx_azimuth_deg: float = 90.0, 
             rx_elevation_deg: float = 0.0):

        self.scenario = scenario
        self.frequency_ghz = frequency_ghz
        self.n_paths = n_paths
        self.rx_velocity_mps = rx_velocity_mps
        self.rx_azimuth_deg = rx_azimuth_deg
        self.rx_elevation_deg = rx_elevation_deg

        self.multipath_delays = None
        self.multipath_powers = None
        self.azimuth_angles = None
        self.elevation_angles = None
        self.arrival_directions = None
        self.doppler_shifts = None

        self._load_scenario_parameters()

    def generate_channel(self):
        """
        Performs a single channel realization, calculating all its parameters.
        """

        # Generate large-scale parameters
        self.delay_spread = self._generate_gaussian(self.mean_sigma_tau_log, self.std_dev_sigma_tau_log, scale="log_to_linear")
        self.kr_factor = 0.0 if self.scenario in {"umi_nlos", "uma_nlos"} else self._generate_gaussian(self.mean_kr_db, self.std_dev_kr_db, scale="db_to_linear")
        self.azimuth_spread_rad = self._generate_gaussian(self.mean_sigma_theta_log_deg, self.std_dev_sigma_theta_log_deg, scale="log_deg_to_linear_rad")
        self.elevation_spread_rad = self._generate_gaussian(self.mean_sigma_phi_log_deg, self.std_dev_sigma_phi_log_deg, scale="log_deg_to_linear_rad")

        # Generate small-scale parameters (multipath components)
        self.multipath_delays = self._calculate_multipath_delays(sigma_tau=self.delay_spread)
        self.multipath_powers = self._calculate_multipath_powers(delays=self.multipath_delays, sigma_tau=self.delay_spread)   

        power_ratios = self.multipath_powers / np.max(self.multipath_powers)
        self.azimuth_angles = self._calculate_angles(power_ratios=power_ratios, sigma_angle_rad=self.azimuth_spread_rad)
        self.elevation_angles = self._calculate_angles(power_ratios=power_ratios, sigma_angle_rad=self.elevation_spread_rad, is_elevation=True)
        self.arrival_directions = self._calculate_arrival_directions()
        self.doppler_shifts = self._calculate_doppler_shift()

        print(f"--> Channel realization generated successfully.")
        print(f"    - Delay Spread (σ_τ): {self.delay_spread*1e9} ns")
        print(f"    - Rice Factor (K_R): {self.kr_factor} (linear)")
        print(f"    - Total power: {np.sum(self.multipath_powers)}")
        if self.scenario in {"umi_los", "uma_los"}:
            print(f"    - LoS Power [% of Total Power]: {self.multipath_powers[0]*100:.2f}%")

    def _load_scenario_parameters(self):
        try:
            params = self._SCENARIO_PARAMS[self.scenario]
        except KeyError:
            raise ValueError(f"Invalid scenario: '{self.scenario}'")

        def _get_value(key: str):
            par = params.get(key)
            if not isinstance(par, tuple): return par
            a, b = par[0], par[1]
            use_direct_fc = len(par) > 2 and par[2]
            log_arg = self.frequency_ghz if use_direct_fc else (1 + self.frequency_ghz)
            return a * np.log10(log_arg) + b

        # Delay Spread
        self.mean_sigma_tau_log = _get_value("mean_sigma_tau_log")
        self.std_dev_sigma_tau_log = _get_value("std_dev_sigma_tau_log")
        self.r_tau = _get_value("r_tau")

        # Power
        self.std_dev_sigma_xi_db = _get_value("std_dev_sigma_xi_db")
        self.mean_kr_db = _get_value("mean_kr_db")
        self.std_dev_kr_db = _get_value("std_dev_kr_db")

        # Azimuth Angle
        self.mean_sigma_theta_log_deg = _get_value("mean_sigma_theta_log_deg")
        self.std_dev_sigma_theta_log_deg = _get_value("std_dev_sigma_theta_log_deg")

        # Elevation Angle
        self.mean_sigma_phi_log_deg = _get_value("mean_sigma_phi_log_deg")
        self.std_dev_sigma_phi_log_deg = _get_value("std_dev_sigma_phi_log_deg")

    def _generate_gaussian(self, mean: float, std_dev: float, size: int = 1, scale: str | None = None) -> float | np.ndarray:
        if mean is None or std_dev is None: return np.zeros(size) if size > 1 else 0.0
        g = np.random.normal(mean, std_dev, size)
        
        if scale == "log_to_linear": g = 10**g
        elif scale == "db_to_linear": g = 10**(g / 10)
        elif scale == "log_deg_to_linear_rad": g = np.deg2rad(10**g)
            
        return g.item() if size == 1 and isinstance(g, np.ndarray) else g

    def _calculate_multipath_delays(self, sigma_tau):
        mean_tau_n = self.r_tau * sigma_tau
        raw_delays = np.random.exponential(mean_tau_n, self.n_paths)
        norm_multipath_delays = raw_delays - np.min(raw_delays)

        return np.sort(norm_multipath_delays)

    def _calculate_multipath_powers(self, delays, sigma_tau):
        xi_n = self._generate_gaussian(0, self.std_dev_sigma_xi_db, size=self.n_paths)
        delay_factor = (self.r_tau - 1) / (self.r_tau * sigma_tau)
        preliminary_powers = np.exp(-delays * delay_factor) * 10**(-xi_n / 10)

        total_dispersed_power = np.sum(preliminary_powers[1:])
        norm_powers = (1 / (self.kr_factor + 1)) * (preliminary_powers / total_dispersed_power)
        norm_powers[0] = self.kr_factor / (self.kr_factor + 1)

        # print(norm_powers[0]/np.sum(norm_powers[1:]))

        return norm_powers

    def _calculate_angles(self, power_ratios: np.ndarray, sigma_angle_rad: float, is_elevation: bool = False) -> np.ndarray:
        safe_power_ratios = np.clip(power_ratios, 1e-12, 1)

        if is_elevation:
            initial_angles_rad = -sigma_angle_rad * np.log(safe_power_ratios)
        else: # Azimuth
            initial_angles_rad = 1.42 * sigma_angle_rad * np.sqrt(-np.log(safe_power_ratios))

        random_signs = np.random.choice([-1, 1], size=self.n_paths)
        fluctuations_rad = self._generate_gaussian(0, sigma_angle_rad / ANGULAR_FLUCTUATION_FACTOR, size=self.n_paths)
        final_angles_rad = (random_signs * initial_angles_rad) + fluctuations_rad
        
        if self.scenario in {"umi_los", "uma_los"}:
            final_angles_rad -= final_angles_rad[0]
            if is_elevation:
                final_angles_rad += np.deg2rad(MEAN_ELEVATION_DEG)

        return np.rad2deg(final_angles_rad)

    def _calculate_arrival_directions(self):
        azimuth_rad = np.deg2rad(self.azimuth_angles)
        elevation_rad = np.deg2rad(self.elevation_angles)

        arrival_directions = np.array([
            np.cos(azimuth_rad) * np.sin(elevation_rad),
            np.sin(azimuth_rad) * np.sin(elevation_rad),
            np.cos(elevation_rad)
        ])
        return arrival_directions

    def _calculate_doppler_shift(self):
        frequency_hz = self.frequency_ghz * 1e9
        wavelength = LIGHT_SPEED / frequency_hz
        max_doppler_shift = self.rx_velocity_mps / wavelength

        rx_azimuth_rad = np.deg2rad(self.rx_azimuth_deg)
        rx_elevation_rad = np.deg2rad(self.rx_elevation_deg)
        rx_direction = np.array([
            np.cos(rx_azimuth_rad) * np.sin(rx_elevation_rad),
            np.sin(rx_azimuth_rad) * np.sin(rx_elevation_rad),
            np.cos(rx_elevation_rad)
        ])

        cos_angles = np.dot(rx_direction, self.arrival_directions)
        doppler_shifts = max_doppler_shift * cos_angles
        
        return doppler_shifts
