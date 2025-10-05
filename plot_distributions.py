import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as mticker
from matplotlib.offsetbox import AnchoredText
import numpy as np
from channel_model import Channel3GPP

class PlotChannelModel:
    def __init__(self, channel: Channel3GPP):
        self.channel = channel

    def _check_data(self) -> bool:
        if self.channel.multipath_delays is None:
            print("Erro: A realização do canal não foi gerada. Chame 'generate_channel()' primeiro.")
            return False
        return True

    def plot_power_delay_profile(self) -> None:
        if not self._check_data(): return
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.stem(self.channel.multipath_delays * 1e6, self.channel.multipath_powers, linefmt='k-', markerfmt='k^', basefmt=' ', label='NLoS')
        if self.channel.kr_factor > 0:
            ax.stem(self.channel.multipath_delays[0] * 1e6, self.channel.multipath_powers[0], linefmt='b-', markerfmt='b^', basefmt=' ', label='LoS')

        ax.set_yscale('log')
        ax.set_title(f'Perfil de Potência e Atraso (PDP) - {self.channel.scenario.upper()}', fontsize=15)
        ax.set_xlabel('Atraso (μs)'), ax.set_ylabel('Potência Normalizada')
        ax.grid(True, which="both", ls="--", alpha=0.5), ax.legend()
        plt.tight_layout(), plt.show()

    def plot_azimuth_spread(self) -> None:
        if not self._check_data(): return
        self._plot_generic_angles(self.channel.azimuth_angles, self.channel.multipath_powers, "Azimute")

    def plot_elevation_spread(self) -> None:
        if not self._check_data(): return
        self._plot_generic_angles(self.channel.elevation_angles, self.channel.multipath_powers, "Elevação")

    def _plot_generic_angles(self, angles_deg: np.ndarray, powers: np.ndarray, angle_type_name: str) -> None:
        """
        Método genérico que cria uma figura com um plot polar e um cartesiano lado a lado.
        """
        # 1. Preparação dos Dados (agora mais simples)
        los_angle_deg, nlos_angles_deg = angles_deg[0], angles_deg[1:]
        los_power, nlos_powers = powers[0], powers[1:]
        
        los_angle_rad = np.deg2rad(los_angle_deg)
        nlos_angles_rad = np.deg2rad(nlos_angles_deg)
        
        scaled_nlos_powers = np.array([])
        if len(nlos_powers) > 0:
            min_p, max_p = np.min(nlos_powers), np.max(nlos_powers)
            scaled_nlos_powers = 0.4 + 0.5 * (nlos_powers - min_p) / (max_p - min_p) if max_p > min_p else np.full_like(nlos_powers, 0.6)

        # 2. Criação da Figura e Subplots
        fig = plt.figure(figsize=(15, 6))
        ax_polar = fig.add_subplot(1, 2, 1, projection='polar')
        ax_cartesian = fig.add_subplot(1, 2, 2)
        fig.suptitle(f'Análise de Ângulos de {angle_type_name} - {self.channel.scenario.upper()}', fontsize=16)

        # 3. Plotagem do Gráfico Polar (sem alterações na lógica de plot)
        ax_polar.set_title(f"Direção de Chegada", fontsize=14)
        if len(nlos_angles_rad) > 0:
            ax_polar.stem(nlos_angles_rad, scaled_nlos_powers, linefmt='r-', markerfmt='ro', basefmt=' ', label='NLoS')
        if self.channel.kr_factor > 0:
            markerline, _, _ = ax_polar.stem([los_angle_rad], [1.0], linefmt='b-', markerfmt='bo', basefmt=' ', label='LoS')
            plt.setp(markerline, markerfacecolor='white', markeredgewidth=1.5)
        ax_polar.set_yticklabels([]), ax_polar.grid(True)

        # 4. Plotagem do Gráfico Cartesiano (sem alterações na lógica de plot)
        ax_cartesian.set_title(f"Espectro de Potência Angular", fontsize=14)
        if len(nlos_angles_deg) > 0:
            ax_cartesian.stem(nlos_angles_deg, nlos_powers, linefmt='k-', markerfmt='k^', basefmt=' ', label='NLoS')
        if self.channel.kr_factor > 0:
            ax_cartesian.stem([los_angle_deg], [los_power], linefmt='b-', markerfmt='b^', basefmt=' ', label='LoS')
        ax_cartesian.set_yscale('log'), ax_cartesian.set_xlabel(f'Ângulo de {angle_type_name} (°)', fontsize=12)
        ax_cartesian.set_ylabel('Potência Normalizada', fontsize=12), ax_cartesian.grid(True, which="both", ls=":")

        # 5. Finalização
        handles, labels = ax_cartesian.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.93), ncol=2)
        plt.tight_layout(rect=[0, 0, 1, 0.9]), plt.show()

    def plot_3d_directions(self) -> None:
        if not self._check_data(): return
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        dirs = self.channel.arrival_directions
        xs, ys, zs = dirs[0, :], dirs[1, :], dirs[2, :]
        
        if self.channel.n_paths > 1:
            for i in range(1, self.channel.n_paths):
                ax.plot([0, xs[i]], [0, ys[i]], [0, zs[i]], 'r-', alpha=0.7)
            ax.plot(xs[1:], ys[1:], zs[1:], 'r^', markersize=6, label='NLoS')
        
        if self.channel.kr_factor > 0:
            ax.plot([0, xs[0]], [0, ys[0]], [0, zs[0]], 'b-'), ax.plot([xs[0]], [ys[0]], [zs[0]], 'b^', markersize=8, label='LoS')

        ax.set_title('Direções de Chegada em 3D', fontsize=15)
        ax.set_xlabel('Eixo X'), ax.set_ylabel('Eixo Y'), ax.set_zlabel('Eixo Z')
        ax.set_xlim([-1, 1]), ax.set_ylim([-1, 1]), ax.set_zlim([-1, 1])
        ax.view_init(elev=30, azim=-120), ax.set_box_aspect([1,1,1]), ax.legend()
        plt.tight_layout(), plt.show()

    def plot_doppler_spectrum(self) -> None:
        """Plota o Espectro de Potência Doppler."""
        if not self._check_data(): return
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.stem(self.channel.doppler_shifts, self.channel.multipath_powers, linefmt='k-', markerfmt='k^', basefmt=' ', label='NLoS')
        if self.channel.scenario in {"umi_los", "uma_los"}:
            ax.stem(self.channel.doppler_shifts[0], self.channel.multipath_powers[0], linefmt='b-', markerfmt='b^', basefmt=' ', label='LoS')

        info_text = (f'$v_{{rx}} = {self.channel.rx_velocity_mps:.0f}$ m/s, $f_c = {self.channel.frequency_ghz:.0f}$ GHz\n'
                     f'Azimute = ${self.channel.rx_azimuth_deg:.0f}^\\circ$, Elevação = ${self.channel.rx_elevation_deg:.0f}^\\circ$')
        anchored_text = AnchoredText(info_text, loc="upper right", frameon=True, prop={'size': 10})
        
        ax.add_artist(anchored_text)
        ax.set_yscale('log')
        ax.set_title(f'Espectro de Potência Doppler - {self.channel.scenario.upper()}', fontsize=15)
        ax.set_xlabel('Desvio Doppler (Hz)', fontsize=12)
        ax.set_ylabel('Potência Normalizada', fontsize=12)
        ax.grid(True, which="both", ls=":", alpha=0.6)
        ax.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_statistical_delay_spread(scenario: str) -> None:
        frequencies_ghz = np.logspace(np.log10(0.5), np.log10(100), 200)
        calc = Channel3GPP(scenario=scenario, frequency_ghz=frequencies_ghz, n_paths=1, rx_velocity_mps=1)
        
        mean_log_s, std_dev_log_s = calc.mean_sigma_tau_log, calc.std_dev_sigma_tau_log
        mean_us = (10**mean_log_s) * 1e6
        upper_bound_us, lower_bound_us = (10**(mean_log_s + std_dev_log_s)) * 1e6, (10**(mean_log_s - std_dev_log_s)) * 1e6

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(frequencies_ghz, mean_us, color='blue', lw=2, label='Média de $\\sigma_\\tau$')
        ax.fill_between(frequencies_ghz, lower_bound_us, upper_bound_us, color='blue', alpha=0.2, label='Média ± 1 Desvio Padrão')
        
        ax.set_xscale('log'), ax.set_title(f'Espalhamento de Atraso (Modelo Estatístico) - {scenario.upper()}', fontsize=15)
        ax.set_xlabel('Frequência da Portadora ($f_c$) [GHz]'), ax.set_ylabel('Espalhamento de Atraso ($\\sigma_\\tau$) [$\\mu$s]')
        ax.set_xlim(0.5, 100), ax.set_ylim(bottom=0), ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.grid(True, which='both', linestyle='--', linewidth=0.5), ax.legend()
        plt.tight_layout(), plt.show()