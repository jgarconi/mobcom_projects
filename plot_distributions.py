import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as mticker
import numpy as np

from class_3gpp import Channel3GPP

class PlotChannelModel:
    def __init__(self, channel_model):
        self.channel_model = channel_model

    def plot_power_delay_profile(self):
        """Plots all available multipath profiles: powers, azimuth angles, and elevation angles (if available)."""
        channel = self.channel_model
        if channel.multipath_powers is None:
            print("Run the 'generate_channel()' method first.")
            return

        # --- Power Delay Profile (PDP) ---
        plt.figure(figsize=(10, 6))
        plt.stem(channel.multipath_delays * 1e6, channel.multipath_powers, linefmt='k-', markerfmt='k^', basefmt=' ')
        if channel.kr_factor > 0:
            plt.stem(channel.multipath_delays[0] * 1e6, channel.multipath_powers[0], linefmt='b-', markerfmt='b^', basefmt=' ')
        plt.yscale('log')
        plt.title(f'Power Delay Profile (PDP) - {channel.scenario}')
        plt.xlabel('Delay (μs)')
        plt.ylabel('Normalized Power (αn²)')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.show()

    def plot_delay_spread_profile(self):
        """
        Plota o perfil de Espalhamento de Atraso vs. frequência, usando a própria
        classe Channel3GPP para calcular os parâmetros do modelo.
        """
        # 1. Pega o nome do cenário do objeto de canal principal
        channel = self.channel_model
        scenario = channel.scenario

        # 2. Define a faixa de frequências para o eixo X do gráfico
        frequencies_ghz = np.logspace(np.log10(0.5), np.log10(100), 200)

        # 3. Cria uma nova instância temporária de Channel3GPP para ser nossa "calculadora".
        #    Passamos a ela o ARRAY de frequências. n_paths é irrelevante aqui.
        params_calculator = Channel3GPP(
            scenario=scenario,
            frequency_ghz=frequencies_ghz,
            n_paths=1
        )

        # 4. Agora, os atributos .mean_sigma_tau e .std_dev_sigma_tau deste objeto
        #    são ARRAYS contendo os valores para cada frequência.
        mean_log_s = params_calculator.mean_sigma_tau

        std_dev_log_s = params_calculator.std_dev_sigma_tau
        
        # 5. A partir daqui, a lógica de plotagem é a mesma de antes
        upper_bound_log_s = mean_log_s + std_dev_log_s
        lower_bound_log_s = mean_log_s - std_dev_log_s

        mean_us = (10**mean_log_s) * 1e6
        upper_bound_us = (10**upper_bound_log_s) * 1e6
        lower_bound_us = (10**lower_bound_log_s) * 1e6

        # 6. Cria o gráfico
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(frequencies_ghz, mean_us, color='blue', linewidth=2, label='Média de $\sigma_\\tau$')
        ax.fill_between(frequencies_ghz, lower_bound_us, upper_bound_us, 
                        color='blue', alpha=0.2, label='Média ± Desvio Padrão')
        
        # Configuração da aparência do gráfico...
        ax.set_xscale('log')
        ax.set_title(f'Espalhamento de Atraso para o Cenário: {scenario.upper()}', fontsize=16)
        ax.set_xlabel('Frequência da portadora – $f_c$ (GHz)', fontsize=14)
        ax.set_ylabel('Espalhamento de Atraso, $\sigma_\\tau$ ($\\mu$s)', fontsize=14)
        ax.set_xlim(0.5, 100)
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.legend(fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=12)

        plt.tight_layout()
        plt.show()

    def plot_elevation_angles(self):
        """
        Gera os gráficos (polar e cartesiano) para os ÂNGULOS DE ELEVAÇÃO.
        """
        print("Gerando gráfico para Ângulos de Elevação...")
        self._plot_generic_angles(
            angles_data=self.channel_model.multipath_elevation_angles,
            angle_type_name="Elevação",
            angle_type_name_lower="elevação",
            cartesian_xlim_dynamic=True  # Limites dinâmicos para elevação
        )

    def plot_azimuth_angles(self):
        """
        Gera os gráficos (polar e cartesiano) para os ÂNGULOS DE AZIMUTE.
        """
        print("Gerando gráfico para Ângulos de Azimute...")
        self._plot_generic_angles(
            angles_data=self.channel_model.multipath_azimuth_angles,
            angle_type_name="Azimute",
            angle_type_name_lower="azimute",
            cartesian_xlim_dynamic=True
        )

    def _plot_generic_angles(self, angles_data, angle_type_name, angle_type_name_lower, cartesian_xlim_dynamic, cartesian_xlim_range=None):
        """
        Método genérico interno para plotar qualquer tipo de ângulo.
        """
        channel = self.channel_model

        # --- 1. CRIAÇÃO DA FIGURA E SUBPLOTS ---
        fig = plt.figure(figsize=(14, 6))
        ax_polar = fig.add_subplot(1, 2, 1, projection='polar')
        ax_cartesian = fig.add_subplot(1, 2, 2)

        # --- DADOS PARA PLOTAGEM ---
        multipath_powers = channel.multipath_powers
        
        los_angle = angles_data[0]
        los_power = multipath_powers[0]

        if channel.n_paths > 1:
            nlos_angles = angles_data[1:]
            nlos_powers = multipath_powers[1:]
        else:
            nlos_angles = np.array([])
            nlos_powers = np.array([])

        # --- PLOTAGEM DO GRÁFICO POLAR (ax_polar) ---
        ax_polar.set_title(f"Direção de Chegada ({angle_type_name})", fontsize=14)

        # **
        # ** AJUSTE AQUI: Converte os ângulos para a faixa [0, 360] para o gráfico polar **
        # ** O operador '%' em Python lida corretamente com números negativos
        # ** (ex: -90 % 360 se torna 270).
        # **
        los_angle_polar = los_angle % 360
        nlos_angles_polar = nlos_angles % 360

        if len(nlos_powers) > 0 and np.max(nlos_powers) > np.min(nlos_powers):
            scaled_nlos_powers = 0.4 + 0.5 * (nlos_powers - np.min(nlos_powers)) / (np.max(nlos_powers) - np.min(nlos_powers))
        elif len(nlos_powers) == 1:
             scaled_nlos_powers = np.array([0.6])
        else:
            scaled_nlos_powers = np.array([])

        # Usa os ângulos convertidos (nlos_angles_polar) para plotar
        if len(nlos_angles) > 0:
            ax_polar.stem(
                np.deg2rad(nlos_angles_polar), scaled_nlos_powers,
                linefmt='r-', markerfmt='ro', basefmt=' ',
                label='NLoS'
            )
        
        # Usa o ângulo convertido (los_angle_polar) para plotar
        if channel.kr_factor > 0 and los_power > 0:
            markerline, _, _ = ax_polar.stem(
                [np.deg2rad(los_angle_polar)], [1.0],
                linefmt='b-', markerfmt='bo', basefmt=' ',
                label='LoS'
            )
            plt.setp(markerline, markerfacecolor='white', markeredgewidth=1.5)

        # ax_polar.set_theta_zero_location('E')
        # ax_polar.set_theta_direction(1)
        ax_polar.set_yticklabels([])
        ax_polar.grid(True)
        plt.setp(ax_polar.get_xticklabels(), fontsize=12)

        # --- PLOTAGEM DO GRÁFICO CARTESIANO (ax_cartesian) ---
        # ** NENHUMA MUDANÇA AQUI - Usa os ângulos originais (nlos_angles, los_angle) **
        ax_cartesian.set_title(f"Espectro de Potência Angular ({angle_type_name})", fontsize=14)

        if len(nlos_angles) > 0:
            markerline, _, _ = ax_cartesian.stem(
                nlos_angles, nlos_powers, # Usa o dado original
                linefmt='k-', markerfmt='k^', basefmt=' ',
                label='NLoS'
            )
            plt.setp(markerline, markerfacecolor='black')

        if channel.kr_factor > 0 and los_power > 0:
            markerline, _, _ = ax_cartesian.stem(
                [los_angle], [los_power], # Usa o dado original
                linefmt='b-', markerfmt='b^', basefmt=' ',
                label='LoS'
            )
            plt.setp(markerline, markerfacecolor='blue')

        ax_cartesian.set_yscale('log')
        ax_cartesian.set_xlabel(f'Ângulos de chegada em {angle_type_name_lower} (°)', fontsize=12)
        ax_cartesian.set_ylabel('Potência', fontsize=12)
        ax_cartesian.grid(True, which='both', linestyle=':')
        ax_cartesian.tick_params(axis='both', which='major', labelsize=12)
        
        # if cartesian_xlim_dynamic:
        #     if len(angles_data) > 0:
        #          ax_cartesian.set_xlim(np.min(angles_data) - 5, np.max(angles_data) + 5)
        # else:
        #     ax_cartesian.set_xlim(cartesian_xlim_range)
        #     ax_cartesian.set_xticks(np.arange(cartesian_xlim_range[0], cartesian_xlim_range[1] + 1, 60))

        # --- FINALIZAÇÃO ---
        plt.tight_layout(pad=3.0)
        plt.show()

    def plot_3d_directions(self):
        """
        Gera um gráfico 3D das direções de chegada (vetores).
        """
        print("Gerando gráfico 3D das direções de chegada...")
        channel = self.channel_model

        # Calcula as direções de chegada (X, Y, Z)
        directions = channel._calculate_arrival_directions()
        
        # Garante que directions seja um array 2D (3, N)
        if directions.shape[0] != 3 and directions.shape[1] == 3:
            directions = directions.T

        xs = directions[0, :]
        ys = directions[1, :]
        zs = directions[2, :]

        # Cria a figura e o eixo 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # --- Plotar componentes NLoS ---
        # As linhas partem da origem (0,0,0)
        # O LoS está na posição 0, então os NLoS são de 1 em diante.
        if channel.n_paths > 1:
            nlos_xs = xs[1:]
            nlos_ys = ys[1:]
            nlos_zs = zs[1:]

            # Plotar as linhas (stems) vermelhas
            for i in range(len(nlos_xs)):
                ax.plot([0, nlos_xs[i]], [0, nlos_ys[i]], [0, nlos_zs[i]], 'r-')
            
            # Plotar os marcadores (triângulos vazados) vermelhos
            ax.plot(nlos_xs, nlos_ys, nlos_zs, 'r^', markerfacecolor='white', markeredgecolor='red', markersize=6, label='NLoS')

        # --- Plotar componente LoS ---
        # Apenas se houver um LoS significativo
        if channel.kr_factor > 0 and channel.multipath_powers[0] > 0:
            los_x = xs[0]
            los_y = ys[0]
            los_z = zs[0]

            # Plotar a linha (stem) azul
            ax.plot([0, los_x], [0, los_y], [0, los_z], 'b-')

            # Plotar o marcador (triângulo vazado) azul
            ax.plot([los_x], [los_y], [los_z], 'b^', markerfacecolor='white', markeredgecolor='blue', markersize=8, label='LoS')


        # --- Configurações do Gráfico 3D ---
        ax.set_xlabel('Eixo X', fontsize=12)
        ax.set_ylabel('Eixo Y', fontsize=12)
        ax.set_zlabel('Eixo Z', fontsize=12)
        ax.set_title('Direções de Chegada em 3D', fontsize=14)

        # 1. Ajustar os limites e as marcações (ticks) dos eixos
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        
        ticks = [-1, -0.5, 0, 0.5, 1]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_zticks(ticks)
        
        # 2. Ajustar a posição da câmera para o eixo Z ficar à esquerda
        # 'elev' = ângulo de elevação da câmera (olhar de cima/baixo)
        # 'azim' = ângulo de azimute da câmera (girar em torno do eixo Z)
        ax.view_init(elev=20, azim=-135)

        # Garante que a proporção dos eixos seja igual (visualização cúbica)
        ax.set_box_aspect([1,1,1])
        
        ax.grid(True)
        ax.legend()

        plt.tight_layout()
        plt.show()

# if __name__ == '__main__':
# TODO: create unit tests for all functions
# TODO: refactor _plot_generic_angles