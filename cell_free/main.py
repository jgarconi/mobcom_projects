import numpy as np
import matplotlib.pyplot as plt

BOLTZMANN = 1.381e-23  # Constante de Boltzmann (J/K)

class CellFreeSystem:
    
    def __init__(self):
        self.Nbc = 100      # Número de blocos de coerência por rede
        self.Ncf = 300      # Total de redes avaliadas
        self.fc = 3e3       # Frequência da portadora (MHz)
        self.Bw = 20        # Largura de banda do sistema (MHz)
        self.Fn = 8         # Figura de ruído do receptor (W ou 9 dB)
        self.hAP= 15        # Altura da antena dos APs (m)
        self.hUE= 1.65      # Altura da antena dos UEs (m)
        self.T0 = 296.15    # Temperatura (K)
        self.Lx = 1000      # Dimensão x da área da célula (m)
        self.Ly = 1000      # Dimensão y da área da célula (m)
        self.Pp = 200e-3    # Potência das sequências piloto (W)
        self.Pdl = 200e-3   # Potência de transmissão DL (W)
        self.tau_p = 50     # Comprimento das sequências piloto

        self.M = 20         # Número de APs
        self.K = 10         # Número de UEs

        # Potência de ruído (W)
        self.Pn = BOLTZMANN * self.T0 * self.Bw * self.Fn

        # Número de usos do canal (TODO: calcular tau_c como TcBc)
        self.tau_c = 100

    ########### Distribuição dos APs e UEs
    def _generate_uniform_positions(self, N_devices, height):
        """ Gera posições uniformemente distribuídas na área da célula. """
        x_positions = np.random.uniform(-self.Lx/2, self.Lx/2, N_devices)
        y_positions = np.random.uniform(-self.Ly/2, self.Ly/2, N_devices)
        z_positions = np.full(N_devices, height)
        return np.column_stack((x_positions, y_positions, z_positions))
    
    def _calculate_distances(self, ap_positions, ue_positions):
        """ Calcula as distâncias entre APs e UEs. """
        diff = ap_positions[:, np.newaxis, :] - ue_positions[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)
        return distances
    
    ########### Coeficientes de canal
    def _calculate_fspl(self, frequency_MHz, distance_km):
        """ Calcula as perdas de espaço livre de acordo com a ITU-R P.525.4 """
        return 20*np.log10(distance_km) + 20*np.log10(frequency_MHz) + 32.4
    
    def calculate_channel_coefficients(self, frequency_MHz, ap_pos, ue_pos):
        """ Calcula os coeficientes do canal com desvanecimento entre APs e UEs """
        distances = self._calculate_distances(ap_pos, ue_pos)
        distances = np.maximum(distances, 1)

        fspl_1m = self._calculate_fspl(frequency_MHz, 0.001)
        x_sf = np.random.normal(0, 8, distances.shape)

        # Desvanecimento em larga escala
        path_loss_db = fspl_1m + 28*np.log10(distances) + x_sf
        slow_fading = 10**(path_loss_db/10)

        # Desvanecimento em pequena escala
        fast_fading = np.random.normal(0, 1/2, distances.shape) + \
                 1j * np.random.normal(0, 1/2, distances.shape)

        # Coeficiente do canal com desvanecimento
        g_mk = np.sqrt(slow_fading)*fast_fading

        return slow_fading, fast_fading, g_mk

    ########### Fase de estimação do canal
    def estimate_channel(self, g_mk, slow_fading):
        """ Estimação do canal entre o AP m e os k UEs usando MMSE baseado em sequências piloto. """
        # Ruído equivalente
        v_mk = np.random.normal(0, self.Pn, g_mk.shape) + \
                1j * np.random.normal(0, self.Pn, g_mk.shape)

        # Projeção da sequência piloto
        yp = np.sqrt(self.tau_p*self.Pp) * g_mk + v_mk

        # Coeficiente de MMSE
        num = np.sqrt(self.tau_p * self.Pp) * slow_fading
        den = (self.tau_p * self.Pp * slow_fading) + self.Pn
        c_mk = num / den

        # Canal estimado
        mmse_channel = c_mk*yp

        return mmse_channel, c_mk
    
    ########### Fase de transmissão de dados (DL)
    def _calculate_coef_pow_ctrl(self, slow_fading, c_mk):
        """ Calcula os coeficientes de controle de potência para cada AP m. """

        # Ganho do canal estimado
        estimated_channel_gain = np.sqrt(self.tau_p * self.Pp) * slow_fading * c_mk

        # Soma o ganho sobre todos os usuários K (axis=1) para cada AP m
        channel_gain_m = np.sum(estimated_channel_gain, axis=1)

        # Coeficiente de controle de potência do AP m (ηm)
        power_control_coef_m = 1 / channel_gain_m

        return power_control_coef_m, channel_gain_m

    def calculate_dl_sinr(self, channel_gain, coef_pow_ctrl):
        """ Calcula o SINR de downlink para cada UE k. """
        sinr_ec_dl = np.zeros(self.K)

        for k in range(self.K):
            # Potência do sinal recebido pelo usuário k dos m APs
            numerator = self.Pdl * (np.sum(np.sqrt(coef_pow_ctrl) * channel_gain[:, k]))**2
            interference = 0
            # Interferência dos outros usuários + ruído
            for j in range(self.K):
                if j != k:
                    interference += self.Pdl * (np.sum(np.sqrt(coef_pow_ctrl) * channel_gain[:, j]))**2
            denominator = interference + self.Pn
            sinr_ec_dl[k] = numerator / denominator

        return sinr_ec_dl

    def calculate_achivable_rate(self, sinr_ec_dl):
        """ Calcula a taxa alcançável para cada UE k com base no SINR de downlink. """
        rates_ci = self.Bw * np.log2(1 + sinr_ec_dl)
        rates_ce = np.log2(1 + sinr_ec_dl)
        return rates_ci, rates_ce


    def plot_positions(self):
        """ Plota as posições dos APs e UEs na área da célula. """
        ap_positions = self._generate_uniform_positions(self.M, self.hAP)
        ue_positions = self._generate_uniform_positions(self.K, self.hUE)

        plt.figure(figsize=(8, 8))
        plt.scatter(ap_positions[:, 0], ap_positions[:, 1], c='blue', marker='^', label='APs')
        plt.scatter(ue_positions[:, 0], ue_positions[:, 1], c='red', marker='o', label='UEs')
        plt.xlim(-self.Lx/2, self.Lx/2)
        plt.ylim(-self.Ly/2, self.Ly/2)
        plt.xlabel('Posição X (m)')
        plt.ylabel('Posição Y (m)')
        plt.title('Posições dos APs e UEs na Área da Célula')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    system = CellFreeSystem()

    ap_positions = system._generate_uniform_positions(system.M, system.hAP)
    ue_positions = system._generate_uniform_positions(system.K, system.hUE)
    slow_fd, fast_fd, channel_coef = system.calculate_channel_coefficients(
                                     frequency_MHz=system.fc,
                                     ap_pos=ap_positions,
                                     ue_pos=ue_positions)
    
    mmse_channel, c_mk = system.estimate_channel(channel_coef, slow_fd)

    coef_pow_ctrl = system._calculate_coef_pow_ctrl(slow_fd, c_mk)

    print("Coeficientes de Controle de Potência para os 20 APs:")
    print(coef_pow_ctrl)