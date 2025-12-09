import numpy as np
import matplotlib.pyplot as plt

KO_BOLTZMANN = 1.381e-23  # Constante de Boltzmann (J/K)

class CellFreeSystem:
    
    def __init__(self):
        self.Nbc = 100 # Número de blocos de coerência por rede
        self.Ncf = 300 # Total de redes avaliadas
        self.fc = 3e9  # Frequência da portadora (Hz)
        self.Bw = 20e6  # Largura de banda do sistema (Hz)
        self.Fn = 8  # Figura de ruído do receptor (em linear ou 9 dB)
        self.hAP= 15  # Altura da antena dos APs (m)
        self.hUE= 1.65 # Altura da antena dos UEs (m)
        self.T0 = 296.15 # Temperatura (K)
        self.Lx = 1000 # Dimensão x da área da célula (m)
        self.Ly = 1000 # Dimensão y da área da célula (m)
        self.Pp = 200e-3 # Potência das sequências piloto (W)
        self.Pdl = 200e-3 # Potência de transmissão DL (W)
        self.tau_p = 50 # Comprimento das sequências piloto

        self.M = 20 # Número de APs
        self.K = 10 # Número de UEs

        self.Pn = KO_BOLTZMANN * self.T0 * self.Bw * self.Fn # Potência de ruído (W)

    ########### Distribuição dos APs e UEs
    def _generate_uniform_positions(self, N_element, height):
        """ Gera posições uniformemente distribuídas na área da célula. """
        x_positions = np.random.uniform(-self.Lx/2, self.Lx/2, N_element)
        y_positions = np.random.uniform(-self.Ly/2, self.Ly/2, N_element)
        z_positions = np.full(N_element, height)
        return np.column_stack((x_positions, y_positions, z_positions))
    
    def _calculate_distances(self, ap_positions, ue_positions):
        """ Calcula as distâncias entre APs e UEs. """
        diff = ap_positions[:, np.newaxis, :] - ue_positions[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)
        return distances
    
    ########### Coeficientes de canal
    def _calculate_fspl(self, frequency_MHz, distance_km):
        # ITU-R P.525-4
        return 20*np.log10(distance_km) + 20*np.log10(frequency_MHz) + 32.44
    
    def calculate_fading(self, frequency_MHz, ap_pos, ue_pos):
        distances = self._calculate_distances(ap_pos, ue_pos)
        distances = np.maximum(distances, 1)

        fspl_1m = self._calculate_fspl(frequency_MHz, 0.001)
        x_sf = np.random.normal(0, 8, distances.shape)

        # Desvanecimento em grande escala
        path_loss_db = fspl_1m + 28*np.log10(distances) + x_sf
        slow_fading = 10**(path_loss_db/10)
        # Desvanecimento em pequenacoef_pow_ctrl escala
        fast_fading = np.random.normal(0, 1/2, distances.shape) + \
                      1j * np.random.normal(0, 1/2, distances.shape)
        # Coeficientes do canal com desvanecimento
        g_mk = np.sqrt(slow_fading)*fast_fading

        return slow_fading, fast_fading, g_mk

    ########### Fase de estimação do canal
    def estimate_channel(self, g_mk, slow_fading):
        # Ruído equivalente
        v_mk = np.random.normal(0, self.Pn, g_mk.shape) + \
                1j * np.random.normal(0, self.Pn, g_mk.shape)

        yp = np.sqrt(self.tau_p*self.Pp) * g_mk + v_mk

        num = np.sqrt(self.tau_p * self.Pp) * slow_fading
        den = (self.tau_p * self.Pp * slow_fading) + self.Pn
        c_mk = num / den

        mmse_channel = c_mk*yp

        return mmse_channel, c_mk
    
    ########### Fase de transmissão de dados (DL)
    def _calculate_coef_pow_ctrl(self, slow_fading, c_mk):
        mmse_ch_pow = np.sqrt(self.tau_p * self.Pp) * slow_fading * c_mk
        # Soma sobre todos os usuários K (axis=1) para cada AP m
        sum_mmse_ch_pow = np.sum(mmse_ch_pow, axis=1)

        coef_pow_ctrl = 1.0 / sum_mmse_ch_pow

        return coef_pow_ctrl, mmse_ch_pow
    




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
    slow_fd, fast_fd, channel_coef = system.calculate_fading(frequency_MHz=system.fc/1000,
                                           ap_pos=ap_positions,
                                           ue_pos=ue_positions)
    
    mmse_channel, c_mk = system.estimate_channel(channel_coef, slow_fd)

    coef_pow_ctrl = system._calculate_coef_pow_ctrl(slow_fd, c_mk)

    print("Coeficientes de Controle de Potência para os 20 APs:")
    print(coef_pow_ctrl)