import numpy as np
import matplotlib.pyplot as plt

KO_BOLTZMANN = 1.381e-23  # Constante de Boltzmann (J/K)

class CellFreeSystem:
    """
    Representa o sistema de modulação M-QAM, incluindo a constelação, 
    canais (Rayleigh, AWGN) e o detector de Mínima Distância.
    """
    def __init__(self):
        self.Nbc = 100 # Número de blocos de coerência por rede
        self.Ncf = 300 # Total de redes avaliadas
        self.fc = 3e9  # Frequência da portadora (Hz)
        self.Bw = 20e6  # Largura de banda do sistema (Hz)
        self.Fn = 9  # Figura de ruído do receptor (dB)
        self.hAP= 15  # Altura da antena dos APs (m)
        self.hUE= 1.65 # Altura da antena dos UEs (m)
        self.T0 = 296.15 # Temperatura (K)
        self.Lx = 1000 # Dimensão x da área da célula (m)
        self.Ly = 1000 # Dimensão y da área da célula (m)
        self.Pp = 200e-3 # Potência das sequências piloto (W)
        self.Pdl = 200e-3 # Potência de transmissão DL (W)
        self.taup = 50 # Comprimento das sequências piloto

        self.M = 20 # Número de APs
        self.K = 10 # Número de UEs

        self.Pn = KO_BOLTZMANN * self.T0 * self.Bw * 10**(self.Fn / 10) # Potência de ruído (W)

    def _generate_uniform_positions(self, N_nodes, height):
        """ Gera posições uniformemente distribuídas na área da célula. """
        x_positions = np.random.uniform(-self.Lx/2, self.Lx/2, N_nodes)
        y_positions = np.random.uniform(-self.Ly/2, self.Ly/2, N_nodes)
        z_positions = np.full(N_nodes, height)
        return np.column_stack((x_positions, y_positions, z_positions))
    
    def _calculate_distances(self, ap_positions, ue_positions):
        """ Calcula as distâncias entre APs e UEs. """
        distances = np.zeros((self.M, self.K))
        for m in range(self.M):
            for k in range(self.K):
                distances[m, k] = np.linalg.norm(ap_positions[m] - ue_positions[k])
        return distances
    
    def _calculate_fspl(self, frequency_MHz, distance_km=0.001):
        # ITU-R P.525-4
        return 20*np.log10(distance_km) + 20*np.log10(frequency_MHz) + 32.44
    
    def calculate_fading(self, frequency_MHz, ap_pos, ue_pos):
        distances = self._calculate_distances(ap_pos, ue_pos)
        fspl_1m = self._calculate_fspl(frequency_MHz)
        x_sf = np.random.normal(0, 8, len(distances.T))

        # Desvanecimento em grande escala
        slow_fadig = 10**((fspl_1m + 28*np.log10(distances) + x_sf)/10)
        # Desvanecimento em pequena escala
        fast_fading = np.random.normal(0, 1/2, len(distances.T)) + \
                      1j * np.random.normal(0, 1/2, len(distances.T))
        # Coeficientes do canal
        g_mk = np.sqrt(slow_fadig)*fast_fading

        return slow_fadig, fast_fading, g_mk

    def estimate_channel(self, g_mk, slow_fading):
        # Ruído equivalente
        v_mk = np.random.normal(0, self.Pn)
        yp = np.sqrt(self.taup*self.Pp)*g_mk + v_mk
        c_mk = (np.sqrt(self.taup*self.Pp)*slow_fading)/ \
                (self.taup*self.Pp*slow_fading + self.Pn)
        
        mmse_channel = c_mk*yp

        return mmse_channel
    
    def _calculate_power_control(self, slow_fading, mmse_channel):
        pow_mmse_channel = np.sqrt(self.taup*self.Pp)*slow_fading*mmse_channel
        coef_power_control = np.zeros(self.M)

        for m in range(self.M):
            for k in range(self.K):
                coef_power_control[m] = 1/np.sum(pow_mmse_channel)
        return coef_power_control


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
    
    mmse_channel = system.estimate_channel(channel_coef, slow_fd)

    coef_power_ctl = system._calculate_power_control(slow_fd, mmse_channel)

    print((coef_power_ctl))