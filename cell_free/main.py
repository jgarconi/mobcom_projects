import numpy as np
import matplotlib.pyplot as plt

# Configuração de estilo dos gráficos
plt.rcParams.update({'font.size': 12, 'axes.grid': True, 'grid.alpha': 0.5})

class CellFreeSystem:
    def __init__(self, M, K):
        """
        Cria as instâncias do sistema Cell-Free com M APs e K UEs
        """

        self.M = M              # Número de APs
        self.K = K              # Número de UEs
        self.Ncf = 300          # Redes para avaliar (Snapshots de Larga Escala)
        self.Nbc = 100          # Blocos de coerência (Snapshots de Pequena Escala)

        self.fc = 3e9           # Frequência da portadora
        self.Bw = 20e6          # Largura de Banda
        self.Fn = 10**(9/10)    # Figura de ruído (9 dB em linear)
        self.T0 = 296.15        # Temperatura de ruído (K)

        self.Lx = 1000          # Comprimento da área de simulação (m)
        self.Ly = 1000          # Largura da área de simulação (m)
        self.hAP = 15           # Altura dos APs (m)
        self.hUE = 1.65         # Altura dos UEs (m)

        self.Pp = 200e-3        # Potência das sequências piloto (W)
        self.Pdl = 200e-3       # Potência da downlink (W)
        self.tau_p = 50         # Comprimento da sequência piloto

        # Cálculo da potência de ruído
        k_bolt = 1.381e-23
        self.Pn = k_bolt * self.T0 * self.Bw * self.Fn

    def generate_topology(self):
        """
        Gera as posições dos APs e UEs aleatoriamente na área definida
        """

        ap_pos = np.zeros((self.M, 3))
        ap_pos[:, 0] = np.random.uniform(-self.Lx/2, self.Lx/2, self.M)
        ap_pos[:, 1] = np.random.uniform(-self.Ly/2, self.Ly/2, self.M)
        ap_pos[:, 2] = self.hAP
        
        ue_pos = np.zeros((self.K, 3))
        ue_pos[:, 0] = np.random.uniform(-self.Lx/2, self.Lx/2, self.K)
        ue_pos[:, 1] = np.random.uniform(-self.Ly/2, self.Ly/2, self.K)
        ue_pos[:, 2] = self.hUE
        
        return ap_pos, ue_pos

    def calculate_large_scale_fading(self, ap_pos, ue_pos):
        """
        Calcula o desvanecimento em larga escala para as posições dos APs e UEs fornecidas.
        """

        # Distâncias (Slide 41)
        diff = ap_pos[:, np.newaxis, :] - ue_pos[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        dist = np.maximum(dist, 10) # Evita singularidade
        
        # Modelo de Perda de Percurso (ITU-R P.525-4)
        f_mhz = self.fc / 1e6
        PL_fs_1m = 20*np.log10(0.001) + 20*np.log10(f_mhz) + 32.4
        
        # Sombreamento
        shadowing = np.random.normal(0, 8, (self.M, self.K))

        PL_db = PL_fs_1m + 28*np.log10(dist/1e3) + shadowing

        # Conversão para linear
        Omega = 10**(-PL_db/10)
        return Omega

    def run_simulation(self):
        """
        Executa a simulação completa para o par (M, K) configurado.
        Retorna listas com SINR (dB) e Taxas (Mbps) acumuladas de todos os usuários.
        """

        # Listas para acumular dados de todos os snapshots e usuários
        sinr_ecsi_db_all = []
        sinr_pcsi_db_all = []
        rate_ecsi_all = []
        rate_pcsi_all = []
        
        print(f"Simulando M={self.M}, K={self.K}...")

        # Loop Monte Carlo Externo (Geometria/Larga Escala)
        for _ in range(self.Ncf):
            # 1. Gera posições e Larga Escala
            ap_pos, ue_pos = self.generate_topology()
            Omega = self.calculate_large_scale_fading(ap_pos, ue_pos)
            
            # 2. Coeficientes MMSE de Estimação de Canal
            num_c = np.sqrt(self.tau_p * self.Pp) * Omega
            den_c = (self.tau_p * self.Pp * Omega) + self.Pn
            c_mk = num_c / den_c
            
            # 3. Estatísticas do Canal Estimado (Gamma)
            gamma_mk = np.sqrt(self.tau_p * self.Pp) * Omega * c_mk
            
            # 4. Controle de Potência
            sum_gamma = np.sum(gamma_mk, axis=1) # Soma sobre usuários (K) -> vetor (M,)
            eta_m = 1.0 / np.maximum(sum_gamma, 1e-21)
            
            # Expande eta para matriz (M, K)
            eta_mk = eta_m[:, np.newaxis] * np.ones((self.M, self.K))
            
            # --- CÁLCULO ECSI (Estimated CSI / Estatístico) ---

            # SINR ECSI (Equação 37)
            # Numerador: Pd * ( soma_m( sqrt(eta_mk) * gamma_mk ) )^2
            # Potência recebida em K dos M APs
            term_num = np.sum(np.sqrt(eta_mk) * gamma_mk, axis=0) # (K,)
            num_ecsi = self.Pdl * (term_num**2)

            # Denominador: Pd * sum_k sum_m (eta_mk' * gamma_mk' * Omega_mk) + Pn
            # Potência total transmitida por M APs
            power_per_ap = np.sum(eta_mk * gamma_mk, axis=1) # (M,)

            # Sinal total recebido pelo usuário K
            interf_matrix = power_per_ap[:, np.newaxis] * Omega # (M, K)
            interf_total = np.sum(interf_matrix, axis=0) # (K,)

            denom_ecsi = self.Pdl * interf_total + self.Pn

            # Razão Sinal/Interferência + Ruído
            sinr_ecsi = num_ecsi / denom_ecsi # (K,)
            sinr_ecsi_db_all.extend(10*np.log10(sinr_ecsi))

            # Taxa ECSI (Equação 24)
            rate_ecsi = self.Bw * np.log2(1 + sinr_ecsi)
            rate_ecsi_all.extend(rate_ecsi / 1e6) # Mbps

            # --- CÁLCULO PCSI (Perfect CSI / Instantâneo) ---
            
            # SINR PCSI (Equação 21)
            # Loop Interno Monte Carlo (Pequena Escala) para a taxa ergódica

            # Coeficientes de pequena escala (M, K, Nbc)
            h_real = np.random.normal(0, 1/2, (self.M, self.K, self.Nbc)) + \
                     1j * np.random.normal(0, 1/2, (self.M, self.K, self.Nbc))

            # Canal g_mk = sqrt(Omega) * h_real, Broadcast Omega (M, K) -> (M, K, 1)
            g_mk_inst = np.sqrt(Omega)[:, :, np.newaxis] * h_real # (M, K, Nbc)

            # Ruído na estimativa (M, K, Nbc)
            noise_est = (np.random.normal(0, self.Pn, (self.M, self.K, self.Nbc)) + \
                         1j*np.random.normal(0, self.Pn, (self.M, self.K, self.Nbc))) * np.sqrt(self.Pn/2)

            # Potência recebida: Pdl * | sum_m (sqrt(eta)*conj(g_hat)*g_mk_inst) |^2
            yp = np.sqrt(self.tau_p * self.Pp) * g_mk_inst + noise_est
            g_hat_inst = c_mk[:, :, np.newaxis] * yp
            precoder = np.sqrt(eta_mk)[:, :, np.newaxis] * np.conj(g_hat_inst)
            term_cross = np.sum(g_mk_inst[:, :, np.newaxis, :] * precoder[:, np.newaxis, :, :], axis=0)

            # Potência que o usuário k recebe do sinal do usuário k'
            H_power = self.Pdl * (np.abs(term_cross)**2) # (K, K, Nbc)

            # Sinal Útil (Numerador): Quando quem transmite (tx) é igual a quem recebe (rx)
            signal_power = np.diagonal(H_power, axis1=0, axis2=1).T  # (K, Nbc)

            # Potência total recebida por cada usuário K
            total_received_power = np.sum(H_power, axis=1) # (K, Nbc)
            interference = total_received_power - signal_power

            denom_pcsi = interference + self.Pn

            sinr_pcsi = signal_power / denom_pcsi

            # Taxa PCSI (Equação 24)
            sinr_pcsi_db_all.extend(10*np.log10(sinr_pcsi.flatten()))
            rate_inst = self.Bw * np.log2(1 + sinr_pcsi)
            rate_avg_ue = np.mean(rate_inst, axis=1)
            rate_pcsi_all.extend(rate_avg_ue / 1e6)  # Mbps

        return sinr_ecsi_db_all, sinr_pcsi_db_all, rate_ecsi_all, rate_pcsi_all

def plot_ecdf(data_dict, title, xlabel, output_filename=None, xlim=None, ylim=None):
    """ Plota a CDF comparativa de ECSI e PCSI """

    plt.figure(figsize=(8, 6))

    styles = ['--', '-'] # ECSI tracejado, PCSI sólido
    colors = ['#1f77b4', '#d62728', '#ff7f0e']

    idx_color = 0
    for i, (label, data) in enumerate(data_dict.items()):
        sorted_data = np.sort(data)
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)

        style = styles[i % 2]
        color = colors[idx_color]

        plt.plot(sorted_data, yvals, linestyle=style, color=color, linewidth=2, label=label)

        if i % 2 == 1: # Mudou o par, muda a cor
            idx_color += 1

    if xlim:
        plt.xlim(xlim)
            
    plt.ylim(0, 1)
    plt.xlabel(xlabel)
    plt.ylabel('ECDF')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    if output_filename:
        plt.savefig(output_filename, dpi=300)
    plt.show()

def plot_topology(system_instance, title="Topologia da Rede (1 Snapshot)"):
    """ Plota a posição dos APs e UEs de uma única realização """
    ap_pos, ue_pos = system_instance.generate_topology()

    plt.figure(figsize=(8, 8))
    # APs como triângulos azuis
    plt.scatter(ap_pos[:, 0], ap_pos[:, 1], c='blue', marker='^', s=50, label='APs', alpha=0.7)
    # UEs como pontos vermelhos
    plt.scatter(ue_pos[:, 0], ue_pos[:, 1], c='red', marker='o', s=50, label='UEs', alpha=0.7)

    plt.xlim(-system_instance.Lx/2, system_instance.Lx/2)
    plt.ylim(-system_instance.Ly/2, system_instance.Ly/2)
    plt.xlabel('Posição X (m)')
    plt.ylabel('Posição Y (m)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig('topologia.png', dpi=300)
    plt.show()

if __name__ == "__main__":

    # ==========================================
    # CENÁRIO 1: Variando
    # Fixar K=20
    # ==========================================
    K_fixo = 20
    M_values = [100, 150, 200]

    results_sinr_M = {}
    results_rate_M = {}

    print("--- Iniciando Cenário 1: Variando M ---")
    for M in M_values:
        sys = CellFreeSystem(M=M, K=K_fixo)
        s_ecsi, s_pcsi, r_ecsi, r_pcsi = sys.run_simulation()
        
        results_sinr_M[f'ECSI - M={M}'] = s_ecsi
        results_sinr_M[f'PCSI - M={M}'] = s_pcsi
        
        results_rate_M[f'ECSI - M={M}'] = r_ecsi
        results_rate_M[f'PCSI - M={M}'] = r_pcsi

    # Plot SINR
    plot_ecdf(results_sinr_M, 
              f'CDF da SINR (K={K_fixo})', 
              'SINR (dB)', 
              'fig4_sinr.png',
              xlim=(-10, 30))

    # Plot Taxa Alcançável
    plot_ecdf(results_rate_M, 
              f'CDF da Taxa Alcançável (K={K_fixo})', 
              'Taxa Alcançável (Mbits/s)', 
              'fig5_rate.png',
              xlim=(0, 150))

    # ==========================================
    # CENÁRIO 2: Variando K (Figuras 6 e 7)
    # Fixar M=100
    # ==========================================
    M_fixo = 100
    K_values = [10, 20, 30]

    results_sinr_K = {}
    results_rate_K = {}

    print("\n--- Iniciando Cenário 2: Variando K ---")
    for K in K_values:
        sys = CellFreeSystem(M=M_fixo, K=K)
        s_ecsi, s_pcsi, r_ecsi, r_pcsi = sys.run_simulation()
        
        results_sinr_K[f'ECSI - K={K}'] = s_ecsi
        results_sinr_K[f'PCSI - K={K}'] = s_pcsi
        
        results_rate_K[f'ECSI - K={K}'] = r_ecsi
        results_rate_K[f'PCSI - K={K}'] = r_pcsi

    # Plot SINR
    plot_ecdf(results_sinr_K, 
              f'CDF da SINR (M={M_fixo})', 
              'SINR (dB)', 
              'fig6_sinr.png',
              xlim=(-10, 30))

    # Plot Taxa Alcançável
    plot_ecdf(results_rate_K, 
              f'CDF da Taxa Alcançável (M={M_fixo})', 
              'Taxa Alcançável (Mbits/s)',
              'fig7_rate.png',
              xlim=(0, 150))

    print("Simulação Finalizada! Verifique os arquivos PNG gerados.")