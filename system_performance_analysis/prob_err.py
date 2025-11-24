import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# --- 1. CLASSE DE SISTEMA QAM ---

class QAMSystem:
    """
    Representa o sistema de modulação M-QAM, incluindo a constelação, 
    canais (Rayleigh, AWGN) e o detector de Mínima Distância.
    """
    def __init__(self, M):
        self.M = M
        self.distance = np.sqrt(3 / (2 * (M - 1)))
        self.constellation_ref = self._generate_constellation_points()

    def _generate_constellation_points(self):
        """ Gera os M pontos da constelação normalizada (Es=1). """
        order = np.sqrt(self.M)
        idx = np.arange(1, order, 2) * self.distance
        amplitude = np.concatenate((-idx[::-1], idx))
        a, b = np.meshgrid(amplitude, amplitude)
        return (a + 1j * b).ravel()

    def transmit(self, N_samples):
        """ Gera N_samples de símbolos QAM aleatórios. """
        return np.random.choice(self.constellation_ref, N_samples)

    def _generate_awgn(self, snr_linear, N_samples):
        """ Gera o ruído AWGN complexo para uma dada SNR. """
        n0_variancia = 1 / (2 * snr_linear)
        sigma_noise = np.sqrt(n0_variancia)
        
        # Ruído AWGN Complexo (I + jQ)
        noise = (np.random.normal(0, sigma_noise, N_samples) + 
                 1j * np.random.normal(0, sigma_noise, N_samples))
        return noise

    def receive_awgn(self, tx_symbols, snr_linear, N_samples):
        """ Simula a transmissão através do canal AWGN puro. """
        noise = self._generate_awgn(snr_linear, N_samples)
        # Ganho do canal AWGN é sempre 1
        beta = np.ones(N_samples) 
        rx_symbols = tx_symbols + noise
        return rx_symbols, beta

    def receive_rayleigh(self, tx_symbols, snr_linear, N_samples):
        """ Simula a transmissão através do canal Rayleigh com AWGN. """
        noise = self._generate_awgn(snr_linear, N_samples)
        # Ganhos do canal Rayleigh (aleatórios)
        beta = np.random.rayleigh(scale=1.0, size=N_samples)
        rx_symbols = (tx_symbols * beta) + noise
        return rx_symbols, beta

    def decode_ml_csi(self, received_symbols, channel_gains):
        """ Decodificação de Mínima Distância (ML) assumindo CSI perfeito. """
        decisions = []
        
        # Compara r com o ponto de referência desvanecido (h * X)
        for r, h in zip(received_symbols, channel_gains):
            distances_sq = np.abs(r - h * self.constellation_ref)**2
            decisions.append(self.constellation_ref[np.argmin(distances_sq)])
            
        return np.array(decisions)

# --- 2. FUNÇÕES TEÓRICAS ---

def theoretical_ser_awgn(snr_linear, M_order):
    """ Calcula a SER teórica aproximada para M-QAM em canal AWGN. """
    log2M = np.log2(M_order)
    arg = np.sqrt( (3 * log2M / (M_order - 1)) * snr_linear )
    
    # Função Q(x) = 0.5 * erfc(x / sqrt(2))
    Q_func = 0.5 * erfc(arg / np.sqrt(2)) 
    
    SER = 4 * (1 - 1/np.sqrt(M_order)) * Q_func
    return SER

def theoretical_ser_rayleigh(snr_linear_avg, M_order):
    """ Calcula a SER teórica aproximada para M-QAM em canal Rayleigh Fading. """
    P = (1 - 1 / np.sqrt(M_order))
    A = (3 * np.log2(M_order)) / (2 * (M_order - 1))
    
    # Fórmula integrada de SER média para Rayleigh
    term_inside_sqrt = A * snr_linear_avg / (1 + A * snr_linear_avg)
    Q_rayleigh = 0.5 * (1 - np.sqrt(term_inside_sqrt))
    
    SER = 4 * P * Q_rayleigh
    return SER


# --- 3. FUNÇÃO DE PLOTAGEM MODIFICADA ---

def plot_ser_curve(M, snr_db_values, results):
    """ Plota todas as curvas de SER (Simulação e Teórica) na mesma imagem. """
    
    plt.figure(figsize=(10, 6))
    
    # Plot Simulação Rayleigh
    plt.semilogy(snr_db_values, results['simulated_rayleigh'], 
                 marker='o', linestyle=None, color='red', 
                 label=f'Simulação {M}-QAM (Rayleigh)')
    
    # Plot Teórico Rayleigh
    plt.semilogy(snr_db_values, results['theoretical_rayleigh'], 
                 marker='', linestyle='--', color='darkred', 
                 label=f'Teórico {M}-QAM (Rayleigh)')
    
    # Plot Simulação AWGN
    plt.semilogy(snr_db_values, results['simulated_awgn'], 
                 marker='s', linestyle=None, color='green', 
                 label=f'Simulação {M}-QAM (AWGN)')
    
    # Plot Teórico AWGN
    plt.semilogy(snr_db_values, results['theoretical_awgn'], 
                 marker='', linestyle='--', color='darkgreen', 
                 label=f'Teórico {M}-QAM (AWGN)')
    
    plt.title(f'Probabilidade de Erro de Símbolo (SER) para {M}-QAM')
    plt.xlabel('SNR (dB)')
    plt.ylabel('SER (Taxa de Erro de Símbolo)')
    plt.grid(True, which="both")
    plt.legend()
    plt.ylim(1e-5, 1)
    plt.xlim(snr_db_values.min(), snr_db_values.max())
    # 
    plt.show()

# --- 4. EXECUÇÃO DA SIMULAÇÃO DE MONTE CARLO ---

def run_monte_carlo_simulation(M, snr_db_range, N_samples):
    """ Executa o loop de Monte Carlo para gerar a curva SER para AWGN e Rayleigh. """
    
    system = QAMSystem(M)
    # Dicionário para armazenar todos os resultados
    results = {
        'simulated_rayleigh': [],
        'simulated_awgn': []
    }
    
    print(f"Iniciando simulação de Monte Carlo para {M}-QAM (AWGN e Rayleigh)...")

    for snr_db in snr_db_range:
        snr_linear = 10**(snr_db / 10)
        tx_symbols = system.transmit(N_samples)
        
        # --- SIMULAÇÃO CANAL AWGN ---
        rx_awgn, beta_awgn = system.receive_awgn(tx_symbols, snr_linear, N_samples)
        decisions_awgn = system.decode_ml_csi(rx_awgn, beta_awgn)
        ser_awgn = np.sum(tx_symbols != decisions_awgn) / N_samples
        results['simulated_awgn'].append(ser_awgn)
        
        # --- SIMULAÇÃO CANAL RAYLEIGH ---
        rx_rayleigh, beta_rayleigh = system.receive_rayleigh(tx_symbols, snr_linear, N_samples)
        decisions_rayleigh = system.decode_ml_csi(rx_rayleigh, beta_rayleigh)
        ser_rayleigh = np.sum(tx_symbols != decisions_rayleigh) / N_samples
        results['simulated_rayleigh'].append(ser_rayleigh)
        
        print(f'SNR: {snr_db:02} dB | SER_AWGN: {ser_awgn:.4e} | SER_Rayleigh: {ser_rayleigh:.4e}')
        
    return results

# --- CONFIGURAÇÃO E CHAMADA PRINCIPAL ---

if __name__ == '__main__':
    # Configurações
    MODULATION_ORDER = 64
    SNR_DB_RANGE = np.arange(-30, 30, 5)
    SAMPLES_PER_SNR_POINT = 5 * 10**4 

    # 1. Rodar Simulação
    simulated_results = run_monte_carlo_simulation(
        MODULATION_ORDER, 
        SNR_DB_RANGE, 
        SAMPLES_PER_SNR_POINT
    )
    
    # 2. Calcular a Curva Teórica para ambos os canais
    snr_linear_avg = 10**(SNR_DB_RANGE / 10)
    simulated_results['theoretical_rayleigh'] = theoretical_ser_rayleigh(snr_linear_avg, MODULATION_ORDER)
    simulated_results['theoretical_awgn'] = theoretical_ser_awgn(snr_linear_avg, MODULATION_ORDER)
    
    # 3. Gerar Gráfico
    plot_ser_curve(MODULATION_ORDER, SNR_DB_RANGE, simulated_results)