import numpy as np
import matplotlib.pyplot as plt

# 1. GERAÇÃO DA CONSTELAÇÃO DE REFERÊNCIA
def get_constellation_points(order, distance):
    idx = np.arange(1, np.sqrt(order), 2) * distance
    amplitude = np.concatenate((-idx[::-1], idx))
    a, b = np.meshgrid(amplitude, amplitude)
    return (a + 1j * b).ravel()

# 2. DECODIFICAÇÃO DE MÍNIMA DISTÂNCIA (ML com CSI)
def dec_min_distance(received_symbols, constellation_ref, channel_gains):
    decisions = []
    # Itera sobre cada símbolo recebido 'r' e seu ganho de canal 'h' (beta)
    for r, h in zip(received_symbols, channel_gains):
        # DECISÃO ML: Comparar R com a constelação de referência afetada pelo canal (h * X)
        distances_sq = np.abs(r - h * constellation_ref)**2
        decisions.append(constellation_ref[np.argmin(distances_sq)])
    return np.array(decisions)


# FUNÇÃO DE PLOTAGEM (para visualizar o efeito do Rayleigh)
def plot_constellation(tx_symbols, rx_symbols, modulation_order):
    plt.figure(figsize=(6, 6))
    plt.scatter(tx_symbols.real, tx_symbols.imag, color='blue', label='Transmitido')
    plt.scatter(rx_symbols.real, rx_symbols.imag, color='red', s=1, label='Recebido')
    plt.title(f'Constelação {modulation_order:.0f}-QAM (SNR = {snr_db} dB)')
    plt.xlabel('Fase')
    plt.ylabel('Quadratura')
    plt.grid(True)
    plt.axhline(0, color='black', lw=0.5)
    plt.axvline(0, color='black', lw=0.5)
    plt.legend()
    plt.show()


# ==========================================================
if __name__ == "__main__":
    MODULATION_ORDER = np.array([4, 16, 64])
    N_SNR_SAMPLES = 10**3
    plt.figure(figsize=(10, 6))

    for M in MODULATION_ORDER:
        distance = np.sqrt(3/(2*(M-1)))

        # 1. Definir o range de SNR para a curva
        snr_db_values = np.arange(-30, -8, 2)
        simulated_ser = []
        constellation_ref = get_constellation_points(M, distance)

        print(f"Iniciando simulação de Monte Carlo para {M}-QAM em Rayleigh Fading (apenas simulação)...")

        for snr_db in snr_db_values:
            # 2. Parâmetros do Canal
            snr_linear = 10**(snr_db / 10)
            n0_variancia = 1/(2*snr_linear)
            sigma_noise = np.sqrt(n0_variancia)

            # Geração dos Ganhos de Canal e Ruído
            beta = np.random.rayleigh(scale=1.0, size=N_SNR_SAMPLES)
            # Ruído complexo (I + jQ)
            noise = (np.random.normal(0, sigma_noise, N_SNR_SAMPLES) + 
                    1j * np.random.normal(0, sigma_noise, N_SNR_SAMPLES))
            
            # Geração e Transmissão dos Símbolos
            tx_symbols = np.random.choice(constellation_ref, N_SNR_SAMPLES)
            
            # 3. Sinal Recebido (Rayleigh Fading)
            rx_rayleigh = (tx_symbols * beta) + noise 

            # Decodificação
            decisions_rayleigh = dec_min_distance(rx_rayleigh, constellation_ref, beta)

            # 4. Cálculo do Erro e SER
            erros_detectados = tx_symbols != decisions_rayleigh
            Ne = np.sum(erros_detectados)
            ser = Ne / N_SNR_SAMPLES
            simulated_ser.append(ser)

            print(f'SNR: {snr_db:02} dB | SER: {ser:.4f}')

        simulated_ser = np.array(simulated_ser)
    
        # Curva Simulação
        plt.semilogy(snr_db_values, simulated_ser, 
                    marker='o', linestyle='-', color='red', 
                    label=f'Simulação {M}-QAM (Rayleigh)')

        plt.title(f'Probabilidade de Erro de Símbolo (SER) para {M}-QAM')
        plt.xlabel('SNR (dB)')
        plt.ylabel('SER (Taxa de Erro de Símbolo)')
        plt.grid(True, which="both")
        plt.legend()
        # Ajusta os limites para melhor visualização da curva em escala logarítmica
        plt.ylim(1e-5, 1) 
        plt.xlim(snr_db_values.min(), snr_db_values.max())
        plt.show()


        # print(f'Relação de Sinal-Ruído (SNR): {snr_db} dB')
        # print(f'Número de símbolos transmitidos: {N_SNR_SAMPLES}')
        # print(f'Número de erros detectados (Rayleigh Fading): {Ne}')
        # print(f'Taxa de erro de símbolo (SER - Rayleigh Fading): {Ne / N_SNR_SAMPLES:.4f}')

        # plot_constellation(constellation_ref, rx_rayleigh, M)