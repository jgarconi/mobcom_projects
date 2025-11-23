import numpy as np
import matplotlib.pyplot as plt
import marcumq as q

# Número de amostras de SNR (Nt)
N_SNR_SAMPLES = 10**5

# Variância (Rice normalizado (beta): 2*sigma² = 1)
kr_factor = np.array([0.1, 1, 10])

# Limiar de SNR - eixo x variando de 10 em 10 de -30 a 30
gamma_th_db = np.arange(-30, 31, 5)
gamma_th_lin = 10**(gamma_th_db / 10)

# SNR média por símbolo (gamma_s barra)
snr_media_db = np.array([-20, 0, 20])
snr_media_lin = 10**(snr_media_db / 10)

for kr in kr_factor:
    plt.figure(figsize=(8, 6))

    # Cálculo da probabilidade de outage para canal Rice
    for g_med_db, g_med in zip(snr_media_db, snr_media_lin):
        # SNR instantânea por símbolo
        los_power = kr / (kr + 1)
        mu = np.sqrt(los_power)  # valor médio da portadora
        sigma = np.sqrt(1/(2*(kr + 1))) 

        # Envoltória do canal (Rice normalizado)
        x = np.random.normal(0, sigma, N_SNR_SAMPLES)
        y = np.random.normal(0, sigma, N_SNR_SAMPLES)

        beta = np.sqrt((x + mu)**2 + y**2)

        snr_inst = g_med * beta**2

        # Estimativa experimental da probabilidade de outage
        p_out_exp = []
        p_out_analitica = []
        for g_th in gamma_th_lin:
            interrupcao = snr_inst <= g_th
            p_out_exp.append(np.sum(interrupcao) / N_SNR_SAMPLES)
            # Solução analítica para canal Rice
            p_out_analitica.append(1 - q.marcumq(nu=1, a=np.sqrt(2*kr), b=np.sqrt((2*(kr+1)*g_th)/g_med)))

        p_out_exp = np.array(p_out_exp)
        p_out_analitica = np.array(p_out_analitica)

        # Plotar ambas
        plt.semilogy(gamma_th_db, p_out_analitica, 'o-', label=f'Analítica ({g_med_db} dB)')
        plt.semilogy(gamma_th_db, p_out_exp, 's--', label=f'Experimental ({g_med_db} dB)')

    # Configurações do gráfico
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.xlabel('Limiar de SNR γ_th (dB)')
    plt.ylabel('Probabilidade de Outage P_out')
    plt.title(f'Probabilidade de Outage - Canal Rice (K_r = {kr})')
    plt.legend()
    plt.tight_layout()
    plt.show()
