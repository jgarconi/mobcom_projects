import numpy as np
import matplotlib.pyplot as plt

# Número de amostras de SNR (Nt)
N_SNR_SAMPLES = 10**5

# Variância (estudar o porquê de ser 1/2)
sigma = 1/2

# Limiar de SNR - eixo x variando de 10 em 10 de -30 a 30
gamma_th_db = np.arange(-30, 31, 5)
gamma_th_lin = 10**(gamma_th_db / 10)

# SNR média por símbolo (gamma_s barra)
snr_media_db = np.array([-20, 0, 20])
snr_media_lin = 10**(snr_media_db / 10)

# Envoltória do canal (Rayleigh)
x = np.random.normal(0, sigma, N_SNR_SAMPLES)
y = np.random.normal(0, sigma, N_SNR_SAMPLES)
beta = np.sqrt(x**2 + y**2)

plt.figure(figsize=(8, 6))

# Loop sobre cada SNR média
for g_med_db, g_med in zip(snr_media_db, snr_media_lin):
    # SNR instantânea
    snr_inst = g_med * beta**2

    # Estimativa experimental da probabilidade de outage
    p_out_exp = []
    for g_th in gamma_th_lin:
        interrupcao = snr_inst <= g_th
        p_out_exp.append(np.sum(interrupcao) / N_SNR_SAMPLES)
    p_out_exp = np.array(p_out_exp)

    # Solução analítica para canal Rayleigh
    p_out_analitica = 1 - np.exp(-gamma_th_lin / g_med)

    # Plotar ambas
    plt.semilogy(gamma_th_db, p_out_analitica, 'o-', label=f'Analítica ({g_med_db} dB)')
    plt.semilogy(gamma_th_db, p_out_exp, 's--', label=f'Experimental ({g_med_db} dB)')

# Configurações do gráfico
plt.grid(True, which="both", ls="--", lw=0.5)
plt.xlabel('Limiar de SNR γ_th (dB)')
plt.ylabel('Probabilidade de Outage P_out')
plt.title('Probabilidade de Outage - Canal Rayleigh')
plt.legend()
plt.tight_layout()
plt.show()
