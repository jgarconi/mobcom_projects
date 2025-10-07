import numpy as np
import matplotlib.pyplot as plt
from channel_model import Channel3GPP
from plot_distributions import PlotChannelModel

TIME_SAMPLES = int(1e5)
PULSE_WIDTHS = [1e-7, 1e-5, 1e-3]

def create_rectangular_pulse(pulse_width: float) -> tuple[np.ndarray, np.ndarray]:
    """Gera um pulso retangular de amplitude unitária."""
    total_duration = 5 * pulse_width
    t = np.linspace(0, total_duration, TIME_SAMPLES)
    pulse = np.where(t <= pulse_width, 1.0, 0.0)
    return t, pulse

def apply_channel(channel: Channel3GPP, tx_signal: np.ndarray, time_vector: np.ndarray) -> np.ndarray:
    """Aplica os efeitos do canal a um sinal transmitido."""
    rx_signal = np.zeros_like(tx_signal, dtype=np.complex128)
    dt = time_vector[1] - time_vector[0]
    fc_hz = channel.frequency_ghz * 1e9

    for n in range(channel.n_paths):
        delay = channel.multipath_delays[n]
        power = channel.multipath_powers[n]
        doppler = channel.doppler_shifts[n]
        amplitude = np.sqrt(power)

        delay_samples = int(round(delay / dt))
        tx_delayed_signal = np.roll(tx_signal, delay_samples)
        if delay_samples > 0:
            tx_delayed_signal[:delay_samples] = 0
            
        phase = 2 * np.pi * ((fc_hz + doppler) * delay - doppler * time_vector)
        phasor = np.exp(-1j * phase)

        rx_signal += amplitude * phasor * tx_delayed_signal
        
    return rx_signal

if __name__ == '__main__':
    meu_canal = Channel3GPP(
        scenario = "umi_nlos",
        frequency_ghz = 3.0,
        n_paths = 100,
        rx_velocity_mps = 5,
        rx_azimuth_deg = 90,
        rx_elevation_deg = 0
    )
    
    meu_canal.generate_channel()
    plotter = PlotChannelModel(meu_canal)
    
    kappa_hz = np.logspace(3, 8, 5000)
    corr_freq = meu_canal.calculate_freq_correlation(kappa_hz)
    bc_95 = Channel3GPP.get_coherence_value(corr_freq, kappa_hz, 0.95)
    bc_90 = Channel3GPP.get_coherence_value(corr_freq, kappa_hz, 0.90)
    plotter.plot_coherence_bandwidth(kappa_hz, corr_freq, bc_95, bc_90, meu_canal)

    sigma_s = np.logspace(-5, -1, 5000)
    corr_time = meu_canal.calculate_time_correlation(sigma_s)
    tc_95 = Channel3GPP.get_coherence_value(corr_time, sigma_s, 0.95)
    tc_90 = Channel3GPP.get_coherence_value(corr_time, sigma_s, 0.90)
    plotter.plot_coherence_time(sigma_s, corr_time, tc_95, tc_90, meu_canal)

    plotter.plot_3d_directions()
    plotter.plot_azimuth_spread()
    plotter.plot_elevation_spread()
    plotter.plot_doppler_spectrum()
    plotter.plot_power_delay_profile()
    PlotChannelModel.plot_statistical_delay_spread(meu_canal.scenario)

    for delta_t in PULSE_WIDTHS:
        t, tx_pulse = create_rectangular_pulse(pulse_width=delta_t)
        rx_signal = apply_channel(meu_canal, tx_pulse, t)

        plt.figure(figsize=(10, 6))
        plt.plot(t, tx_pulse, label='Sinal Transmitido', lw=2)
        plt.plot(t, np.abs(rx_signal), label='Sinal Recebido (Magnitude)', lw=2)
        
        bw_mhz = (1 / delta_t)
        plt.title(f'Sinal Recebido vs. Transmitido (δt = {delta_t*1e9:.0f} ns, Bw ≈ {bw_mhz:.2f} MHz)', fontsize=15)
        plt.xlabel('Tempo (s)', fontsize=12)
        plt.ylabel('Magnitude', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.show()