import numpy as np
import matplotlib.pyplot as plt
from channel_model import Channel3GPP

TIME_SAMPLES = int(1e5)
PULSE_WIDTH = [1e-7, 1e-5, 1e-3]

class Signal:
    def __init__(self, samples: np.ndarray, time_vector: np.ndarray, name: str = "Sinal"):
        self.samples = samples
        self.time_vector = time_vector
        self.name = name

def create_rectangular_pulse(pulse_width: float) -> Signal:
    t = np.linspace(0, 5 * PULSE_WIDTH, TIME_SAMPLES)
    pulse_samples = np.where(t <= pulse_width, 1.0, 0.0)
    return Signal(pulse_samples, t, name="TX Signal")

def apply_channel(channel: Channel3GPP, tx_signal: Signal) -> Signal:
    rx_samples = np.zeros_like(tx_signal.samples, dtype=np.complex128)
    
    n_paths = channel.n_paths
    powers = channel.multipath_powers
    delays = channel.multipath_delays
    dopplers = channel.doppler_shifts
    fc = channel.frequency_ghz.item() * 1e9
    dt = tx_signal.time_vector[1] - tx_signal.time_vector[0]

    # Loop para somar a contribuição de cada um dos N percursos
    for n in range(n_paths):
        alpha_n = np.sqrt(powers[n])
        tau_n = delays[n]
        nu_n = dopplers[n]

        # 1. Calcula a fase variante no tempo
        phase_term = 2 * np.pi * ((fc + nu_n) * tau_n - nu_n * tx_signal.time_vector)
        phasor = np.exp(-1j * phase_term)

        # 2. Gera o sinal transmitido atrasado, s(t - τ_n)
        delay_samples = int(round(tau_n / dt))
        delayed_tx_signal = np.roll(tx_signal.samples, delay_samples)
        if delay_samples > 0:
            delayed_tx_signal[:delay_samples] = 0

        # 3. Soma a contribuição deste percurso ao sinal recebido
        rx_samples += alpha_n * phasor * delayed_tx_signal
        
    return Signal(rx_samples, tx_signal.time_vector, name="Sinal Recebido")