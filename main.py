# Em um arquivo como main.py

from class_3gpp import Channel3GPP
from plot_distributions import PlotChannelModel

# 1. Crie e gere o modelo de canal
my_channel = Channel3GPP(scenario="umi_nlos", frequency_ghz=3, n_paths=100)
my_channel.generate_channel()

# 2. Passe o canal gerado para o plotter
plotter = PlotChannelModel(my_channel)

# 3. Use o plotter para criar os gráficos
plotter.plot_power_delay_profile()
plotter.plot_delay_spread_profile()
plotter.plot_elevation_angles()
plotter.plot_azimuth_angles()
plotter.plot_3d_directions()