import matplotlib.pyplot as plt

# TODO: update plots to exclude the LOS component if K_R = 0

class PlotChannelModel:
    def __init__(self, channel_model):
        self.channel_model = channel_model

    def plot_pdp(self):
        """Plots all available multipath profiles: powers, azimuth angles, and elevation angles (if available)."""
        channel = self.channel_model
        if channel.multipath_powers is None:
            print("Run the 'generate_channel()' method first.")
            return

        # Plot Power Delay Profile (PDP)
        plt.figure(figsize=(10, 6))
        plt.stem(channel.multipath_delays * 1e6, channel.multipath_powers, linefmt='k-', markerfmt='k^', basefmt=' ')
        if channel.kr_factor[0] > 0:
            plt.stem(channel.multipath_delays[0] * 1e6, channel.multipath_powers[0], linefmt='b-', markerfmt='b^', basefmt=' ')
        plt.yscale('log')
        plt.title(f'Power Delay Profile (PDP) - {channel.scenario}')
        plt.xlabel('Delay (μs)')
        plt.ylabel('Normalized Power (αn²)')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.show()

        # Plot Azimuth Angles vs Power if available
        if hasattr(channel, 'multipath_azimuth_angles') and channel.multipath_azimuth_angles is not None:
            plt.figure(figsize=(10, 6))
            plt.stem(channel.multipath_azimuth_angles, channel.multipath_powers, linefmt='k-', markerfmt='k^', basefmt=' ')
            plt.title(f'Multipath Azimuth Angles vs Power - {channel.scenario}')
            if channel.kr_factor[0] > 0:
                plt.stem(channel.multipath_azimuth_angles[0], channel.multipath_powers[0], linefmt='b-', markerfmt='b^', basefmt=' ')
            plt.xlabel('Azimuth Angle (°)')
            plt.ylabel('Normalized Power (αn²)')
            plt.yscale('log')
            plt.grid(True, which="both", ls="--", alpha=0.5)
            plt.show()

        # Plot Elevation Angles vs Power if available
        if hasattr(channel, 'multipath_elevation_angles') and channel.multipath_elevation_angles is not None:
            plt.figure(figsize=(10, 6))
            plt.stem(channel.multipath_elevation_angles, channel.multipath_powers, linefmt='k-', markerfmt='k^', basefmt=' ')
            plt.title(f'Multipath Elevation Angles vs Power - {channel.scenario}')
            if channel.kr_factor[0] > 0:
                plt.stem(channel.multipath_elevation_angles[0], channel.multipath_powers[0], linefmt='b-', markerfmt='b^', basefmt=' ')
            plt.xlabel('Elevation Angle (°)')
            plt.ylabel('Normalized Power (αn²)')
            plt.yscale('log')
            plt.grid(True, which="both", ls="--", alpha=0.5)
            plt.show()