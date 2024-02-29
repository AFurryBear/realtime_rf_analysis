
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors

def plot_peaks(axs, time_series, data, fs, peak_times, thr):
    time_points = (time_series*fs).astype(int)
    peak_index = (peak_times*fs).astype(int)
    axs.plot(time_series, data[time_points],linewidth=.5)
    axs.scatter(peak_times, data[peak_index], c='red',s=10)
    axs.plot(np.ones_like(data)*thr, "--", color="gray")
    axs.plot(-1*np.ones_like(data)*thr, "--", color="gray")
    axs.set_xlim((time_series[0], time_series[-1]))
    # axs.set_ylabel('V')
    return axs

def get_cm():
    # Define the colors for the custom colormap (coolwarm but with grey in the center)
    colors = [(0, 0, 1), (0.5, 0.5, 0.5), (1, 0, 0)]  # R -> Grey -> B
    n_bins = 100  # Number of bins in the colormap
    cmap_name = 'custom_coolwarm'

    # Create the colormap
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    return cm

def get_colors(cm,values):
    # Example array of values
    values = values.flatten()


    # Normalize the values to the [0, 1] range
    norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())


    # Apply the colormap to the normalized values
    colors = cm(norm(values))

    return colors
