
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

def plot_nSpk_contrast(axs, time_axis, data, stim_on):
    d = (data - data.mean(axis=0))/data.std(axis=0)
    axs.plot(time_axis.T,d,c='black')
    axs.fill_betweenx([-3,3],stim_on[0],stim_on[1],color="lightgrey", alpha=.3)
    return axs


def plot_multiple_rawdata(axs, time_axis, data_arr, average_data, fs, linecolor, chan_id, offset=30, stim_tLabel=[0,1]):
    yoffset_mat = offset * np.arange(1,len(data_arr)+1)[:,np.newaxis] * np.ones(data_arr.shape)
    data_arr = data_arr + yoffset_mat
    time_axis_new = np.tile(time_axis,data_arr.shape[0]).reshape(data_arr.shape[0], len(time_axis))
    data_arr[:,0:5] = np.nan
    time_axis_new[:,0:5] = np.nan
    axs.plot(time_axis_new.T, data_arr.T, linewidth=.2, c='black')
    
    axs.plot(time_axis, average_data[0], linewidth=.2, c=linecolor)
    axs.plot(time_axis, average_data[1]-offset, linewidth=.2, c=linecolor)
    
    axs.axvline(x=0,linestyle='--',c='lightgrey')
    axs.fill_betweenx([-1.5*offset,offset*(len(data_arr)+1.2)],stim_tLabel[0],stim_tLabel[1],color="lightgrey", alpha=.3)
    
    # Remove x and y ticks
    axs.set_yticks([])
    axs.text(time_axis[0], offset*(len(data_arr)+1.4), 'chan:%d,%d'%(chan_id[0],chan_id[1]))
    # Optional: Remove spines
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['left'].set_visible(False)
    axs.spines['bottom'].set_visible(False)
    
    return axs, data_arr
