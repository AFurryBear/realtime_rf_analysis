import argparse
import numpy as np

# Create the parser
parser = argparse.ArgumentParser(description='Example script to demonstrate argparse usage.')

# Add arguments
parser.add_argument('--data_path', type=str, help='Path to the file')
parser.add_argument('--stim_path', type=str, help='recording foldername')
parser.add_argument('--start_chan', type=int, help='start index of coi')
parser.add_argument('--end_chan', type=int, help='end index of coi')
parser.add_argument('--ttl_chan', type=int, help='ttl channel index')
parser.add_argument('--channel1', type=int, default=np.nan, help='channel to be plotted')
parser.add_argument('--channel2', type=int, default=np.nan, help='channel to be plotted')
parser.add_argument('--stim_duration', type=int, default=1, help='stimuli_duration')

# Parse the arguments
args = parser.parse_args()

# Accessing arguments
data_path = args.data_path
stim_path = args.stim_path
start_chan = args.start_chan
end_chan = args.end_chan
ttl_chan = args.ttl_chan
coi = np.array([args.channel1, args.channel2])
stimuli_duration = args.stim_duration

# Example usage
# print(f"File path: {data_path}")
# print(f"Log level: {log_level}")
# print(f"Output directory: {output_dir}")
# print(f"Maximum count: {max_count}")
# print(f"Verbose: {verbose}")

import os
import matplotlib.pyplot as plt
from toolbox_ap_process import filter_signal, process_ttl_data
from toolbox_load_data import load_ephyData, load_ttlData
# import time

# # Start timing
# start_time = time.time() 
foldername = os.path.split(data_path)[1]

ap_path = data_path+'/'+foldername+'_imec0/'+foldername+'_t0.imec0.ap.bin'
ttl_path = data_path+'/'+foldername+'_t0.nidq.bin'

## load stim data
import scipy.io as sio
stim_mat = sio.loadmat(stim_path)

nRepeat = stim_mat['sParams'].item()[5][0][0]
stim_ID = stim_mat['sParams'].item()[-1].flatten()
num_pulse = len(stim_ID)
num_pos = max(stim_ID)

## load TTL file
ttl_data, sRate_ttl, meta_ttl = load_ttlData(ttl_path,ttl_chan)
tStimOnset, tStimOffset, tStart, tEnd = process_ttl_data(ttl_data, sRate_ttl, stimuli_duration)
print('%d stimuli were shown in this recording.'%len(tStimOnset))
if len(tStimOnset)!=num_pulse:
    print('WARNING! #Pulse NOT MATCH!!')

## load neuropixel data
chanList = np.arange(start_chan,end_chan)
selectData_ap,sRate_ap,meta_ap = load_ephyData(ap_path, tStart, tEnd, chanList)
# filter neuropixel data
xf = filter_signal(selectData_ap, sRate_ap, 500, 4000)

from toolbox_ap_process import detect_spikes
## detect spikes from data
(s, t), threshold, s_all, nSpk = detect_spikes(xf, sRate_ap, N=4,lockout=10)
print('from chan %03d to %03d: found %d spikes with th %.2f'%(chanList[0],chanList[-1],len(s_all),threshold))


from toolbox_ap_process import get_spkTrain
## plot spike trains
spike_counts, bin_size = get_spkTrain(t)

time_series = np.linspace(0,tEnd-tStart,len(spike_counts))
tStimOnset = tStimOnset.flatten()
tStimOffset = tStimOffset.flatten()

plt.figure(figsize=(15,3))
for i in range(len(tStimOnset)):
    plt.fill_betweenx([0,np.max(spike_counts)], tStimOnset[i], tStimOffset[i], color="grey", alpha=1)
plt.plot(time_series, spike_counts,color='black',linewidth=.5,alpha=.8)
plt.savefig(data_path+'/'+'nSpk_with_time.png')

## plot nSpk during stim_on and stim_off
from toolbox_plot import get_colors,get_cm, plot_multiple_rawdata,plot_nSpk_contrast
from toolbox_ap_process import prepare_rawData_condition, find_stimModulated_channel, prepare_nSpk_condition

nSpk_pos_0_norm, nSpk_pos_1_norm = prepare_nSpk_condition(spike_counts, tStimOnset, tStimOffset,stimuli_duration, bin_size, nRepeat, num_pos)

time_axis_nSpk = np.linspace(-stimuli_duration,stimuli_duration,int(2*stimuli_duration*1000/bin_size),endpoint=True)

fig, axes = plt.subplots(nrows=3,ncols=3,sharey = True, sharex=True, figsize = (10,10))
for i in range(num_pos):
    row, col = i%3,i//3    
    axes[row,col] = plot_nSpk_contrast(axes[row,col], time_axis_nSpk, nSpk_pos_0_norm[:,i,:].T, [0,stimuli_duration])
plt.savefig(data_path+'/'+'nSpk_with_contrast0.png')
plt.close('all')

fig, axes = plt.subplots(nrows=3,ncols=3,sharey = True, sharex=True, figsize = (10,10))
for i in range(num_pos):
    row, col = i%3,i//3    
    axes[row,col] = plot_nSpk_contrast(axes[row,col], time_axis_nSpk, nSpk_pos_1_norm[:,i,:].T, [-stimuli_duration,0])
plt.savefig(data_path+'/'+'nSpk_with_contrast1.png')
plt.close('all')

## calculate difference between stim_on and stim_off
temp_0 = nSpk_pos_0_norm.reshape(nRepeat,num_pos,2,int(stimuli_duration*1000/bin_size))
temp_1 = nSpk_pos_1_norm.reshape(nRepeat,num_pos,2,int(stimuli_duration*1000/bin_size))

diff_0 = (temp_0[:,:,1,:] - temp_0[:,:,0,:]).mean(axis=0).flatten()
diff_1 = (temp_1[:,:,0,:] - temp_1[:,:,1,:]).mean(axis=0).flatten()

cm = get_cm()
colors_0 = get_colors(cm, diff_0)
colors_1 = get_colors(cm, diff_1)

## plot raw data during stim_on and stim_off
tStimOnset_cond = tStimOnset.reshape((3,9))
tStimOffset_cond = tStimOffset.reshape((3,9))
time_axis = np.arange(-stimuli_duration,stimuli_duration,step=1/sRate_ap)

fig, axes = plt.subplots(nrows=3,ncols=3,sharey = True, sharex=True, figsize = (10,10))
for i in range(tStimOnset_cond.shape[1]):
    start_time = tStimOnset_cond[:,i] - stimuli_duration
    stop_time = tStimOnset_cond[:,i] + stimuli_duration
    vol_data = prepare_rawData_condition(xf, start_time, stop_time, sRate_ap)
    voi_average = vol_data.mean(axis=1).mean(axis=0)
    coi_id = np.where(np.isnan(coi))[0]
    if len(coi_id)>0:
        coi[coi_id] = find_stimModulated_channel(vol_data[:,:,int(stimuli_duration*sRate_ap):], vol_data[:,:,:int(stimuli_duration*sRate_ap)], len(coi_id))
    coi = coi.astype(int)
    vol_data = np.transpose(vol_data[coi,:,:], (1, 0, 2)).reshape((-1, ) + vol_data.shape[2:])
    
    avg_data = np.vstack([vol_data.mean(axis=0),voi_average])
    row, col = i%3,i//3
    axes[row,col],data_arr = plot_multiple_rawdata(axes[row,col], time_axis, vol_data, avg_data, sRate_ap, colors_0[i], diff_0[i], coi+chanList[0], 60, [0,stimuli_duration])
plt.savefig(data_path+'/'+'rawData_with_contrast0.png')
plt.close('all')


fig, axes = plt.subplots(nrows=3,ncols=3,sharey = True, sharex=True, figsize = (10,10))
for i in range(tStimOnset_cond.shape[1]):
    start_time = tStimOffset_cond[:,i] - stimuli_duration
    stop_time = tStimOffset_cond[:,i] + stimuli_duration
    vol_data = prepare_rawData_condition(xf, start_time, stop_time, sRate_ap)
    voi_average = vol_data.mean(axis=1).mean(axis=0)
    coi_id = np.where(np.isnan(coi))[0]
    if len(coi_id)>0:
        coi[coi_id] = find_stimModulated_channel(vol_data[:,:,int(stimuli_duration*sRate_ap):], vol_data[:,:,:int(stimuli_duration*sRate_ap)], len(coi_id))
    coi = coi.astype(int)
    vol_data = np.transpose(vol_data[coi,:,:], (1, 0, 2)).reshape((-1, ) + vol_data.shape[2:])
    avg_data = np.vstack([vol_data.mean(axis=0),voi_average])
    row, col = i%3,i//3
    axes[row,col],data_arr = plot_multiple_rawdata(axes[row,col], time_axis, vol_data, avg_data, sRate_ap, colors_1[i], diff_1[i], coi+chanList[0], 60, [-stimuli_duration,0])
plt.savefig(data_path+'/'+'rawData_with_contrast1.png')
plt.close('all')

# end_time = time.time() 

# running_time = end_time - start_time

# print("Running time: %d seconds"%(running_time[2]))