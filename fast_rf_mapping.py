import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Example script to demonstrate argparse usage.')

# Add arguments
parser.add_argument('--data_path', type=str, help='Path to the file')
parser.add_argument('--stim_path', type=str, help='recording foldername')
parser.add_argument('--start_chan', type=int, help='start index of coi')
parser.add_argument('--end_chan', type=int, help='end index of coi')
parser.add_argument('--ttl_chan', type=int, help='ttl channel index')
parser.add_argument('--channel1', type=int, default=False, help='channel to be plotted')
parser.add_argument('--channel2', type=int, default=False, help='channel to be plotted')
parser.add_argument('--stim_duration', type=int, default=1, help='stimuli_duration')

# Parse the arguments
args = parser.parse_args()

# Accessing arguments
data_path = args.data_path
stim_path = args.stim_path
start_chan = args.start_chan
end_chan = args.end_chan
ttl_chan = args.ttl_chan
coi = [args.channel1, args.channel2]
stimuli_duration = args.stim_duration

# Example usage
print(f"File path: {data_path}")
print(f"Log level: {log_level}")
print(f"Output directory: {output_dir}")
print(f"Maximum count: {max_count}")
print(f"Verbose: {verbose}")

import os
import numpy as np
import matplotlib.pyplot as plt
from toolbox_ap_process import filter_signal
from toolbox_load_data import load_ephyData, load_ttlData

foldername = os.path.split(data_path)[1]

ap_path = data_path+'/'+foldername+'_imec0/'+foldername+'_t0.imec0.ap.bin'
ttl_path = data_path+'/'+foldername+'_t0.nidq.bin'

## load stim data
import scipy.io as sio
nRepeat = stim_mat['sParams'].item()[5][0][0]
stim_ID = stim_mat['sParams'].item()[-1].flatten()
num_pulse = len(stim_ID)
num_pos = max(stim_ID)

## load TTL file
ttl_data, sRate_ttl, meta_ttl = load_ttlData(ttl_path,ttl_chan)

ttl_data = np.where(ttl_data>20000,30000,0)# ugly way to denoise the frame

T_before_onset = 3
T_after_offset = 3

stimuli_ontid = np.where(np.diff(ttl_data)==30000)[0]+1
stimuli_offtid = stimuli_ontid+sRate_ttl*stimuli_duration

tStart =  stimuli_ontid[0]/sRate_ttl - T_before_onset
tEnd = stimuli_offtid[-1]/sRate_ttl + T_after_offset

crop_offset = stimuli_ontid[0] - int(sRate_ttl)*T_before_onset
tID_stim_on = (stimuli_ontid - crop_offset).astype(int)
tID_stim_off = (stimuli_offtid - crop_offset).astype(int)

tStimOnset = tID_stim_on/int(sRate_ttl)
tStimOffset = tID_stim_off/int(sRate_ttl)

print('%d stimuli were shown in this recording.'%len(tID_stim_off))

chanList = np.arange(start_chan,end_chan)
selectData_ap,sRate_ap,meta_ap = load_ephyData(ap_path, tStart, tEnd, chanList)
xf = filter_signal(selectData_ap, sRate_ap, 500, 4000)

from toolbox_ap_process import detect_spikes
(s, t), threshold, s_all, nSpk = detect_spikes(xf, sRate_ap, N=4,lockout=1)
print('from chan %03d to %03d: found %d spikes with th %.2f'%(chanList[0],chanList[-1],len(s_all),threshold))

from toolbox_ap_process import get_spkTrain

spike_counts, bin_size = get_spkTrain(t)

time_series = np.linspace(0,tEnd-tStart,len(spike_counts))
tStimOnset = tStimOnset.flatten()
tStimOffset = tStimOffset.flatten()

plt.figure(figsize=(15,3))
for i in range(len(tStimOnset)):
    plt.fill_betweenx([0,np.max(spike_counts)], tStimOnset[i], tStimOffset[i], color="grey", alpha=1)
plt.plot(time_series, spike_counts,color='black',linewidth=.5,alpha=.8)
plt.savefig(data_path+'/'+'nSpk_with_time.png')

nSpk_ONset=np.zeros((len(tStimOnset),int(1000*stimuli_duration/bin_size)))
nSpk_OFFset0=np.zeros((len(tStimOnset),int(1000*stimuli_duration/bin_size)))
nSpk_OFFset1=np.zeros((len(tStimOnset),int(1000*stimuli_duration/bin_size)))

for i in range(len(tStimOnset)):
    nSpk_ONset[i] = spike_counts[int(tStimOnset[i]*1000*stimuli_duration/bin_size):int(tStimOffset[i]*1000*stimuli_duration/bin_size)]
    nSpk_OFFset0[i] = spike_counts[int((tStimOnset[i]-stimuli_duration)*1000*stimuli_duration/bin_size):int(tStimOnset[i]*1000*stimuli_duration/bin_size)]
    nSpk_OFFset1[i] = spike_counts[int(tStimOffset[i]*1000*stimuli_duration/bin_size):int((tStimOnset[i]+stimuli_duration)*1000*stimuli_duration/bin_size)]

nSpk_ONset_pos = nSpk_ONset.reshape(nRepeat,num_pos,int(1000/bin_size))
nSpk_OFFset_pos = nSpk_OFFset.reshape(nRepeat,num_pos,int(1000/bin_size))

