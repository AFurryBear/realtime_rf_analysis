
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from functools import reduce

def filter_signal(x, fs, low, high, order=3):

    """Filter raw signal x. 
    
    Parameters
    ----------
    
    x: pd.DataFrame, (n_samples, 4)
        Each column in x is one recording channel.
    
    fs: int
        Sampling frequency.
    
    low, high: int, int
        Passband in Hz for the butterworth filter.   
        
    order: int
        The order of the Butterworth filter. Default is 3, but you should try 
        changing this and see how it affects the results.
        
    
    Return
    ------
    
    y: pd.DataFrame, (n_samples, 4)
        The filtered x. The filter delay is compensated in the output y.
        
    
    Notes
    ----
    
    1. Try exploring different filters and filter settings. More info:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
    
    2. The output signal should be phase-shift compensated. More info:
    https://dsp.stackexchange.com/a/19086
    
    """

    # --------------------------------------------------------------------
    # implement a suitable filter and apply it to the input data 
    # --------------------------------------------------------------------
    y = np.zeros(x.shape)
    b,a = signal.butter(order, [low ,high], btype='band', fs=fs, analog=False, output='ba')
    for ch in range(len(x)):
        y[ch] = signal.filtfilt(b, a, x[ch])
 
    return y


def app_refractory(x, last_x, lockout):
    if x-last_x > lockout:
        last_x = x
    return last_x 

def detect_spikes(x, fs, N=5, lockout=10):

    """Detect spikes, in this case, the relative local minima of the signal x.
    
    Parameters
    ----------
    
    x: pd.DataFrame
        The filtered signal from Task 1.
        
    fs: int
        the sampling rate (in Hz).
        
    N: int
        An arbitrary number with which you multiply with the standard deviation
        to set a threshold that controls your false positive rate. Default is 5
        but you should try changing it and see how it affects the results.
        
    lockout: int
        a window of 'refactory period', within which there's only one spike. 
        Default is 10 but you should also try changing it. 
    
    
    Returns
    -------
    
    s: np.array, (n_spikes,)
        Spike location / index in the singal x. 
    
    t: np.array, (n_spikes,)
        Spike time in ms. By convention the time of the zeroth sample is 0 ms.
    
    thrd: float
        Threshold = -Nσ. 
        
    ----
    
    Use scipy functions to detect local minima. 
    Noted that there are four channels in signal x. 
        
    """

    # ---------------------------------
    # compute the robust s.d. 
    # ---------------------------------
    
    robustsd = np.median(np.abs(x-np.average(x))/0.6745)
    
    # --------------------------------
    # find all local minima 
    # --------------------------------
    
    lm_all = []
    peak_counts = []

    for ch in range(len(x)):
        
        lm_ch = signal.find_peaks(-x[ch,:], distance = lockout)[0]
        lm_all.append(lm_ch)
        peak_counts.append(len(lm_ch))

    lm_all_clean = reduce(np.union1d, lm_all)

    
    # ---------------------------------
    # calculate the threshold 
    # ---------------------------------
    thrd = -N*robustsd
    ind_pass_thrd = np.where(np.min(x[:,lm_all_clean],axis=0)<thrd)[0]
    s_all = lm_all_clean[ind_pass_thrd]    
  
    if len(s_all)>0:
        ## delete duplicate detection from different channel
        last_lm = s_all[0]
        s_dup = np.ones(s_all.shape)*last_lm.astype(int)
        
        for i, stp in enumerate(s_all[1:]):
            s_dup[i+1] = app_refractory(stp, s_dup[i], lockout)
        s = np.unique(s_dup).astype(int)
        
        t = s/fs*1000
    else:
        s,t = [],[]
    

    return (s, t), thrd, s_all, peak_counts


def get_spkTrain(spike_times, bin_size=100):

    bin_size = 100  # milliseconds

    # Calculate the number of bins required
    time_range = spike_times.max() - spike_times.min()
    num_bins = np.ceil(time_range / bin_size).astype(int)

    # Create bins
    bins = np.arange(spike_times.min(), spike_times.max() + bin_size, bin_size)

    # Use numpy.histogram to count spikes in each bin
    spike_counts, _ = np.histogram(spike_times, bins=bins)
    return spike_counts, bin_size


def prepare_rawData_condition(x, onset_time, offset_time, fs):
    
    time_series = np.linspace(onset_time, offset_time, 
                              int((offset_time-onset_time).mean()*fs), endpoint=True).T # Nsession*Ntimepoint
    time_index = (time_series*fs).astype(int) 
    return x[:,time_index] # Nchannel*Nsession*Ntimepoint


def find_stimModulated_channel(data_stim, data_base, num_ch=2):
    coi = np.argsort((data_stim.std(axis=2) - data_base.std(axis=2)).mean(axis=1))
    return coi[-num_ch:]
    