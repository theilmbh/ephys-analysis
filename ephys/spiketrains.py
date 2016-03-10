
def get_spiketrain(rec,samps,clu,spikes,window,fs):
    '''
    Returns a numpy array of spike times for a single cluster 
        within a window locked to a sampling time.
    
    Parameters
    ------
    rec : int
        the recording to look in
    samps : int
        the time to lock the spiketrain to in samples
    clu : int
        the cluster identifier to get spikes from
    spikes : pandas dataframe
        the pandas dataframe containing spikes (see core)
    window : tuple or list of floats
        the window around the event in seconds to sample spikes
    fs : float
        sampling rate of the recording
    
    Returns
    ------
    spike_train : numpy array of spike times in seconds
    '''
    bds = [w*fs+samps for w in window]
    window_mask = (
        (spikes['recording']==rec)
        & (spikes['time_samples']>bds[0])
        & (spikes['time_samples']<=bds[1])
        )
    clu_mask = spikes['cluster']==clu
    return (spikes['time_samples'][window_mask & clu_mask].values.astype(np.float_) - samps) / fs