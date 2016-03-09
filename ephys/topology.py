import numpy as np
import pandas as pd

import events
import core
import spiketrains as spt

#todo: decorate
def get_mean_fr(cluster, spikes, window):
	'''
	Computes the mean firing rate of a unit within the given spikes DataFrame

	Parameters
	------
	cluster : int
		cluster id of the cluster to compute
	spikes : pandas dataframe 
		Contains the spike data.  Firing rate computed from this data over the window
	window : tuple
		time (samples) over which the spikes in 'spikes' occur.

	Returns
	------
	mean_fr : float 
		Mean firing rate over the interval
	'''

	# Get all spikes from the cluster
	spikes = spikes[spikes['cluster']==cluster]

	# Compute number of spikes
	nspikes = len(spikes.index)

	# Compute duration
	dt = window[1] - window[0]

	# Compute firing rate
	mean_fr = (1.0*nspikes) / dt

	return mean_fr

def get_stddev_fr

def make_cell_groups(spikes, segment, subwin_len, threshold=6., n_subwin=5):
	'''
	Creates cell group dataframe according to Curto and Itskov 2008

	Parameters
	------
	spikes : pandas dataframe
		Dataframe containing the spikes to analyze
	segment : tuple or list of floats
		time window for which to create cell groups 
	subwin_len : int 
		length (samples) for each subwin
	threshold : float, optional 
		Standard deviations above baseline firing rate to include activity in cell group
	n_subwin : int 
		Number of subwindows to use to generate population vectors

	Returns
	------
	cellgroups : pandas dataframe 
		Dataframe containing cell group information
	'''

	# Extract spikes within window
	spikes = spikes[np.logical_and(spikes['time_samples'] >= segment[0], spikes['time_samples'] <= segment[1])]

	# Get cluster IDs of spikes within window
	clusters = pd.DataFrame(dict(cluster=np.unique(spikes['cluster'].values)))

	# Create subwindows

	# Get mean and standard deviation of firing rate for each cluster
	clusters['fr_mean'] = clusters.apply(lambda row: get_mean_fr(row['cluster'], spikes),axis=1)
	clusters['fr_stddev'] = clusters.apply(lambda row: get_stddev_fr(row['cluster'], spikes, subwindows),axis=1)








