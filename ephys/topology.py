import numpy as np
import pandas as pd

import events
import core
import spiketrains as spt

def get_spikes_in_window(spikes, window):
	'''
	Returns a spike DataFrame containing all spikes within a given window

	Parameters
	------
	spikes : pandas DataFrame
		Contains the spike data (see core)
	window : tuple
		The lower and upper bounds of the time window for extracting spikes 

	Returns
	------
	spikes_in_window : pandas DataFrame
		DataFrame with same layout as input spikes but containing only spikes within window 
	'''

	mask = (spikes['time_samples'] <= window[1]) & (spikes['time_samples']>=window[0])
	return spikes[mask]

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

	# Get all of the spikes within the time window
	spikes = get_spikes_in_window(spikes, window)

	# Compute number of spikes
	nspikes = len(spikes.index)

	# Compute duration
	dt = window[1] - window[0]

	# Compute firing rate
	mean_fr = (1.0*nspikes) / dt

	return mean_fr

def create_subwindows():
	''' Create list of subwindows for cell group identification 
	'''

def calc_population_vectors(spikes, clusters, windows, thresh):
	'''
	Builds population vectors according to Curto and Itskov 2008

	Parameters
	------
	spikes : pandas DataFrame
		DataFrame containing spike data. Must have 'fr_mean' column containing mean firing rate
	clusters : pandas DataFrame
		Dataframe containing cluster information
	windows : tuple
		The set of windows to compute population vectors for 
	thresh : float
		how many times above the mean the firing rate needs to be for it to count

	Returns 
	------
	popvec_list : list
		population vector list.  Each element is a list containing the window and the population vector.
		The population vector is an array containing cluster ID and firing rate. 
	'''

	popvec_list = []
	for win in windows:
		popvec = np.zeros([len(clusters.index), 3])
		for ind, cluster in enumerate(clusters['cluster'].values):
			fr = get_mean_fr(cluster, spikes, win)
			popvec[ind, 1] = fr
			popvec[ind, 0] = cluster
			popvec[ind, 2] = fr > 1.0*tresh*clusters[clusters['cluster']==cluster]['mean_fr']#wooboy
		popvec_list.append([win, popvec])
	return popvec_list

def calc_cell_groups(spikes, segment, clusters, cluster_group=None, subwin_len, threshold=6., n_subwin=5):
	'''
	Creates cell group dataframe according to Curto and Itskov 2008

	Parameters
	------
	spikes : pandas dataframe
		Dataframe containing the spikes to analyze
	segment : tuple or list of floats
		time window for which to create cell groups 
	clusters : pandas DataFrame
		dataframe containing cluster information
	cluster_group : str, optional
		Quality of the clusters to include in the analysis (Good, MUA, etc)
	subwin_len : int 
		length (samples) for each subwin
	threshold : float, optional 
		Standard deviations above baseline firing rate to include activity in cell group
	n_subwin : int 
		Number of subwindows to use to generate population vectors

	Returns
	------
	cell_groups : list
		list where each entry is a list containing a time window and the ID's of the cells in that group
	'''

	# Extract spikes within window
	spikes = get_spikes_in_window(spikes, segment)

	if cluster_group ~= None:
		mask = np.ones(len(spikes.index)) < 0
		for grp in cluster_group:
			mask = np.logical_or(mask, clusters['quality'] == grp)
		clusters = clusters[mask]
		spikes = spikes[spikes['cluster'].isin(clusters['cluster'].values)]

	# Create subwindows
	topology_subwindows = create_subwindows()

	# Get mean and standard deviation of firing rate for each cluster
	clusters['fr_mean'] = clusters.apply(lambda row: get_mean_fr(row['cluster'],spikes,segment),axis=1)

	# Build population vectors
	population_vector_list = calc_population_vectors(spikes, clusters, topology_subwindows)

	# Threshold firing rates
	cell_groups = []
	for population_vector_win in population_vector_list:
		win = population_vector_win[0]
		popvec = population_vector_win[1]
		active_cells = popvec[popvec[:, 2], 0]
		cell_groups.append([win, active_cells])
		
	return cell_groups

def build_perseus_input(cell_groups, savefile):
	''' Formats cell group information as input for perseus persistent homology software

	Parameters
	------
	cell_groups : list 
		cell_group information returned by calc_cell_groups
	savefile : str 
		File in which to put the formatted cellgroup information

	Yields
	------
	savefile : text File
		file suitable for running perseus on
	'''

	with open(savefile, 'w+') as pfile:
		#write num coords per vertex
		fd.write('1\n')
		for win_grp in cell_groups:
			grp = win_grp[1]
			#debug_print('Cell group: ' + str(grp) +'\n')
			grp_dim = len(grp) - 1
			if grp_dim < 0:
				continue
			vert_str = str(grp)
			vert_str = vert_str.replace('(', '')
			vert_str = vert_str.replace(')', '')
			vert_str = vert_str.replace(' ', '')
			vert_str = vert_str.replace(',', ' ')
			out_str = str(grp_dim) + ' ' + vert_str + ' 1\n'
			#debug_print('Writing: %s' % out_str)
			fd.write(out_str)

def run_perseus(pfile):
	''' Runs perseus persistent homology software on the data in pfile

	Parameters
	------
	pfile : str 
		file on which to compute homology

	Returns
	------
	bettis : list
		betti numbers

	'''

	bettis = 0
	return bettis

def calc_bettis(spikes, segment):
	''' Calculate betti numbers for spike data in segment
	'''
