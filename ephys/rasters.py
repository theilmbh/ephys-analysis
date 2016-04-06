from spiketrains import get_spiketrain
import core
import events
import numpy as np

import matplotlib.pyplot as plt 


def do_raster(raster_data, times, ticks, ax=None):
	'''
	Generalized raster plotting function

	Parameters
	------
	raster_data : list of lists of floats
		List of lists.  Each sublist corresponds to one row of events
		Each element of a sublist is an event times
	times : list of floats
		The beginning and end times to plot 
	ticks : list of floats
		Will add a vertical tick across the whole plot for each time in this list

	Returns
	------
	raster_plot : 
		Handle to the raster plot 
	'''

	ntrials = len(raster_data)
	if ax is None:
		ax = plt.gca()
	ax.set_xlim(times)
	ax.set_ylim((1, ntrials+1))
	for trial, trialdata in enumerate(raster_data):
		ypts = [1+trial, 2+trial]
		for spiketime in trialdata:
			ax.plot([spiketime, spiketime], ypts, 'k', lw=1.5)

	for pltticks in ticks:
		ax.plot([pltticks, pltticks], [1, ntrials+1], 'r', lw=1.5)

	return ax



def plot_raster_cell_stim(block_path, clusterID, stim, period):

	spikes = core.load_spikes(block_path)
	trials = events.load_trials(block_path)
	fs = core.load_fs(block_path)
	
	stim_trials = trials[trials['stimulus']==stim]
	ntrials = len(stim_trials)
	stim_starts = stim_trials['time_samples'].values
	stim_ends = stim_trials['stimulus_end'].values

	stim_end_seconds = np.unique((stim_ends - stim_starts)/fs)[0]
	window = [period[0], stim_end_seconds[trial]+period[1]]
	raster_data = []
	for trial, start in enumerate(stim_starts):
		
		sptrain = get_spiketrain(rec, start, clusterID, spikes, window)
		raster_data.append(sptrain)

	do_raster(raster_data, window, [0, stim_end_seconds])

