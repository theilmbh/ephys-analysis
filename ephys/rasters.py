from spiketrains import get_spiketrain
import core
import events

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
	ax.xlim(times)
	ax.ylim([1, ntrials])
	for trial, trialdata in enumerate(raster_data):
		ypts = [1+trial, 2+trial]
		for spiketime in trialdata:
			ax.plot([spiketime, spiketime], ypts, 'k', lw=1.5)

	for pltticks in ticks:
		ax.plot([pltticks, pltticks], [1, ntrials], 'r', lw=1.5)

	return ax



def plot_raster_single_cell(block_path, clusterID):
	print('Nothing')