import sys
import os
from ephys import topology as top


def get_args():

	parser = argparse.ArgumentParser(description='Calculate full-segment' 
												 'topology of an ' 
												 'extracellular dataset')
	parser.add_argument('block_path', type=str, help='Path to folder'
													 'containing data files')


def main():

	args = get_args()

	block_path = os.path.abspath(args.block_path)
	cluster_group = ['Good']
	segment_info = {'period': 'prestim', 'segstart': -2.0, 'segend': 0.0}
	windt = 50.
	
	topology.calc_bettis_on_dataset(block_path, 
									cluster_group=cluster_group, 
									windt_ms=windt., 
									segment_info=segment_info)


if __name__ == '__main__':
	main()