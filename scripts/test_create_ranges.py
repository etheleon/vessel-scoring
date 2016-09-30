from __future__ import print_function
from __future__ import division
import numpy as np
import pandas as pd
import datetime
import dateutil
import logging
import create_fishing_nonfishing_ranges as cfr


def create_fishing_series(mmsi, times, ranges):
	"""

	Parameters
	==========
	mmsi : str

	times : sequence of datetime
		Sequence must be sorted

	ranges: sequence of (mmsi, start_time, end_time, is_fishing)
		mmsi : str
		start_time : datetime
		stop_time : datetime
		is_fishing : boolean

	Returns
	=======
	sequence of {0, 1, -1}
		whether the vessel is fishing at given point or -1 for
		don't know

	"""
	# TODO: check that times are sorted
	# Only look at ranges associated with the current mmsi
	ranges = ranges[ranges['mmsi'] == mmsi]
	# Initialize is_fishing to -1 (don't know)
	is_fishing = np.empty([len(times)], dtype=int) 
	is_fishing.fill(-1)
	#
	for _, (_, startstr, endstr, state) in ranges.iterrows():
		try:
			start = dateutil.parser.parse(startstr)
			end = dateutil.parser.parse(endstr)
		except:
			print(startstr, endstr, state)
			raise
		i0 = np.searchsorted(times, start, side="left")
		i1 = np.searchsorted(times, end, side="right")
		is_fishing[i0: i1] = state
	#
	return is_fishing
 

def test_round_trip(source_paths, range_path):
	ranges = pd.read_csv(range_path)
	for pth in in_paths:
		logging.info("testing file: {}".format(pth))
		all_examples = pd.read_csv(pth)
		mmsi = sorted(set(all_examples['MMSI']))
		for m in mmsi:
			logging.info("testing mmsi: {}".format(m))
			examples = all_examples[all_examples['MMSI'] == m]
			times = np.array([cfr.get_kristina_timestamp(x) for (_, x) in examples.iterrows()])	
			fishing = np.array([cfr.get_kristina_is_fishing(x) for (_, x) in examples.iterrows()], dtype=bool)
			ndx_map = np.argsort(times)
			permuted_results = create_fishing_series(m, times[ndx_map], ranges)
			#
			results = np.zeros_like(permuted_results)
			results[ndx_map] = permuted_results
			#
			unknown_mask = (results == -1)
			correct = np.alltrue((results == fishing) | unknown_mask)
			if not correct:
				logging.error("{}: {} failed".format(pth, m))
			if unknown_mask.sum():
				logging.warning("{}% of samples unknown for {}: {}".format(100 * unknown_mask.sum() / 
											len(unknown_mask), pth, m))



if __name__ == "__main__":
	import argparse
	import glob
	import os
	logging.getLogger().setLevel("WARNING")
	parser = argparse.ArgumentParser(description="extract fishing/nonfishing ranges from Kristina's data")
	parser.add_argument('--source-dir', help='directory where converted sources were drawn from')
	parser.add_argument('--range-path', help='path to range file')
	args = parser.parse_args()
	in_paths = glob.glob(os.path.join(args.source_dir, "*.csv"))
	test_round_trip(in_paths, args.range_path)
