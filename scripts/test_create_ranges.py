from __future__ import print_function
from __future__ import division
import numpy as np
import pandas as pd
import datetime
import dateutil
import logging
import create_fishing_nonfishing_ranges as cfr
import merge_ranges_with_tracks as mrwt




 

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
			permuted_results = mrwt.create_fishing_series(m, times[ndx_map], ranges)
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


def test_is_sorted():
	assert is_sorted([0, 1, 1, 2, 3, 4, 5, 5])
	assert not is_sorted([0, 1, 2, 1, 3, 4, 5])


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
	test_is_sorted()
	test_round_trip(in_paths, args.range_path)
