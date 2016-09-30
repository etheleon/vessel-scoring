from __future__ import print_function
from __future__ import division
import numpy as np
import pandas as pd
import datetime
import dateutil
import logging
import create_fishing_nonfishing_ranges as cfr


def test_round_trip(source_paths, range_path):
	ranges = pd.read_csv(range_path)
	for pth in in_paths:
		logging.info("testing file: {}".format(pth))
		examples = pd.read_csv(pth)
		mmsi = sorted(set(examples['MMSI']))
		for m in mmsi:
			logging.info("processing mmsi: {}".format(pth))
			sub_ranges = ranges[ranges['mmsi'] == m]
			sub_examples = examples[examples['MMSI'] == m]
			starts = [dateutil.parser.parse(x) for x in sub_ranges['start_time']]
			stops = [dateutil.parser.parse(x) for x in sub_ranges['end_time']]
			is_fishing = (sub_ranges['is_fishing'] == 1)
			logging.info("{} ranges present".format(len(starts)))
			example_times = np.array([cfr.get_kristina_timestamp(x) for (_, x) in sub_examples.iterrows()])
			example_fishing = np.array([cfr.get_kristina_is_fishing(x) for (_, x) in sub_examples.iterrows()], dtype=bool)
			count = 0
			for t0, t1, state in zip(starts, stops, is_fishing):
				mask = (example_times >= t0) & (example_times <= t1)
				if not np.alltrue(example_fishing[mask] == state):
					logging.error("item did not round trip: {} {} {}".format(mmsi, t0, t1, state))
					logging.error(pth)
				count += mask.sum()
			if count != len(sub_examples):
				logging.warning("{}% of samples covered for {}: {}".format(100 * count / len(sub_examples), pth, m))


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
