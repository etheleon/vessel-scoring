from __future__ import print_function
from __future__ import division
import numpy as np
import pandas as pd
import datetime
import pytz
import logging

def get_kristina_timestamp(x):
    if 'DATETIME' in x:
        return datetime.datetime.strptime(x['DATETIME'], "%Y-%m-%d %H:%M:%S").replace(tzinfo=pytz.utc)
    elif 'TIME' in x:
        return datetime.datetime.strptime(x['TIME'], "%Y%m%d_%H%M%S").replace(tzinfo=pytz.utc)
    else:
        assert False, "NO TIME: {}".format(x)

def get_kristina_is_fishing(x):
	if 'COARSE_FIS' in x:
		return x['COARSE_FIS']
	else:
		return x['COARSE-FIS']



kristina = {'mmsi' : lambda x: x['MMSI'],
            'timestamp' : get_kristina_timestamp,
            'is_fishing' : get_kristina_is_fishing}


 
def points_from_path(pth, dialect=kristina):
	df = pd.read_csv(pth)
	results = []
	get_mmsi = dialect['mmsi']
	get_time = dialect['timestamp']
	get_fish = dialect['is_fishing']
	for index, row in df.iterrows():
		results.append([get_mmsi(row), get_time(row), get_fish(row)])
	return results


def dedup_and_sort_points(points):
    points.sort()
    dedupped = []
    last_key = (None, None)
    last_fishing = None
    for mmsi, timestamp, is_fishing in points:
        key = (mmsi, timestamp)
        is_fishing = None if is_fishing in (-1, 2) else is_fishing
        if key == last_key:
            if is_fishing != last_fishing:
                dedupped[-1] = (mmsi, timestamp, None)
        else:
            dedupped.append((mmsi, timestamp, is_fishing))
            last_key = key
            last_fishing = is_fishing

    return dedupped


def ranges_from_points(points):
    points = dedup_and_sort_points(points)
    current_state = None
    current_mmsi = None
    ranges = []
    for mmsi, time, state in points:
        if mmsi != current_mmsi or state != current_state:
            if current_state is not None:
                ranges.append((current_mmsi, range_start.isoformat(), last_time.isoformat(), current_state))
            current_state = state
            range_start = time
            current_mmsi = mmsi
        last_time = time
    if current_state is not None:
        ranges.append((current_mmsi, range_start.isoformat(), last_time.isoformat(), current_state))
    return ranges


def ranges_from_path(pth, dialect=kristina):
	return ranges_from_points(points_from_path(pth, dialect=dialect))


def ranges_from_multiple_paths(paths):
	results = []
	for pth in paths:
		logging.info("converting {}".format(pth))
		try:
			results.extend(ranges_from_path(pth))
		except StandardError as err:
			logging.warning("conversion failed for {}".format(pth))
			logging.warning(repr(err))
	return results



if __name__ == "__main__":
	import argparse
	import glob
	import os
	parser = argparse.ArgumentParser(description="extract fishing/nonfishing ranges from Kristina's data")
	parser.add_argument('--source-dir', help='directory holding sources to convert')
	parser.add_argument('--dest-path', help='path to write results to')
	args = parser.parse_args()
	in_paths = glob.glob(os.path.join(args.source_dir, "*.csv"))
	results = ranges_from_multiple_paths(in_paths)
	with open(args.dest_path, "w") as f:
		f.write("mmsi,start_time,end_time,is_fishing\n")
		for row in results:
			f.write("{}\n".format(','.join(str(x) for x in row)))
