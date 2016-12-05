#! /usr/bin/env python

import numpy
import csv
import datetime
import sys
import os.path

unscaled_mmsi = [224068000, 224098250, 224108130]

def get_time(r):
    return float(datetime.datetime.strptime(r['timestamp'], "%Y-%m-%d %H:%M:%S UTC").strftime("%s"))


colmap = {
  "mmsi": lambda r: float(r['mmsi']),
  "lon": lambda r: float(r['lon']),
  "lat": lambda r: float(r['lat']),
  "course": lambda r: float(r['course']),
  "speed": lambda r: float(r['speed']),
  "timestamp": get_time,
  "is_fishing": lambda r: 0
}

def get_file_size(path):
    length = 0
    with open(path) as f:
         f = csv.DictReader(f)
         for row in f:
              length += 1
    return length

def convert(path, outpath):

    if os.path.isdir(path):
        paths = [os.path.join(path, p) for p in os.listdir(path) if p.endswith('.csv')]
    else:
        paths = [path]

    length = 0
    for path in paths:
        length += get_file_size(path)

    x = numpy.zeros(length, dtype=[(name, "f8") for name in colmap.keys()])

    pos = 0
    for path in paths:
        with open(path) as f:
            f = csv.DictReader(f)
            for row in f:
                for col, mapper in colmap.iteritems():
                    try:
                        val = mapper(row)
                    except:
                        val = numpy.Infinity
                    x[col][pos] = val
                pos += 1

    numpy.savez_compressed(outpath, x=x)

convert(sys.argv[1], sys.argv[2])
