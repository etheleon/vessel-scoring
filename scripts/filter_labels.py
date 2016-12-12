#! /usr/bin/env python

import numpy as np
import sys

arr = np.load(sys.argv[1])['x']
arr = arr[arr['is_fishing'] != -1]
np.savez_compressed(sys.argv[2], x=arr)
