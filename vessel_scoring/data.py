from __future__ import print_function, division
import numpy as np
import numpy.lib.recfunctions
import warnings
from vessel_scoring.utils import *


def _subsample_even(x0, mmsi, n):
    """Return `n` subsamples from `x0`

    - all samples have given `mmsi`

    - samples are evenly divided between fishing and nonfishing
    """
    # Create a mask that is true whenever mmsi is one of the mmsi
    # passed in
    mask = np.zeros([len(x0)], dtype=bool)
    for m in mmsi:
        mask |= (x0['mmsi'] == m)
    x = x0[mask]
    # Pick half the values from fishy rows and half from nonfishy rows.
    f = fishy(x)
    nf = nonfishy(x)
    if n//2 > len(f) or n//2 > len(nf):
        warnings.warn("inufficient items to sample, returning fewer")
    f = np.random.choice(f, min(n//2, len(f)), replace=False)
    nf = np.random.choice(nf, min(n//2, len(nf)), replace=False)
    ss = np.concatenate([f, nf])
    np.random.shuffle(ss)
    return ss

def _subsample_proportional(x0, mmsi, n):
    """Return `n` subsamples from `x0`

    - all samples have given `mmsi`

    - samples are random, so should have ~same be in the same proportions
      as the x0 for the given mmsi.
    """
    # Create a mask that is true whenever mmsi is one of the mmsi
    # passed in
    mask = np.zeros([len(x0)], dtype=bool)
    for m in mmsi:
        mask |= (x0['mmsi'] == m)
    x = x0[mask]
    # Pick values randomly
    # Pick values randomly
    if n > len(x):
        warnings.warn("inufficient items to sample, returning {}".format(len(x)))
        n = len(x)
    ss = np.random.choice(x, n, replace=False)
    np.random.shuffle(ss)
    return ss


def _bin_counts(bins, counts):
    bin_counts = []
    for bin in bins:
        bin_counts.append(sum(counts[i] for i in bin))
    return bin_counts
    

def divide_counts(counts, proportions):
    """Divide counts up in the proportions specified in props
    
    Parameters
    ----------
    counts : sequence of integers
        Counts
    
    proportions : sequence of numbers
        Proportions to divide sequences in. These are normalized
        withing the function so `proportions=[2, 1, 1]` is equivalent
        to `proportions=[0.5, 0.25, 0.25]`
    
    Returns
    -------
    list of lists of indices to counts.
    
    >>> some_counts = np.array([1, 2, 3])
    >>> bins = divide_counts(some_counts, [3, 2, 1])
    >>> [list(some_counts[x]) for x in bins]
    [[3], [2], [1]]
    
    >>> many_counts = np.array([1, 2, 3] * 4)
    >>> bins = divide_counts(many_counts, [3, 3, 1])
    >>> [list(many_counts[x]) for x in bins]
    [[3, 3, 2, 1, 1], [3, 2, 2, 2, 1], [3, 1]]
    
    """
    if len(counts) < len(proportions):
        raise ValueError("less sequences than proportions")
    
    # Normalize props
    props = np.array(proportions, dtype=float)
    props /= props.sum()
    
    # Sort our sequences from largest to smallest; we'll insert 
    # largest first.
    indices = list(np.argsort(counts))
      
    # Initialize our output; placing our largest sequence in
    # the largest bin
    bins = [[] for _ in props]
    bins[np.argmax(props)].append(indices.pop())
    
    # Add the largest sequence to the bin with the largest deficit
    # until we are out of sequences
    while indices:
        bincnts = _bin_counts(bins, counts)
        expected = sum(bincnts) * props
        deficits = expected - bincnts
        bins[np.argmax(deficits)].append(indices.pop())
        
    return bins
        

def load_dataset_by_vessel(path, size = 20000, even_split=None, seed=4321):
    """Load a dataset from `path` and return train, valid and test sets

    path - path to the dataset
    size - number of samples to return in total, divided between the
           three sets as (size//2, size//4, size//4)
    even_split - if True, use 50/50 fishing/nonfishing split for training
                  data, otherwise sample the data randomly.

    The data at path is first randomly divided by divided into
    training (1/2), validation (1/4) and test(1/4) data sets.
    These sets are chosen so that MMSI values are not shared
    across the datasets.

    The validation and test data are sampled randomly to get the
    requisite number of points. The training set is sampled randomly
    if `even_split` is False, otherwise it is chose so that half the
    points are fishing.

    """
    # Set the seed so that we can reproduce results consistently
    np.random.seed(seed)

    # Load the dataset and strip out any points that aren't classified
    # (has classification == Inf)
    x = np.load(path)['x']
    x = x[~np.isinf(x['classification']) & 
          ~np.isnan(x['classification']) & 
          ~np.isnan(x['timestamp']) & 
          ~np.isnan(x['speed']) & 
          ~np.isnan(x['course'])]

    if size > len(x):
        warnings.warn("insufficient items to sample, returning all")
        size = len(x)


    mmsi = np.array(list(set(x['mmsi'])))
    
    if even_split is None:
        even_split = (x['classification'].sum() > 1 and 
                      x['classification'].sum() < len(x))
        
    if even_split:
        # Exclude mmsi that don't have at least one fishing and non-fishing 
        # point. This helped with some pathological MMSI that were
        # creeping in.
        base_mmsi = mmsi
        mmsi = []
        for m in base_mmsi:
            subset = x[x['mmsi'] == m]
            fishing_count = subset['classification'].sum()
            if fishing_count == 0 or fishing_count == len(subset):
                continue
            mmsi.append(m)
        mmsi = np.array(mmsi)
        
    counts = [(x['mmsi'] == m).sum() for m in mmsi]
          
    indices = divide_counts(counts, [2, 1, 1])
    train_subsample = _subsample_even if even_split else _subsample_proportional
    
    xtrain = train_subsample(x, mmsi[indices[0]], size//2)
    xcross = _subsample_proportional(x, mmsi[indices[1]], size//4)
    xtest = _subsample_proportional(x, mmsi[indices[2]], size//4)

    return x, xtrain, xcross, xtest
