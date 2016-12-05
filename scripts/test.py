import numpy, sys
x = numpy.load(sys.argv[1])['x']
x = x[~numpy.isinf(x['is_fishing']) & ~numpy.isnan(x['is_fishing']) & ~numpy.isnan(x['timestamp']) & ~numpy.isnan(x['speed']) & ~numpy.isnan(x['course'])]

for name in x.dtype.names:
    s = numpy.sum(numpy.isnan(x[name]))
    if s > 0:
        print name, s
