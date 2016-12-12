#! /usr/bin/env python

import os
from add_features import add_features


base_path = 'datasets/data/labeled/'

# (path, default-arg)
defaults = {'false_positives.npz': 0}             

if __name__ == '__main__':
    import argparse
        
    parser = argparse.ArgumentParser(description='Update all standard datasets')
                             
    parser.add_argument('--keep', type=float, 
                        help='What fraction of input to keep\n'
                             '(defaults to 1)',
                        default=1)
               
    args = parser.parse_args()
         
    assert 0 < args.keep <= 1
    if args.keep < 1:
        suffix = '-' + str(args.keep).replace('.', '')
    else:
        suffix = ''

    for name in os.listdir(base_path):
        if not name.endswith(".npz"): continue
        default = defaults.get(name, None)
        in_path = os.path.join(base_path, name)
        basename, ext = os.path.splitext(name)
        out_path = os.path.join(base_path, basename + '.measures' + suffix + ext)
        
        print("Creating %s with default %s" % (out_path, default))

        add_features(in_path, out_path, 
                     default=default, 
                     keep_prob=args.keep)
