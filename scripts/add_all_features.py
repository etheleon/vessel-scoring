import os
from add_features import add_features


base_path = 'datasets/data/labeled/'

# (path, default-arg)
templates = [('false_positives{}.npz', 0),
             ('kristina_trawl{}.npz', None),
             ('kristina_ps{}.npz', None),
             ('kristina_longliner{}.npz', None)]
             

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

    for name, default in templates:
        in_path = os.path.join(base_path, name.format(''))
        out_path = os.path.join(base_path, name.format('.measures' + suffix))
        
        print("Creating", out_path)

        add_features(in_path, out_path, 
                     default=default, 
                     keep_prob=args.keep)
