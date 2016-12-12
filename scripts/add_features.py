from __future__ import print_function, division

# Find the root directory of vessel-scoring, if running directly from git
import sys, os.path; sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from vessel_scoring import utils
from vessel_scoring import add_measures
import numpy as np

def dedup_messages(messages):
    last_key = None
    for msg in messages:
        key = (msg.get('mmsi', None), mdg.get("seg_id", None), msg['timestamp'])
        if key == last_key:
            continue
        yield msg
        last_key = key


def add_features(in_path, out_path, default=None, keep_prob=1):
    data = np.load(in_path)['x']
    
    if default is not None:
        print("Replacing Infs and NaNs with", default)
        is_missing = (np.isnan(data['classification']) | 
                      np.isinf(data['classification']))
        data['classification'][is_missing] = default
        
    if keep_prob < 1:
        np.random.seed(4321)
        print("Keeping points with probability", keep_prob)
        keep = np.random.random(size=len(data)) < keep_prob
        data = data[keep]  

    # Sort by mmsi, then by timestamp
    data.sort(order=['mmsi', 'timestamp'])
    messages = utils.numpy_to_messages(data)
    messages = dedup_messages(messages)
    messages = add_measures.AddMeasures(messages)
    result = utils.messages_to_numpy(messages, len(data))

    np.savez_compressed(out_path, x=result)


if __name__ == "__main__":
    import argparse
        
    parser = argparse.ArgumentParser(
        description='Create new dataset with features added')
        
    parser.add_argument('in_path', metavar='input-path', type=str,
                        help='path to input dataset')
                        
    parser.add_argument('out_path', metavar='output-path', type=str,
                        help='path to output dataset')
                        
    parser.add_argument('--default', type=float,
                        help='value to replace Inf/NaN values with\n'
                             '(do nothing if ommited)',
                        default=None)
                             
    parser.add_argument('--keep', type=float, 
                        help='What fraction of input to keep\n'
                             '(defaults to 1)',
                        default=1)

    args = parser.parse_args()

    add_features(args.in_path, args.out_path, 
                 default=args.default, 
                 keep_prob=args.keep)
    

