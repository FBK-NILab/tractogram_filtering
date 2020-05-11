#!/usr/bin/env python

import glob
import os
import sys
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-min',
                        nargs='?',
                        default=0,
                        help='minimum num of epoch to save a run')

    args = parser.parse_args()

    min_epoch = int(args.min)

    all_runs = glob.glob('runs/*/*')
    print('found %d runs' % (len(all_runs)))

    bad_runs = []
    for d in all_runs:
        if 'models' not in os.listdir(d):
            bad_runs.append(d)
        else:
            model_dir = os.path.join(d, 'models')
            models = glob.glob('%s/*model*' % model_dir)

            model_last_ep = 0
            for m in models:
                ep = int(m.split('ep-')[1].split('_')[0])
                if ep > model_last_ep:
                    model_last_ep = ep

            if model_last_ep < min_epoch:
                bad_runs.append(d)
                print(d)

    if len(bad_runs) == 0:
        sys.exit('nothing to be deleted')

    print('found %d bad runs' % (len(bad_runs)))
    for d in bad_runs:
        os.system('rm -r {}'.format(d))

    print('bad runs deleted')