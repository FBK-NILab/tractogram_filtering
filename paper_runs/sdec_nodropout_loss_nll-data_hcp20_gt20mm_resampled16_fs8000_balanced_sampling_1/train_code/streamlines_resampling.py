#!/usr/bin/env python
import sys
import os
import argparse
import ConfigParser
import numpy as np
import nibabel as nib
import glob
from time import time
import pickle

import ipdb

from sampling_functions import bspline


def load_trk(sub, trk_dir, trk_name, src_dir, with_gt, gt_dir='', gt_name=''):
    """Load trk files with nibabel.streamlines api.
    """
    trk_src = os.path.join(src_dir, 'derivatives', trk_dir, 'sub-%s' % sub)
    trk_file = os.path.join(trk_src, '*' + trk_name + '.trk')
    trk_files = glob.glob(trk_file)
    trks = []
    trk_names = []
    gt_files = []
    for t in trk_files:
        trks.append(nib.streamlines.load(t))
        trk_names.append(t.split('/')[-1])

        if with_gt:
            gt_file = '%s_gt.pkl' % t.rsplit('.', 1)[0]
            if os.path.exists(gt_file):
                gt_files.append(gt_file)
            elif gt_dir != '':
                gt_src = os.path.join(src_dir, 'derivatives', gt_dir,
                                      'sub-%s' % sub)
                for name in gt_name:
                    gt_file = os.path.join(gt_src, '*' + name + '*')
                    gt_files.append(glob.glob(gt_file))
    return trks, trk_names, gt_files


def resample(streamlines, type='bspline', n_pts=20):
    resampled = nib.streamlines.ArraySequence()
    for sl in streamlines:
        if type == 'bspline':
            resampled.append(bspline(sl, n=n_pts))
    return resampled


def save_trk(tractogram, hdr, sub, out_dir, out_name, src_dir, **kwargs):
    """Load trk files with nibabel.streamlines api.
    """
    out_src = os.path.join(src_dir, 'derivatives', out_dir, 'sub-%s' % sub)
    if not os.path.exists(out_src):
        os.makedirs(out_src)
    out_file = os.path.join(out_src, out_name)

    nib.streamlines.save(tractogram, out_file, header=hdr)

    if 'gt_file' in kwargs.keys():
        if type(kwargs['gt_file']) == str:
            kwargs['gt_file'] = [kwargs['gt_file']]
        for gt_file in kwargs['gt_file']:
            out_gt_file = '%s_gt.pkl' % out_file.rsplit('.', 1)[0]
            if gt_file.split('.')[-1] == 'pkl':
                os.system("cp -L %s %s" % (gt_file, out_gt_file))
            elif gt_file.split('.')[-1] == 'txt':
                gt = np.loadtxt(gt_file, dtype=np.uint8)
                with open(out_gt_file, 'wb') as f:
                    pickle.dump(gt, f, protocol=pickle.HIGHEST_PROTOCOL)
            elif gt_file.split('.')[-1] == 'npy':
                gt = np.load(gt_file)
                with open(out_gt_file, 'wb') as f:
                    pickle.dump(gt, f, protocol=pickle.HIGHEST_PROTOCOL)

    if 'gt' in kwargs.keys():
        out_gt_file = '%s_gt.pkl' % out_file.rsplit('.', 1)[0]
        with open(out_gt_file, 'wb') as f:
            pickle.dump(gt, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    #### SETTING REPO LOCATION
    script_src = os.path.basename(sys.argv[0]).strip('.py')
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    src_dir = os.path.abspath(os.path.join(script_dir, '../..'))

    #### ARGUMENT PARSING
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'sub_file',
        nargs='?',
        default='',
        help='Text file containing the list of the subjects')
    parser.add_argument(
        '-var',
        nargs='?',
        const=0,
        default='DEFAULT',
        help='The tag for the configuration file.')
    parser.add_argument(
        '-tract_src', nargs='?', const=0, default='', help='tract dir')
    parser.add_argument(
        '-tract_name', nargs='?', const=0, default='', help='bundle name')
    parser.add_argument(
        '-gt_dir', nargs='?', const=0, default='', help='tract dir')
    parser.add_argument(
        '-gt_name', nargs='?', const=0, default='', help='gt bundle name')
    parser.add_argument(
        '-n_pts',
        nargs='?',
        const=0,
        default='',
        help='if specified these prototypes are used for all subjects')
    parser.add_argument(
        '-with_gt',
        nargs='?',
        const=1,
        default='',
        help='use this option if you want to save the streamlines labels')
    args = parser.parse_args()

    if args.var:
        cfg_src = os.path.join(script_dir, script_src + '_' + args.var + '.py')
        cfg = ConfigParser.ConfigParser()
        cfg.read(cfg_src)

        if not args.tract_src:
            args.tract_src = cfg.get(args.var, 'tract_src')
        if not args.tract_name:
            args.tract_name = cfg.get(args.var, 'tract_name')
        if not args.gt_dir:
            if cfg.has_option(args.var, 'gt_dir'):
                args.gt_dir = cfg.get(args.var, 'gt_dir')
            else:
                args.gt_dir = ''
        if not args.gt_name:
            if cfg.has_option(args.var, 'gt_name'):
                args.gt_name = cfg.get(args.var, 'gt_name').split()
            else:
                args.gt_name = ''
        if not args.n_pts:
            args.n_pts = cfg.get(args.var, 'n_pts')

    if args.sub_file:
        sub_file = os.path.join(script_dir, args.sub_file)
        if not os.path.exists(sub_file):
            sys.exit('subjects list file not found')

        tract_src = args.tract_src
        tract_name = args.tract_name
        n_pts = args.n_pts

        with open(sub_file) as f:
            subjects = f.readlines()
            # delete spaces or newline chars
        subjects = [s.strip() for s in subjects]
        print(subjects)

    for sub in subjects:
        print("Processing subject %s" % sub)
        tract_list, tract_names, gt_files = \
                    load_trk(sub, tract_src, tract_name, src_dir, args.with_gt,
                            gt_dir=args.gt_dir, gt_name=args.gt_name)

        print('Found %d tracts' % len(tract_names))

        for i, tract in enumerate(tract_list):

            t0 = time()
            sl_resampled = resample(
                tract.streamlines, type='bspline', n_pts=int(n_pts))
            print("%s sec." % (time() - t0))

            tract_resampled = nib.streamlines.Tractogram(sl_resampled, affine_to_rasmm=np.eye(4))
            out_dir = 'streamlines_resampling_%s' % args.var
            out_name = 'sub-%s_var-%s_%s.trk' % (
                sub, args.var, tract_names[i].split('_')[-1].rsplit('.', 1)[0])
            if args.with_gt:
                save_trk(
                    tract_resampled, tract.header, sub, out_dir, out_name, src_dir, gt_file=gt_files[i])
            else:
                save_trk(tract_resampled, tract.header, sub, out_dir, out_name, src_dir)
