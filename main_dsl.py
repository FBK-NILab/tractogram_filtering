import os
import sys
import argparse
import ConfigParser
import warnings
import ipdb

from train_dsl import train
from test_dsl import test

if __name__ == '__main__':


    #### ARGUMENT PARSING
    parser = argparse.ArgumentParser()
    parser.add_argument('var', nargs='?', const=0, default='DEFAULT',
                        help='The tag for the configuration file.')
    parser.add_argument('-opt', nargs='?', const=0, default='train',
                        help='type of exec: train | validate | test')
    parser.add_argument('--exp', nargs='?', const=0, default='',
                        help='experiment path')
    parser.add_argument('--weights', nargs='?', const=0, default='',
                        help='file of the saved model')
    parser.add_argument('--sub_list', nargs='?', const=0, default='',
                        help='sub list containing the test subjects')
    parser.add_argument('--root_dir', nargs='?', const=0, default='',
                        help='dataset root dir')
    parser.add_argument('--config', nargs='?', const=1, default='',
                        help='load config.txt in exp dir')
    parser.add_argument('--with_gt', nargs='?', const=1, default='',
                        help='if gt is available')
    parser.add_argument('--save_pred', nargs='?', const=1, default='',
                        help='if present save prediction no otherwise')
    args = parser.parse_args()


    #### CONFIG PARSING

    # Reading configuration file with specific setting for a run.
    # Mandatory variables for this script:

    cfg_parser = ConfigParser.ConfigParser()
    if not args.config and args.exp:
        cfg_parser.read(args.exp + '/config.txt')
    elif args.config:
        cfg_parser.read(args.config)
    else:
        cfg_parser.read('main_dsl_config.py')
    cfg = {}
    cfg[args.var] = {}
    for name, value in cfg_parser.items('DEFAULT'):
        if value == 'y':
            value = True
        elif value == 'n':
            value = False
        cfg[args.var][name] = value
    for name, value in cfg_parser.items(args.var):
        if value == 'y':
            value = True
        elif value == 'n':
            value = False
        cfg[args.var][name] = value
    cfg['opt'] = args.opt

    for c in cfg[args.var].keys():
        print('%s : %s' % (c, cfg[args.var][c]))
    # TODO: set seed

    #### LAUNCH RUNS
    if cfg['opt'] == 'train':
        train(cfg[args.var])
    if cfg['opt'] == 'test':
        if not args.exp:
            sys.exit('Missing argument --exp')
        cfg[args.var]['exp_path'] = args.exp
        if args.weights:
            cfg[args.var]['weights_path'] = args.weights
        else:
            cfg[args.var]['weights_path'] = ''
        if args.sub_list:
            cfg[args.var]['sub_list_test'] = args.sub_list
        if args.root_dir:
            cfg[args.var]['val_dataset_dir'] = args.root_dir
        if args.with_gt:
            cfg[args.var]['with_gt'] = True
        else:
            cfg[args.var]['with_gt'] = False
        if args.save_pred:
            cfg[args.var]['save_pred'] = True
        else:
            cfg[args.var]['save_pred'] = False
        test(cfg[args.var])
