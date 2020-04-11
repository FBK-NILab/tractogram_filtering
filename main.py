  
#!/usr/bin/env python

import os
import sys
import argparse
import configparser
import warnings
import glob

import numpy as np
import torch

from loops.test import test
from loops.train import train
from utils.general_utils import get_cfg_value, print_cfg, set_exp_name

if __name__ == '__main__':

    #### ARGUMENT PARSING
    parser = argparse.ArgumentParser()
    parser.add_argument('model_var', 
                        nargs='?', 
                        const=0, 
                        default='',
                        help='The tag for the model config file.')
    parser.add_argument('dataset_var', 
                        nargs='?', 
                        const=0, 
                        default='',
                        help='The tag for the data config file.')
    parser.add_argument('-opt', 
                        nargs='?', 
                        const=0, 
                        default='train',
                        help='type of exec: train | test')
    parser.add_argument('--exp', 
                        nargs='?', 
                        const=0, 
                        default='',
                        help='experiment path')
    parser.add_argument('--weights', 
                        nargs='?', 
                        const=0, 
                        default='',
                        help='file of the saved model')
    parser.add_argument('--lr', 
                        nargs='?', 
                        const=0, 
                        default='',
                        help='learning rate when resuming training')
    parser.add_argument('--root_dir', 
                        nargs='?', 
                        const=0, 
                        default='',
                        help='dataset root dir')
    parser.add_argument('--config', 
                        nargs='?', 
                        const=1, 
                        default='',
                        help='load config.txt in exp dir')
    parser.add_argument('--with_gt', 
                        nargs='?', 
                        const=1, 
                        default='',
                        help='if gt is available')
    parser.add_argument('--save_pred', 
                        nargs='?', 
                        const=1, 
                        default='',
                        help='if present save prediction no otherwise')
    args = parser.parse_args()


    #### CONFIG PARSING

    # Reading configuration file with specific setting for a run.
    # Mandatory variables for this script:

    cfg_parser = configparser.ConfigParser()
    
    if args.config:
        cfg_parser.read(args.config)
    #elif args.exp:
    #    cfg_parser.read(args.exp + '/config.txt')
    else:
        cfg_parser.read([
            'configs/main_config.ini', 'configs/model_config.ini',
            'configs/data_config.ini'
        ])

    cfg = {}
    for name, value in cfg_parser.items('GENERAL'):
        cfg[name] = get_cfg_value(value)
    for name, value in cfg_parser.items(args.model_var):
        cfg[name] = get_cfg_value(value)
    for name, value in cfg_parser.items(args.dataset_var):
        cfg[name] = get_cfg_value(value)

    cfg['opt'] = args.opt

    set_exp_name(cfg, args.model_var, args.dataset_var)

    if cfg['fixed_seed']:
        torch.manual_seed(cfg['fixed_seed'])
        np.random.seed(cfg['fixed_seed'])

    #### LAUNCH RUNS
    if 'train' in cfg['opt']:
        for name, value in cfg_parser.items('TRAIN'):
            cfg[name] = get_cfg_value(value)

        if args.weights:
            cfg['experiment_name'] = 'resume'
            cfg['resume_training'] = True
            cfg['weights_path'] = args.weights
            if args.lr:
                cfg['resuming_lr'] = float(args.lr)
        else:
            cfg['resume_training'] = False
        print_cfg(cfg)
        train(cfg)
        args.exp = cfg['experiment_name']
        args.weights = glob.glob('%s/models/best_*' % args.exp)[0]

        if cfg['val_dataset_dir']:
            cfg['dataset_dir'] = cfg['val_dataset_dir']
        if 'test_dataset_dir' in cfg.keys():
            cfg['dataset_dir'] = cfg['test_dataset_dir']

    if 'test' in cfg['opt']:
        for name, value in cfg_parser.items('TEST'):
            cfg[name] = get_cfg_value(value)
        if not args.exp:
            sys.exit('Missing argument --exp')
        cfg['exp_path'] = args.exp
        if args.weights:
            cfg['weigths_path'] = args.weights
        else:
            cfg['weights_path'] = ''
        if args.sub_list:
            cfg['sub_list_test'] = args.sub_list
        if args.root_dir:
            cfg['val_dataset_dir'] = args.root_dir
        if args.with_gt:
            cfg['with_gt'] = True
        else:
            cfg['with_gt'] = False
        if args.save_pred:
            cfg['save_pred'] = True
        else:
            cfg['save_pred'] = False
        print_cfg(cfg)
        test(cfg)
