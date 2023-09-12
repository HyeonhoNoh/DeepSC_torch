import argparse
import os
import datetime
import torch
import numpy as np

def def_param():
    param = dict()

    # Info. params
    param['gpu_id'] = 0
    param['test'] = False
    param['verbose'] = False
    param['model_name'] = 'DRL_semantic'

    # Training params
    param['batch_size'] = 256
    param['optimizer'] = 'SGD'
    param['n_epochs'] = 200
    param['lr_master'] = 1e-2
    param['save_freq'] = int(param['n_epochs'] / 2)
    param['loss'] = 'MSE'  # [MSE]
    param['continue_train'] = False  # if train_or_test == 'train' else True
    param['checkpoint'] = None

    #  Optimizer
    param['b1'] = 0.9
    param['b2'] = 0.98
    param['eps'] = 1e-9

    ## Transformer params
    param['num_encoder_layers'] = 3
    param['num_decoder_layers'] = 3
    param['emb_size'] = 128
    param['nhead'] = 8
    param['TGT_LANGUAGE'] = 'en'
    param['dim_feedforward'] = 512
    param['dropout'] = 0.5
    param['coding_rate'] = 16

    ##  Communication params
    param['channel'] = 'AWGN'
    param['noise_var'] = 1e-1
    param['test_noise_var'] = param['noise_var']
    param['SNR'] = -10 * np.log10(param['noise_var'])
    param['test_SNR'] = -10 * np.log10(param['test_noise_var'])

    return param


def get_path(param):
    path = dict()

    checkpoint_path = './checkpoints/train_' + str(param['SNR'])
    path['checkpoint_path'] = checkpoint_path

    data_path = './dataset/'

    path['data_path'] = data_path

    return path


class Config:
    def __init__(self):
        self.param = def_param()

        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument('--param', type=list, default=self.param)
        self.parser.add_argument('--gpu_id', type=int, default=self.param['gpu_id'])
        self.parser.add_argument('--test', type=bool, default=self.param['test'])
        self.parser.add_argument('--verbose', type=bool, default=self.param['verbose'])
        self.parser.add_argument('--continue_train', type=str, default=self.param['continue_train'])

        self.parser.add_argument('--SNR', type=float, default=self.param['SNR'])
        self.parser.add_argument('--test_SNR', type=float)

        self.opt, _ = self.parser.parse_known_args()
        args = self.parser.parse_args()

        self.param = args.param
        self.param['gpu_id'] = int(args.gpu_id)
        self.param['test'] = args.test
        self.param['continue_train'] = args.continue_train
        self.param['verbose'] = args.verbose

        self.param['SNR'] = args.SNR
        self.param['noise_var'] = 10 ** (-self.param['SNR'] / 10)

        if args.test_SNR:
            self.param['test_SNR'] = args.test_SNR
            self.param['test_noise_var'] = 10 ** (-self.param['test_SNR'] / 10)
        else:
            self.param['test_SNR'] = self.param['SNR']
            self.param['test_noise_var'] = 10 ** (-self.param['test_SNR'] / 10)

        self.device = torch.device("cuda:{}".format(self.param['gpu_id']) if torch.cuda.is_available() else "cpu")

        self.paths = get_path(param=self.param)

    def print_options(self):
        """Print and save options
                It will print both current options and default values(if different).
                It will save options into a text file / [checkpoints_dir] / opt.txt
                """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(self.opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        if not self.param['test']:
            expr_dir = os.path.join(self.paths['checkpoint_path'])
            mkdirs(expr_dir)
            file_name = os.path.join(expr_dir, 'train_opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')
        else:
            expr_dir = os.path.join(self.paths['checkpoint_path'])
            mkdirs(expr_dir)
            file_name = os.path.join(expr_dir, 'test_opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')

def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
