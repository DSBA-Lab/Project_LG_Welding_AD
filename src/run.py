import argparse
import os
import sys
import random
import numpy as np
from omegaconf import OmegaConf
import torch

from utils import slice_bead_data, get_window_bead_num, set_seed
from exp import ExpDeepLearning

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='LBAD')

    # basic config
    parser.add_argument('--seed', type=int, default=72, help="set randomness")
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model', type=str, required=True,
                        choices=['LSTM_VAE', 'USAD', 'MADGAN', 'TAnoGAN'])

    # data loader
    parser.add_argument('--data_polarity', type=str, required=True, choices=['Anode', 'Cathode'])
    parser.add_argument('--model_type', type=str, required=True, choices=['reconstruction'])

    # model define 수정 필요
    parser.add_argument('--seq_len', type=int, default=100, help='input sequence length')
    parser.add_argument('--rnn_type', type=str, default='LSTM', choices=['LSTM', 'GRU'])
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=3, help='number of layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    set_seed(args.seed)

    Exp = ExpDeepLearning

    if args.is_training:
        # setting record of experiments
        setting = 'test'
        # setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
        #     args.model,
        #     args.features,
        #     args.seq_len,
        #     args.label_len,
        #     args.pred_len,
        #     args.d_model,
        #     args.n_heads,
        #     args.e_layers,
        #     args.d_layers,
        #     args.d_ff,
        #     args.factor,
        #     args.embed,
        #     args.distil,
        #     args.des, ii)

        exp = Exp(args)
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)
        exp.valid_test(setting)

        if not args.train_only:
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
        torch.cuda.empty_cache()

    else:
        # setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
        #     args.task_name,
        #     args.model_id,
        #     args.model,
        #     args.data,
        #     args.features,
        #     args.seq_len,
        #     args.label_len,
        #     args.pred_len,
        #     args.d_model,
        #     args.n_heads,
        #     args.e_layers,
        #     args.d_layers,
        #     args.d_ff,
        #     args.factor,
        #     args.embed,
        #     args.distil,
        #     args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()