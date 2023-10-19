import argparse
import torch

from utils import set_seed
from exp import ExpDeepLearning

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='LBAD')

    # basic config
    parser.add_argument('--seed', type=int, default=72, help="set randomness")
    parser.add_argument('--task_name', type=str, required=True, default='window_mean',
                        help='task name, options:[window_mean, window_max]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--train_only', type=int, default=0, help='only train')
    parser.add_argument('--model', type=str, required=True,
                        choices=['LSTM_VAE', 'USAD', 'MADGAN', 'TAnoGAN'])

    # data loader
    parser.add_argument('--data_path', type=str, default='../data/230507', help='data directory')
    parser.add_argument('--data_polarity', type=str, required=True, choices=['Anode', 'Cathode'])
    parser.add_argument('--num_train', type=int, default=28, help='number of train data')
    parser.add_argument('--model_type', type=str, required=True, choices=['reconstruction'])
    parser.add_argument('--feature_num', type=int, default=4, help='feature number')
    parser.add_argument('--slide_size', type=int, default=100, help='slide size')
    parser.add_argument('--set_bead_100', type=int, default=1)
    parser.add_argument('--set_end_to_end', type=int, default=1)
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # model define 수정 필요
    parser.add_argument('--seq_len', type=int, default=100, help='input sequence length')
    parser.add_argument('--rnn_type', type=str, default='LSTM', choices=['LSTM', 'GRU'])
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=3, help='number of layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--latent_length', type=int, default=20, help='latent length')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--Lambda', type = float, help='Lambda for TAnoGAN optimization', default=0.1)
    parser.add_argument('--iterations', type = int, help='Iterations for TAnoGAN optimization', default=100)


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
        setting = '{}_{}_{}_ft{}_wl{}_sl{}_hs{}_nl{}_dr{}_ll{}'.format(
            args.model,
            args.data_polarity,
            args.task_name,
            args.feature_num,
            args.seq_len,
            args.slide_size,
            args.hidden_size,
            args.num_layers,
            args.dropout,
            args.latent_length)

        exp = Exp(args)
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)
        # exp.valid_test(setting)

        if not args.train_only:
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, args.seq_len, task=args.task_name)
        torch.cuda.empty_cache()

    else:
        setting = '{}_{}_{}_ft{}_wl{}_sl{}_hs{}_nl{}_dr{}_ll{}'.format(
            args.model,
            args.data_polarity,
            args.task_name,
            args.feature_num,
            args.seq_len,
            args.slide_size,
            args.hidden_size,
            args.num_layers,
            args.dropout,
            args.latent_length)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, args.seq_len, task=args.task_name, test=1)
        torch.cuda.empty_cache()