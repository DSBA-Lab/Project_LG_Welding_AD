import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


from sklearn.preprocessing import MinMaxScaler, StandardScaler
import sys
sys.path.append("..")
from data_provider import BuildDataset
from models import LSTM_VAE, MadGAN, UsadModel, TAnoGAN_MOCAP


from omegaconf import OmegaConf
from utils import slice_bead_data, get_window_bead_num, set_seed


def main():
    """
    main experiment

    Parameters
    ----------
    """
    set_seed(args.seed)

    '''set data'''
    #data_params
    data_polarity=config['data_params']['data_polarity'] # 'Anode' or 'Cathode'
    window_size = config['data_params']['window_size']
    slide_size = config['data_params']['slide_size']
    model_type = config[args.model]['model_type']
    loader_params=config['loader_params']
    num_train = config['data_params']['num_train']
    set_bead_100=config['data_params']['set_bead_100']
    set_end_to_end=config['data_params']['set_end_to_end']
    
    # 최소로  raw data를 가지고 있는 폴더
    wd = Path.cwd()
    data_path = wd.parent / config['path']
    Normal = sorted([x for x in Path(os.path.join(data_path, f"{data_polarity}/Normal")).glob("*.csv")])
    Normal = sorted(Normal)
    Abnormal = sorted([x for x in Path(os.path.join(data_path, f"{data_polarity}/Abnormal")).glob("*.csv")])
    Abnormal = sorted(Abnormal)

    # 데이터의 bead를 detect하는 코드
    data_folder_list = Normal + Abnormal
    num_train_dataset = num_train
    Train = pd.DataFrame(columns=['LO', 'BR', 'NIR', 'VIS', 'label','dataset_idx','bead_num'])
    Test = pd.DataFrame(columns=['LO', 'BR', 'NIR', 'VIS', 'label','dataset_idx','bead_num'])
    for i in range(num_train_dataset):
        bead_i = slice_bead_data(str(data_folder_list[i]),set_bead_100=set_bead_100, set_end_to_end=set_end_to_end)
        Train = pd.concat([Train, bead_i])

    for i in range(num_train_dataset, len(data_folder_list)):
        bead_i = slice_bead_data(str(data_folder_list[i]),set_bead_100=set_bead_100, set_end_to_end=set_end_to_end)
        Test = pd.concat([Test, bead_i])


    #dataset.py의 load_dataset 함수
    trn = Train[['LO', 'BR', 'NIR', 'VIS']]
    trn = trn.reset_index(drop=True)
    trn = trn.dropna()
    trn_ts = trn.index

    tst = Test[['LO', 'BR', 'NIR', 'VIS']]
    tst_ts = np.arange(len(tst))
    tst_label = Test[['label']]

    # 데이터 전처리
    scaler = StandardScaler()
    scaler.fit(trn)
    trn = scaler.transform(trn)
    tst = scaler.transform(tst)

    # bead_num check
    train_window_bead_num_list = get_window_bead_num(Train, window_size=window_size, slide_size=slide_size)
    test_window_bead_num_list = get_window_bead_num(Test, window_size=window_size, slide_size=slide_size)

    # Dataloader.py에 있는 데이터셋 구축 부분
    # build dataset
    trn_dataset = BuildDataset(trn, trn_ts, window_size, slide_size,
                                attacks=None, model_type=model_type)
    tst_dataset = BuildDataset(tst, tst_ts, window_size, slide_size,
                                attacks=tst_label, model_type=model_type)

    print(f'Number of train_window_bead_num_list: {len(train_window_bead_num_list)}')
    print(f'Number of test_window_bead_num_list: {len(test_window_bead_num_list)}')

    trn_dataloader = torch.utils.data.DataLoader(trn_dataset,
                                                    batch_size=loader_params['batch_size'],
                                                    shuffle=loader_params['shuffle'],
                                                    num_workers=loader_params['num_workers'],
                                                    pin_memory=loader_params['pin_memory'],
                                                    drop_last=False)

    tst_dataloader = torch.utils.data.DataLoader(tst_dataset,
                                                    batch_size=loader_params['batch_size'],
                                                    shuffle=loader_params['shuffle'],
                                                    num_workers=loader_params['num_workers'],
                                                    pin_memory=loader_params['pin_memory'],
                                                    drop_last=False)

    '''set model'''
    # data 완성 이하는 model에 대한 parameter 설정
    model_params = config[args.model]
    model_params['feature_num'] = int(next(iter(trn_dataloader))['given'].shape[2])
    model_params['sequence_length']= window_size
    model_params['batch_size'] = loader_params['batch_size']
        
    # model
    device = torch.device('cuda:{}'.format(model_params['gpu']) if torch.cuda.is_available() else 'cpu')
    if args.model == 'LSTM_VAE':
        model = LSTM_VAE(model_params).to(device)
    elif args.model == 'USAD':
        # w_size = window_size*len(Cathode_Train_X_scaled[0]) # window size * feature 수
        model = UsadModel(model_params).to(device)
    elif args.model == 'MADGAN':
        #모델 선언을 아래의 학습과정에서 동시에 수행
        pass
    elif args.model == 'TAnoGAN':
        model = TAnoGAN_MOCAP(model_params).to(device)

    #train model & test model
    if args.model == 'LSTM_VAE':
      model.fit(train_loader = trn_dataloader, train_epochs=args.epochs)
      scores, attack, _ = model.test(test_loader = tst_dataloader)
      # split scores by window_size
      windowed_scores = scores.reshape(-1, window_size)
      windowed_attack = attack.reshape(-1, window_size)

    elif args.model == 'USAD':
      # USAD의 경우 value와 window가 대응이 되어야 하기 때문에 slide size = 1
      model.fit(train_loader = trn_dataloader, train_epochs=args.epochs)
      scores=model.test(test_loader = tst_dataloader)


      windows_labels=[]
      for i in range(0, len(tst_label), slide_size):
          windows_labels.append(list(np.int_(tst_label[i:i+window_size])))

      y_test = [1.0 if (np.sum(window) > 0) else 0 for window in windows_labels]
      y_test_trimmed = y_test[:-4]
      y_test_trimmed = np.array(y_test_trimmed)
      attack = y_test_trimmed

      # split scores by window_size
      windowed_scores = scores
      windowed_attack = attack
      #window max 구하는 이후 과정이 다른 모델들과 조금 다름


    elif args.model == 'MADGAN':
      modeltrain = MadGAN.Modeltrain(
                              trainloader=trn_dataloader,
                              validloader=None,
                              epochs=50,
                              device=device)
      savedir = "/root/Project_LG_Welding_AD/result/madgan"
      modeltest = MadGAN.ModelTest(models=[modeltrain.G, modeltrain.D], 
                              savedir=savedir, 
                              device=device)
      scores = modeltest.inference(lam=0.5, 
                            testloader=tst_dataloader, 
                            optim_iter=25, 
                            method = 'append',
                            real_time=False)

      windows_labels=[]
      for i in range(0, len(tst_label), slide_size):
          if i + window_size > len(tst_label):
              pass
          else:
              windows_labels.append(list(np.int_(tst_label[i:i+window_size])))
      attack = np.array(windows_labels).squeeze()
      windowed_scores, windowed_attack = scores, attack
      
    elif args.model == 'TAnoGAN':
        pass

    # 이후의 과정은 windowed_scores, windowed_attack에 대해 모든 모델이 동일

if __name__ == '__main__':
    # set arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        choices=['LSTM_VAE', 'USAD', 'MADGAN', 'TAnoGAN'])
    parser.add_argument('--data_polarity', type=str, required=True, choices=['Anode', 'Cathode'])
    # train options
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
    # parser.add_argument('--patience', type=int, default=5, help='early stopping patience')

    # setting
    parser.add_argument('--seed', type=int, default=72, help="set randomness")
    parser.add_argument('--configure', type=str, default='config.yaml', help='configure file load')

    # parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    # parser.add_argument('--gpu', type=int, default=0, help='gpu number')
    # parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    # parser.add_argument('--devices', type=str, default='0,1', help='device ids of multiple gpus')
    # parser.add_argument('--log_to_wandb', action='store_true', default=False)
    # parser.add_argument('--batch_size', type=int, default=32, help='batch size for dataloader')

    args = parser.parse_args()

    # yaml file load
    with open(args.configure) as f:
        config = OmegaConf.load(f)

    main()