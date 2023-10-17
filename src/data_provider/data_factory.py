import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader
from data_provider.dataset import BuildDataset
from utils import slice_bead_data, get_window_bead_num, set_seed


def data_provider(args, flag):
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size  # bsz for train and valid

    data_path = args.data_path
    Normal = sorted([x for x in Path(os.path.join(data_path, f"{args.data_polarity}/Normal")).glob("*.csv")])
    Normal = sorted(Normal)
    Abnormal = sorted([x for x in Path(os.path.join(data_path, f"{args.data_polarity}/Abnormal")).glob("*.csv")])
    Abnormal = sorted(Abnormal)

    # 데이터의 bead를 detect하는 코드
    data_folder_list = Normal + Abnormal
    num_train_dataset = args.num_train
    Train = pd.DataFrame(columns=['LO', 'BR', 'NIR', 'VIS', 'label', 'dataset_idx', 'bead_num'])
    Test = pd.DataFrame(columns=['LO', 'BR', 'NIR', 'VIS', 'label', 'dataset_idx', 'bead_num'])

    for i in range(num_train_dataset):
        bead_i = slice_bead_data(str(data_folder_list[i]),
                                 set_bead_100=args.set_bead_100,
                                 set_end_to_end=args.set_end_to_end)
        Train = pd.concat([Train, bead_i])

    for i in range(num_train_dataset, len(data_folder_list)):
        bead_i = slice_bead_data(str(data_folder_list[i]),
                                 set_bead_100=args.set_bead_100,
                                 set_end_to_end=args.set_end_to_end)
        Test = pd.concat([Test, bead_i])

    # dataset.py의 load_dataset 함수
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
    train_window_bead_num_list = get_window_bead_num(Train, window_size=args.seq_len, slide_size=args.slide_size)
    test_window_bead_num_list = get_window_bead_num(Test, window_size=args.seq_len, slide_size=args.slide_size)

    print(f'Number of train_window_bead_num_list: {len(train_window_bead_num_list)}')
    print(f'Number of test_window_bead_num_list: {len(test_window_bead_num_list)}')

    # Dataloader.py에 있는 데이터셋 구축 부분
    # build dataset
    if flag == 'test':
        dataset = BuildDataset(tst, tst_ts, args.seq_len, args.slide_size,
                               attacks=tst_label, model_type=args.model_type)
    else:
        dataset = BuildDataset(trn, trn_ts, args.seq_len, args.slide_size,
                               attacks=None, model_type=args.model_type)


    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle_flag,
                            num_workers=args.num_workers,
                            drop_last=drop_last)


    return dataset, dataloader