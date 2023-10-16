from torch.utils.data import Dataset
import torch
import pandas as pd
import os
import numpy as np
from pathlib import Path
from datetime import timedelta
import dateutil
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def dataframe_from_csv(target):
    return pd.read_csv(target).rename(columns=lambda x: x.strip())

def dataframe_from_csvs(targets):
    return pd.concat([dataframe_from_csv(x) for x in targets])

class BuildDataset(Dataset):
    """
    Build Dataset
    Parameters
    ----------
    data : ndarray(dtype=float, ndim=2, shape=(time, num of features))
        time-series data
    timestamps : ndarray
        time-series data's timestamp
    window_size : int
        window size for time series condition
    slide_size : int(default=1)
        moving window size
    attacks : ndarray(dtype=np.int, ndim=2, shape=(time,))
        attack label
    model_type : str(default=reconstruction)
        model type (reconstruction, prediction)
    Attributes
    ----------
    ts : ndarray
        time-series data's timestamp
    tag_values : ndarray(dtype=np.float32, ndim=2, shape=(time, num of features))
        time-series data
    window_size : int
        window size for time series condition
    model_type : str
        model type (reconstruction, prediction)
    valid_idxs : list
        first index of data divided by window
    Methods
    -------
    __len__()
        return num of valid windows
    __getitem__(idx)
        return data(given, ts, answer, attack)
    """

    def __init__(self,
                 data: np.ndarray,
                 timestamps: np.ndarray,
                 window_size: int,
                 slide_size: int = 1,
                 attacks: np.ndarray = None,
                 model_type: str = 'reconstruction'):
        self.ts = np.array(timestamps)
        self.tag_values = np.array(data, dtype=np.float32)
        self.window_size = window_size
        self.model_type = model_type
        self.slide_size = slide_size

        self.valid_idxs = []
        if self.model_type == 'reconstruction':
            for L in range(0, len(self.ts) - window_size + 1, slide_size):
                R = L + window_size - 1
                try:
                    if dateutil.parser.parse(self.ts[R]) - dateutil.parser.parse(
                            self.ts[L]
                    ) == timedelta(seconds=window_size - 1):
                        self.valid_idxs.append(L)
                except:
                    if self.ts[R] - self.ts[L] == window_size - 1:
                        self.valid_idxs.append(L)
        elif self.model_type == 'prediction':
            for L in range(0, len(self.ts) - window_size, slide_size):
                R = L + window_size
                try:
                    if dateutil.parser.parse(self.ts[R]) - dateutil.parser.parse(
                            self.ts[L]
                    ) == timedelta(seconds=window_size):
                        self.valid_idxs.append(L)
                except:
                    if self.ts[R] - self.ts[L] == window_size:
                        self.valid_idxs.append(L)

        print(f"# of valid windows: {len(self.valid_idxs)}")

        if attacks is not None:
            self.attacks = np.array(attacks, dtype=np.float32)
            self.with_attack = True
        else:
            self.with_attack = False

    def __len__(self):
        return len(self.valid_idxs)

    def __getitem__(self, idx: str) -> dict:
        i = self.valid_idxs[idx]
        last = i + self.window_size
        item = {"given": torch.from_numpy(self.tag_values[i:last])}
        # valid_values = self.tag_values[self.valid_idxs]
        # item['window_dataset'] = torch.tensor(valid_values[np.arange(self.window_size)[None, :] + np.arange(len(self.valid_idxs)-self.window_size)[:, None]])
        if self.model_type == 'reconstruction':
            # item["ts"] = self.ts[i:last]
            item["answer"] = torch.from_numpy(self.tag_values[i:last])
            if self.with_attack:
                item['attack'] = self.attacks[i:last]
        elif self.model_type == 'prediction':
            # item["ts"] = self.ts[last]
            item["answer"] = torch.from_numpy(self.tag_values[last])
            if self.with_attack:
                item['attack'] = self.attacks[last]
        return item

class scaler:
    """
    normalize the data

    Parameters
    ----------
    scale : str

    Attributes
    ------------
    scaler : sklearn.preprocessing

    Methods
    -------
    fit(traindataset)
        fit the scaler
    transform(dataset)
        transform the dataset
    """

    def __init__(self, scale: str = None):
        self.scale = scale
        self.scaler = None

    def fit(self, traindataset):
        if self.scale == 'minmax':
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
            self.scaler.fit(traindataset)
        elif self.scale == 'minmax_square':
            self.scaler = MinMaxScaler()
            self.scaler.fit(traindataset)
        elif self.scale == 'minmax_m1p1':
            pass
        elif self.scale == 'standard':
            self.scaler = StandardScaler()
            self.scaler.fit(traindataset)
        else:
            raise ValueError(f'Unknown scaler: {self.scale}')

    def transform(self, dataset):
        if self.scale == 'minmax':
            return self.scaler.transform(dataset)
        elif self.scale == 'minmax_square':
            return self.scaler.transform(dataset) ** 2
        elif self.scale == 'minmax_m1p1':
            return 2 * (dataset / dataset.max(axis=0)) - 1
        elif self.scale == 'standard':
            return self.scaler.transform(dataset)
        else:
            raise ValueError(f'Unknown scaler: {self.scale}')
