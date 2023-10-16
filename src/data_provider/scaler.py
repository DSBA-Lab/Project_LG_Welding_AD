from sklearn.preprocessing import MinMaxScaler, StandardScaler

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


