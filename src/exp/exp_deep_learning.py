import pdb

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from utils import bf_search, check_graph
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import warnings
import numpy as np

warnings.filterwarnings('ignore')

class ExpDeepLearning(Exp_Basic):
    def __init__(self, args):
        super(ExpDeepLearning, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss(reduction='none')
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        # vali_data, vali_loader = self._get_data(flag='val')
        # test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        self.model.fit(train_loader,
                       self.args.train_epochs,
                       model_optim,
                       criterion,
                       self.device,
                       path,
                       early_stopping)

        return self.model

    def test(self, setting, window_size, test=0, task='window_mean'):
        test_data, test_loader = self._get_data(flag='test')
        # train_data, train_loader = self._get_data(flag='train')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        criterion = self._select_criterion()
        scores, attack, _ = self.model.test(test_loader, criterion, self.device)

        windows_scores = np.array([])
        if task == 'window_mean':
            for i in range(0, len(scores), window_size):
                windows_scores = np.append(windows_scores, np.mean(scores[i:i + window_size]))
        elif task == 'window_max':
            for i in range(0, len(scores), window_size):
                windows_scores = np.append(windows_scores, np.max(scores[i:i + window_size]))

        windows_labels = np.array([], dtype=float)  # 0, 1: max in window
        for i in range(0, len(attack), window_size):
            windows_labels = np.append(windows_labels, np.max(attack[i:i + window_size]))

        windows_labels = windows_labels.astype(float)
        [f1, precision, recall, _, _, _, _, _, _, _], threshold = bf_search(windows_scores,
                                                                            windows_labels,
                                                                            start=min(
                                                                                windows_scores),
                                                                            end=np.percentile(
                                                                                windows_scores,
                                                                                95),
                                                                            step_num=1000,
                                                                            K=100,
                                                                            verbose=False)

        print("Threshold :", threshold)
        result = f'F1-score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}'
        print(result)

        f = open(folder_path + "result_anomaly_detection.txt", 'a')
        f.write(setting + "  \n")
        f.write(result)
        f.write('\n')
        f.write('\n')
        f.close()

        score_graph = check_graph(windows_scores, windows_labels, 1, threshold)
        score_graph.savefig(folder_path + 'score_graph.png')

        return
