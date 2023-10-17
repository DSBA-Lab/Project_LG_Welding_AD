import pdb

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from utils import bf_search
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
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
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
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

        # attens_energy = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if self.args.model != 'USAD':
            scores, attack, _ = self.model.test(test_loader, self.criterion)

            windows_scores = np.array([])
            if task == 'window_mean':
                for i in range(0, len(scores), window_size):
                    windows_mean_scores = np.append(windows_scores, np.mean(scores[i:i + window_size]))
            elif task == 'window_max':
                for i in range(0, len(scores), window_size):
                    windows_max_scores = np.append(windows_max_scores, np.max(scores[i:i + window_size]))
        else:
            score, attack, window_score = self.model.test(test_loader, self.criterion)

        windows_labels = np.array([], dtype=float)  # 0, 1: max in window
        for i in range(0, len(attack), window_size):
            windows_labels = np.append(windows_labels, np.max(attack[i:i + window_size]))

        windows_labels = windows_labels.astype(float)
        [f1, precision, recall, _, _, _, _, auroc, _], threshold = bf_search(windows_scores,
                                                                             windows_labels,
                                                                             start=min(
                                                                                 windows_scores),
                                                                             end=np.percentile(
                                                                                 windows_scores,
                                                                                 95),
                                                                             step_num=10000,
                                                                             K=100,
                                                                             verbose=True)

        print("Threshold :", threshold)
        result = f'F1-score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, AUROC: {auroc:.4f}'
        print(result)

        f = open("result_anomaly_detection.txt", 'a')
        f.write(setting + "  \n")
        f.write(result)
        f.write('\n')
        f.write('\n')
        f.close()

        pdb.set_trace()

        threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
        print("Threshold :", threshold)

        # (3) evaluation on the test set
        pred = (test_energy > threshold).astype(int)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # (4) detection adjustment
        gt, pred = adjustment(gt, pred)

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))

        f = open("result_anomaly_detection.txt", 'a')
        f.write(setting + "  \n")
        f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))
        f.write('\n')
        f.write('\n')
        f.close()
        return
