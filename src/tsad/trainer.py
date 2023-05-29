from models import LSTM_AE, LSTM_VAE, USAD, DAGMM, Transformer
from utils.tools import EarlyStopping, adjust_learning_rate, check_graph
from utils.utils import load_model, CheckPoint, progress_bar
from utils.metrics import bf_search

import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import numpy as np
import wandb

warnings.filterwarnings('ignore')


class build_model():
    """
    Build and train or test a model

    Parameters
    ----------
    args : dict
        arguments
    params : dict
        model's hyper parameters
    savedir : str
        save directory

    Attributes
    ------------
    args : dict
        arguments
    params : dict
        model's hyper parameters
    savedir : str
        save directory
    device : torch.device
        device
    model : nn.module
        model

    Methods
    -------
    _build_model()
        Select the model you want to train or test and build it
        + multi gpu setting
    _acquire_device()
        check gpu usage
    _select_optimizer()
        select the optimizer (default AdamW)
    _select_criterion()
        select the criterion (default MSELoss)
    valid()

    train()

    test()


    """

    def __init__(self, args, params, savedir):
        super(build_model, self).__init__()
        self.args = args
        self.params = params
        self.savedir = savedir
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        model_dict = {
            'TF': Transformer,
            'LSTM_AE': LSTM_AE,
            'LSTM_VAE': LSTM_VAE,
            'USAD': USAD,
            'DAGMM': DAGMM,
        }

        model = model_dict[self.args.model].Model(self.params).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            print('using multi-gpu')
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _select_optimizer(self):
        if self.params.optim == 'adamw':
            model_optim = optim.AdamW(self.model.parameters(), lr=self.params.lr)
        elif self.params.optim == 'adam':
            model_optim = optim.Adam(self.model.parameters(), lr=self.params.lr)
        elif self.params.optim == 'sgd':
            model_optim = optim.SGD(self.model.parameters(), lr=self.params.lr)
        return model_optim

    def _select_criterion(self):
        if self.args.loss == 'mse':
            criterion = nn.MSELoss(reduction='none')
        elif self.args.loss == 'mae':
            criterion = nn.L1Loss(reduction='none')
        return criterion

    def valid(self, valid_loader, criterion, epoch):
        """
        validation

        Parameters
        ----------
        valid_loader : torch.utils.data.DataLoader
            given : torch.Tensor (shape=(batch size, window size, num of features))
                input time-series data
            ts : ndarray (shape=(batch size, window size))
                input timestamp
            answer : torch.Tensor (shape=(batch size, window size, num of features))
                target time-series data
                If model_type is prediction, window size is 1

        criterion

        Return
        ------
        total_loss

        """
        total_loss = []
        valid_score = []
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(valid_loader):
                batch_x = batch['given'].float().to(self.device)
                batch_y = batch['answer'].float().to(self.device)

                if self.args.model == 'LSTM_VAE':
                    output = self.model(batch_x)
                    loss = criterion(output[0], batch_y)
                    kl_loss = output[1]
                    loss += loss + kl_loss
                    valid_score.append(np.mean(loss.cpu().detach().numpy(), axis=2))
                    loss = torch.mean(loss)
                elif self.args.model == 'USAD':
                    output = self.model.forward(batch_x)
                    w1 = output[0].view(([batch_x.shape[0], batch_x.shape[1], batch_x.shape[2]]))
                    w2 = output[1].view(([batch_x.shape[0], batch_x.shape[1], batch_x.shape[2]]))
                    w3 = output[2].view(([batch_x.shape[0], batch_x.shape[1], batch_x.shape[2]]))
                    loss1 = 1 / (epoch + 1) * torch.mean((batch_y - w1) ** 2) + (1 - 1 / (epoch + 1)) * torch.mean(
                        (batch_y - w3) ** 2)
                    # Train AE2
                    output = self.model.forward(batch_x)
                    loss2 = 1 / (epoch + 1) * torch.mean((batch_y - w2) ** 2) - (1 - 1 / (epoch + 1)) * torch.mean(
                        (batch_y - w3) ** 2)
                    loss = loss1 + loss2
                    valid_score.append(loss.cpu().detach().numpy())
                    epoch_loss1 = torch.mean(loss1)
                    epoch_loss2 = torch.mean(loss2)
                    loss = epoch_loss1 + epoch_loss2
                elif self.args.model == 'DAGMM':
                    _, x_hat, z, gamma = self.model.forward(batch_x)
                    l1, l2 = criterion(x_hat, batch_x), criterion(gamma, batch_x)
                    valid_score.append(np.mean(loss.cpu().detach().numpy(), axis=2))
                    loss = torch.mean(l1) + torch.mean(l2)
                else:
                    output = self.model.forward(batch_x)
                    loss = criterion(output, batch_y)
                    valid_score.append(np.mean(loss.cpu().detach().numpy(), axis=2))
                    loss = torch.mean(loss)

                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss, valid_score

    def train(self, train_loader, valid_loader, test_loader, alpha=.5, beta=.5):
        """
        training

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            given : torch.Tensor (shape=(batch size, window size, num of features))
                input time-series data
            ts : ndarray (shape=(batch size, window size))
                input timestamp
            answer : torch.Tensor (shape=(batch size, window size, num of features))
                target time-series data
                If model_type is prediction, window size is 1
        valid_loader
            given : torch.Tensor (shape=(batch size, window size, num of features))
                input time-series data
            ts : ndarray (shape=(batch size, window size))
                input timestamp
            answer : torch.Tensor (shape=(batch size, window size, num of features))
                target time-series data
                If model_type is prediction, window size is 1
        test_loader
            given : torch.Tensor (shape=(batch size, window size, num of features))
                input time-series data
            ts : ndarray (shape=(batch size, window size))
                input timestamp
            answer : torch.Tensor (shape=(batch size, window size, num of features))
                target time-series data
                If model_type is prediction, window size is 1
            attack : ndarray
            attack labels

        Return
        ------
        model

        """
        time_now = time.time()

        best_metrics = None
        if self.args.resume is not None:
            print(f'resume version{self.args.resume}')
            weights, start_epoch, self.args.lr, best_metrics = load_model(resume=self.args.resume,
                                                                          logdir=self.savedir)
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(weights)
            else:
                self.model.load_state_dict(weights)

        # set checkpoint
        ckp = CheckPoint(logdir=self.savedir,
                         last_metrics=best_metrics,
                         metric_type='loss')

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        if self.args.model == 'USAD':
            model_optim1 = self._select_optimizer()
            model_optim2 = self._select_optimizer()
        else:
            model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # if self.args.use_amp:
        #     scaler = torch.cuda.amp.GradScaler()

        history = dict()

        for epoch in range(self.args.epochs):
            iter_count = 0
            train_loss = []
            train_score = []

            self.model.train()
            epoch_time = time.time()

            for batch_idx, batch in enumerate(train_loader):
                iter_count += 1
                batch_x = batch['given'].float().to(self.device)
                batch_y = batch['answer'].float().to(self.device)

                # if self.args.use_amp:
                #     with torch.cuda.amp.autocast():
                #         output = self.model.forward(batch_x)
                # else:
                #     output = self.model.forward(batch_x)

                if self.args.model == 'LSTM_VAE':
                    model_optim.zero_grad()
                    output = self.model.forward(batch_x)
                    loss = criterion(output[0], batch_y)
                    kl_loss = output[1]
                    loss += loss + kl_loss
                    train_score.append(np.mean(loss.cpu().detach().numpy(), axis=2))
                    loss = torch.mean(loss)
                    train_loss.append(loss.item())
                    loss.backward()
                    model_optim.step()

                elif self.args.model == 'USAD':
                    model_optim1.zero_grad()
                    output = self.model.forward(batch_x)
                    w1 = output[0].view(([batch_x.shape[0], batch_x.shape[1], batch_x.shape[2]]))
                    w3 = output[2].view(([batch_x.shape[0], batch_x.shape[1], batch_x.shape[2]]))
                    loss1 = 1 / (epoch + 1) * torch.mean((batch_y - w1) ** 2) + (1 - 1 / (epoch + 1)) * torch.mean(
                        (batch_y - w3) ** 2)
                    loss1.backward()
                    model_optim1.step()
                    # Train AE2
                    model_optim2.zero_grad()
                    output = self.model.forward(batch_x)
                    w1 = output[0].view(([batch_x.shape[0], batch_x.shape[1], batch_x.shape[2]]))
                    w2 = output[1].view(([batch_x.shape[0], batch_x.shape[1], batch_x.shape[2]]))
                    w3 = output[2].view(([batch_x.shape[0], batch_x.shape[1], batch_x.shape[2]]))
                    loss2 = 1 / (epoch + 1) * torch.mean((batch_y - w2) ** 2) - (1 - 1 / (epoch + 1)) * torch.mean(
                        (batch_y - w3) ** 2)
                    loss2.backward()
                    model_optim2.step()
                    loss = alpha * criterion(w1, batch_x) + beta * criterion(w2, batch_x)
                    train_score.append(np.mean(loss.cpu().detach().numpy(), axis=2))
                    loss = torch.mean(loss)
                    train_loss.append(loss.item())

                elif self.args.model == 'DAGMM':
                    model_optim.zero_grad()
                    _, x_hat, z, gamma = self.model.forward(batch_x)
                    l1, l2 = criterion(x_hat, batch_x), criterion(gamma, batch_x)
                    loss = l1 + l2
                    train_score.append(np.mean(loss.cpu().detach().numpy(), axis=2))
                    loss = torch.mean(loss)
                    train_loss.append(loss.item())
                    loss.backward()
                    model_optim.step()

                else:
                    model_optim.zero_grad()
                    output = self.model.forward(batch_x)
                    loss = criterion(output, batch_y)
                    train_score.append(np.mean(loss.cpu().detach().numpy(), axis=2))
                    loss = torch.mean(loss)
                    train_loss.append(loss.item())
                    loss.backward()
                    model_optim.step()

                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((self.args.epochs - epoch) * train_steps - batch_idx)
                progress_bar(current=batch_idx,
                             total=len(train_loader),
                             name='TRAIN',
                             msg=f'Total Loss: {np.mean(train_loss):.7f} | speed: {speed:.4f}s/iter; left time: {left_time:.4f}s')

            train_score = np.concatenate(train_score).flatten()
            train_loss = np.average(train_loss)
            valid_loss, valid_score = self.valid(valid_loader, criterion, epoch)

            dist, attack = self.inference(test_loader, epoch)

            # result save
            folder_path = os.path.join(self.savedir, 'results', f'epoch_{epoch}')
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            visual = check_graph(dist, attack, piece=4)
            visual.savefig(os.path.join(folder_path, f'graph.png'))

            if self.args.log_to_wandb:
                wandb.log({'graph': visual})

            np.save(os.path.join(folder_path, f'dist.npy'), dist)
            np.save(os.path.join(folder_path, f'attack.npy'), attack)

            print(f"Epoch: {epoch + 1}, Steps: {train_steps} cost time: {time.time() - epoch_time} | "
                  f"Train Loss: {train_loss:.7f} Vali Loss: {valid_loss:.7f} ")

            history.setdefault('train_loss', []).append(train_loss)
            history.setdefault('validation_loss', []).append(valid_loss)

            if self.args.log_to_wandb:
                wandb.log({
                    "train_loss": train_loss,
                    "validation_loss": valid_loss,
                })

            # check
            if self.args.model == 'USAD':
                ckp.check(epoch=epoch + 1, model=self.model, score=valid_loss, lr=model_optim1.param_groups[0]['lr'])
                adjust_learning_rate(model_optim1, epoch + 1, self.params)
                adjust_learning_rate(model_optim2, epoch + 1, self.params)
            else:
                ckp.check(epoch=epoch + 1, model=self.model, score=valid_loss, lr=model_optim.param_groups[0]['lr'])
                adjust_learning_rate(model_optim, epoch + 1, self.params)

            if early_stopping.validate(valid_loss):
                print("Early stopping")
                break

            # adjust_learning_rate(model_optim, epoch + 1, self.params)

        best_model_path = os.path.join(self.savedir, f'{ckp.best_epoch}.pth')

        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(torch.load(best_model_path)['weight'])
        else:
            self.model.load_state_dict(torch.load(best_model_path)['weight'])

        return history

    def test(self, test_loader, alpha=.5, beta=.5):
        """
        test

        Parameters
        ----------
        test_loader
            given : torch.Tensor (shape=(batch size, window size, num of features))
                input time-series data
            ts : ndarray (shape=(batch size, window size))
                input timestamp
            answer : torch.Tensor (shape=(batch size, window size, num of features))
                target time-series data
                If model_type is prediction, window size is 1
            attack : ndarray
            attack labels

        Returns
        -------

        """

        if self.args.resume is not None and self.args.train is not True:
            print(f'resume version{self.args.resume}')
            weights, start_epoch, self.args.lr, best_metrics = load_model(resume=self.args.resume,
                                                                          logdir=self.savedir)
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(weights)
            else:
                self.model.load_state_dict(weights)

        dist = []
        attack = []
        pred = []
        history = dict()
        criterion = self._select_criterion()

        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                batch_x = batch['given'].float().to(self.device)
                batch_y = batch['answer'].float().to(self.device)

                if self.args.model == 'LSTM_VAE':
                    predictions = self.model.forward(batch_x)
                    score = criterion(predictions[0], batch_y).cpu().detach().numpy()
                    pred.append(predictions[0].cpu().detach().numpy())
                    dist.append(np.mean(score, axis=2))

                elif self.args.model == 'USAD':
                    predictions = self.model.forward(batch_x)
                    w1 = predictions[0].view(([batch_x.shape[0], batch_x.shape[1], batch_x.shape[2]]))
                    w2 = predictions[1].view(([batch_x.shape[0], batch_x.shape[1], batch_x.shape[2]]))
                    pred.append((alpha * (batch_x - w1) + beta * (batch_x - w2)).cpu().detach().numpy())
                    dist.append((alpha * torch.mean((batch_x - w1) ** 2, axis=2) + beta * torch.mean(
                        (batch_x - w2) ** 2, axis=2)).detach().cpu())

                elif self.args.model == 'DAGMM':
                    _, x_hat, _, _ = self.model.forward(batch_x)
                    score = criterion(x_hat, batch_y).cpu().detach().numpy()
                    pred.append(x_hat.cpu().detach().numpy())
                    dist.append(np.mean(score, axis=2))

                else:
                    predictions = self.model.forward(batch_x)
                    score = criterion(predictions, batch_y).cpu().detach().numpy()
                    pred.append(predictions.cpu().detach().numpy())
                    dist.append(np.mean(score, axis=2))

                attack.append(batch['attack'].squeeze().numpy())

        dist = np.concatenate(dist).flatten()
        attack = np.concatenate(attack).flatten()
        pred = np.concatenate(pred)

        # score
        scores = dist.copy()
        [f1, precision, recall, TP, TN, FP, FN, roc_auc, auprc, latency], threshold = bf_search(dist, attack,
                                                                                                start=np.percentile(
                                                                                                    scores,
                                                                                                    90),
                                                                                                end=max(scores),
                                                                                                step_num=1000,
                                                                                                verbose=False)

        print(f"precision: {precision:.4f} recall: {recall:.4f} f1: {f1:.4f} ROC_AUC: {roc_auc:.4f}")
        print(f"TP: {TP} TN: {TN} FP: {FP} FN: {FN}")

        history.setdefault('precision', []).append(precision)
        history.setdefault('recall', []).append(recall)
        history.setdefault('f1', []).append(f1)
        history.setdefault('roc_auc', []).append(roc_auc)

        visual = check_graph(dist, attack, piece=4, threshold=threshold)
        figure_path = os.path.join(self.savedir, 'fig')
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)
        visual.savefig(os.path.join(figure_path, f'whole.png'))

        if self.args.log_to_wandb:
            wandb.log({'graph': visual})

        # result save
        folder_path = os.path.join(self.savedir, 'results')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(os.path.join(folder_path, f'dist.npy'), dist)
        np.save(os.path.join(folder_path, f'attack.npy'), attack)
        np.save(os.path.join(folder_path, f'pred.npy'), pred)

        return history

    def inference(self, test_loader, epoch, alpha=.5, beta=.5):
        """
        inference

        Parameters
        ----------
        test_loader
            given : torch.Tensor (shape=(batch size, window size, num of features))
                input time-series data
            ts : ndarray (shape=(batch size, window size))
                input timestamp
            answer : torch.Tensor (shape=(batch size, window size, num of features))
                target time-series data
                If model_type is prediction, window size is 1
            attack : ndarray
            attack labels
        epoch : int
            train epoch

        Returns
        -------

        """
        dist = []
        attack = []
        criterion = self._select_criterion()

        self.model.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                batch_x = batch['given'].float().to(self.device)
                batch_y = batch['answer'].float().to(self.device)

                if self.args.model == 'LSTM_VAE':
                    predictions = self.model.forward(batch_x)
                    score = criterion(predictions[0], batch_y).cpu().detach().numpy()
                    dist.append(np.mean(score, axis=2))

                elif self.args.model == 'USAD':
                    predictions = self.model.forward(batch_x)
                    w1 = predictions[0].view(([batch_x.shape[0], batch_x.shape[1], batch_x.shape[2]]))
                    w2 = predictions[1].view(([batch_x.shape[0], batch_x.shape[1], batch_x.shape[2]]))
                    dist.append((alpha * torch.mean((batch_x - w1) ** 2, axis=2) + beta * torch.mean(
                        (batch_x - w2) ** 2, axis=2)).detach().cpu())

                elif self.args.model == 'DAGMM':
                    _, x_hat, z, gamma = self.model.forward(batch_x)
                    l1, l2 = criterion(x_hat, batch_y), criterion(gamma, batch_y)
                    score = l1 + l2
                    dist.append(np.mean(score.cpu().detach().numpy(), axis=2))

                else:
                    predictions = self.model.forward(batch_x)
                    score = criterion(predictions, batch_y).cpu().detach().numpy()
                    dist.append(np.mean(score, axis=2))

                attack.append(batch['attack'].squeeze().numpy())

        dist = np.concatenate(dist).flatten()
        attack = np.concatenate(attack).flatten()

        return dist, attack