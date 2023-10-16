import os
import sys

import torch
from torch import nn, optim
from torch import distributions
from torch.utils.data import DataLoader
from torch.autograd import Variable

import numpy as np
from tqdm.auto import trange
from sklearn.base import BaseEstimator
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt


# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# from utils.utils import progress_bar
# Encoder class
# Decoder class
# Lambda class
# Model(LSTM-VAE class)
class MODEL(BaseEstimator, nn.Module):
    '''
    Variational recurrent auto-encoder.

    :param sequence_length: length of the input sequence
    :param number_of_features: number of input features
    :param hidden_size:  hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param batch_size: number of timeseries in a single batch
    :param learning_rate: the learning rate of the module
    :param rnn_type: GRU/LSTM to be used as a basic building rnn_type
    :param dropout_rate: The probability of a node being dropped-out
    '''

    def __init__(self, params):
        super(MODEL, self).__init__()  ###MODEL 로변경해야함 . reference 사용안 할라면

        # params
        # params
        self.sequence_length = params['sequence_length']  # 100
        # torch.Size([32, 100, 25]) : batch크기, window 크기, feature_num 수
        self.feature_num = params['feature_num']
        self.hidden_size = params['hidden_size']
        self.hidden_layer_depth = params['hidden_layer_depth']
        self.latent_length = params['latent_length']
        self.batch_size = params['batch_size']
        self.rnn_type = params['rnn_type']
        self.dropout_rate = params['dropout_rate']
        self.dtype = torch.FloatTensor
        self.lr = params['lr']
        self.optim = params['optim']
        self.criterion = params['criterion']
        self.device = torch.device('cuda:{}'.format(params['gpu']) if torch.cuda.is_available() else 'cpu')

        print('Use GPU: cuda:{}'.format(params['gpu']) if torch.cuda.is_available() else 'cpu')
        # models
        self.encoder = Encoder(number_of_features=self.feature_num,
                               hidden_size=self.hidden_size,
                               hidden_layer_depth=self.hidden_layer_depth,
                               latent_length=self.latent_length,
                               dropout=self.dropout_rate,
                               rnn_type=self.rnn_type)

        self.lmbd = Lambda(hidden_size=self.hidden_size,
                           latent_length=self.latent_length)

        self.decoder = Decoder(sequence_length=self.sequence_length,
                               batch_size=self.batch_size,
                               hidden_size=self.hidden_size,
                               hidden_layer_depth=self.hidden_layer_depth,
                               latent_length=self.latent_length,
                               output_size=self.feature_num,
                               rnn_type=self.rnn_type,
                               dtype=self.dtype)

    def _select_optimizer(self):
        if self.optim == 'adamw':
            model_optim = optim.AdamW(self.parameters(), lr=self.lr)
        elif self.optim == 'adam':
            model_optim = optim.Adam(self.parameters(), lr=self.lr)
        elif self.optim == 'sgd':
            model_optim = optim.SGD(self.parameters(), lr=self.lr)
        return model_optim

    def _select_criterion(self):
        if self.criterion == 'mse':
            criterion = nn.MSELoss(reduction='none')
        elif self.criterion == 'mae':
            criterion = nn.L1Loss(reduction='none')
        return criterion

    # init hidden
    def init_hidden(self, batch_size):
        if self.rnn_type == 'GRU':
            return torch.zeros(self.hidden_layer_depth, batch_size, self.hidden_size).cuda()
        elif self.rnn_type == 'LSTM':
            return (torch.zeros(self.hidden_layer_depth, batch_size, self.hidden_size).cuda(),
                    torch.zeros(self.hidden_layer_depth, batch_size, self.hidden_size).cuda())
        else:
            raise Exception('Unknown rnn_type. Valid options: "gru", "lstm"')

    def forward(self, x):
        '''
        Forward propagation which involves one pass from inputs to encoder to lambda to decoder

        :param x: input tensor
        :return: the decoded output, latent vector
        '''
        # initialize hidden state
        B, S, F = x.size()  # [batch_size, seq_len, feature_size]
        hidden = self.init_hidden(B)

        # pipeline
        x_encoded = self.encoder(x, hidden)
        latent = self.lmbd(x_encoded)
        x_decoded = self.decoder(latent, B)

        # for loss computation
        latent_mean, latent_logvar = self.lmbd.latent_mean, self.lmbd.latent_logvar

        kl_loss = -0.5 * torch.mean(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())

        # x_decoded = x_decoded.permute(1,0,2)
        return [x_decoded, kl_loss]

    def fit(self, train_loader, train_epochs):  # valid_loader, test_loader

        time_now = time.time()

        best_metrics = None

        train_steps = len(train_loader)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        all_loss = []

        max_index = []
        for epoch in tqdm(range(train_epochs), desc='Epochs', position=0, leave=True):
            iter_count = 0
            train_loss = []
            train_score = []

            self.train()
            epoch_time = time.time()

            for batch_idx, batch in tqdm(enumerate(train_loader), desc='Batches', position=0, leave=False,
                                         total=len(train_loader)):
                iter_count += 1
                batch_x = batch['given'].float().to(self.device)
                batch_y = batch['answer'].float().to(self.device)

                model_optim.zero_grad()
                output = self.forward(batch_x)
                loss = criterion(output[0], batch_y)
                kl_loss = output[1]
                loss += loss + kl_loss
                train_score.append(np.mean(loss.cpu().detach().numpy(), axis=2))
                loss = torch.mean(loss)
                train_loss.append(loss.item())
                all_loss.append(loss.item())
                loss.backward()
                model_optim.step()

                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((train_epochs - epoch) * train_steps - batch_idx)

            train_score = np.concatenate(train_score).flatten()
            train_loss = np.average(train_loss)
            # if epoch % 10 == 0:
            # plot all_loss: train_loss list
        # plt.figure(figsize=(10, 5))
        # plt.plot(all_loss, label='train_loss')
        # plt.legend()
        # give max x index
        max_index.append(all_loss.index(max(all_loss)))
        plt.show()

        print(f"Epoch: {epoch + 1}, Steps: {train_steps} cost time: {time.time() - epoch_time} | "
              f"Train Loss: {train_loss:.7f} ")  # Vali Loss: {valid_loss:.7f}

        return max_index

    def test(self, test_loader):

        dist = []
        attack = []
        pred = []
        criterion = self._select_criterion()

        self.eval()
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(test_loader), desc='Batches', position=0, leave=True,
                                         total=len(test_loader)):
                batch_x = batch['given'].float().to(self.device)
                batch_y = batch['answer'].float().to(self.device)

                predictions = self.forward(batch_x)
                score = criterion(predictions[0], batch_y).cpu().detach().numpy()
                pred.append(predictions[0].cpu().detach().numpy())
                dist.append(np.mean(score, axis=2))

                attack.append(batch['attack'].squeeze().numpy())

        dist = np.concatenate(dist).flatten()
        attack = np.concatenate(attack).flatten()
        pred = np.concatenate(pred)

        # score
        scores = dist.copy()

        # # result save
        # folder_path = os.path.join(self.savedir, 'results')
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        # np.save(os.path.join(folder_path, f'dist.npy'), dist)
        # np.save(os.path.join(folder_path, f'attack.npy'), attack)
        # np.save(os.path.join(folder_path, f'pred.npy'), pred)

        return scores, attack, pred



class Encoder(nn.Module):
    '''
    Encoder network containing enrolled LSTM/GRU

    :param number_of_features: number of input features
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param dropout: percentage of nodes to dropout
    :param rnn_type: LSTM/GRU rnn_type
    '''

    def __init__(self, number_of_features, hidden_size, hidden_layer_depth, latent_length, dropout, rnn_type='LSTM'):

        super(Encoder, self).__init__()

        self.number_of_features = number_of_features
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.dropout = dropout

        ## input = feature dimension
        ## hidden size = 바꿔주고 싶은 hidden dimension
        ## batch_first = input의 데이터 구조가 batch first로 바뀐다
        ## RNN에 feed할 때, input의 차원은 [Seq_len, Batch_size, Hidden_size]인데
        ## batch_first = True라면 ,  [Batch_size, Seq_len, Hidden_size]
        ## hidden은 다음과 같이 정의 : [num_layers * num_directions, batch, hidden_size]
        if rnn_type == 'LSTM':
            self.model = nn.LSTM(input_size=self.number_of_features, \
                                 hidden_size=self.hidden_size, \
                                 num_layers=self.hidden_layer_depth, \
                                 batch_first=True, \
                                 dropout=self.dropout)
        elif rnn_type == 'GRU':
            self.model = nn.GRU(input_size=self.number_of_features, \
                                hidden_size=self.hidden_size, \
                                num_layers=self.hidden_layer_depth, \
                                batch_first=True, \
                                dropout=self.dropout)
        else:
            raise NotImplementedError

    def forward(self, x, hidden):
        '''
        Forward propagation of encoder. Given input, outputs the last hidden state of encoder

        :param x: input to the encoder, of shape (batch_size, seq_len, feature_size)
        :return: last hidden state of encoder, of shape (batch_size, hidden_size)
        '''

        _, (h_end, c_end) = self.model(x, hidden)

        h_end = h_end[-1, :, :]

        return h_end


class Lambda(nn.Module):
    '''
    Lambda module converts output of encoder to latent vector

    :param hidden_size: hidden size of the encoder
    :param latent_length: latent vector length
    '''

    def __init__(self, hidden_size, latent_length):
        super(Lambda, self).__init__()

        self.hidden_size = hidden_size
        self.latent_length = latent_length

        self.hidden_to_mean = nn.Linear(self.hidden_size, self.latent_length)
        self.hidden_to_logvar = nn.Linear(self.hidden_size, self.latent_length)

        ## 가중치 초기화
        nn.init.xavier_uniform_(self.hidden_to_mean.weight)
        nn.init.xavier_uniform_(self.hidden_to_logvar.weight)

    def forward(self, x_encoded):
        '''
        Given last hidden state of encoder, passes through a linear layer, and finds the mean and variance

        :param cell_output: last hidden state of encoder
        :return: latent vector
        '''

        self.latent_mean = self.hidden_to_mean(x_encoded)
        self.latent_logvar = self.hidden_to_logvar(x_encoded)

        if self.training:
            std = torch.exp(0.5 * self.latent_logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(self.latent_mean)
        else:
            return self.latent_mean


class Decoder(nn.Module):
    '''
    Converts latent vector into output

    :param sequence_length: length of the input sequence
    :param batch_size: batch size of the input sequence
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param output_size: 2, one representing the mean, other log std dev of the output
    :param rnn_type: GRU/LSTM - use the same which you've used in the encoder
    :param dtype: Depending on cuda enabled/disabled, create the tensor
    '''

    def __init__(self, sequence_length, batch_size, hidden_size, hidden_layer_depth, latent_length, output_size, dtype,
                 rnn_type='LSTM'):

        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.output_size = output_size
        self.dtype = dtype

        if rnn_type == 'LSTM':
            self.model = nn.LSTM(input_size=1, \
                                 hidden_size=self.hidden_size, \
                                 num_layers=self.hidden_layer_depth, \
                                 batch_first=True)
        elif rnn_type == 'GRU':
            self.model = nn.GRU(input_size=1, \
                                hidden_size=self.hidden_size, \
                                num_layers=self.hidden_layer_depth, \
                                batch_first=True)
        else:
            raise NotImplementedError

        self.latent_to_hidden = nn.Linear(self.latent_length, self.hidden_size)
        self.hidden_to_output = nn.Linear(self.hidden_size, self.output_size)

        ##xavier: 각각의 layer 특성에 맞춰서 초기화를 진행함
        nn.init.xavier_uniform_(self.latent_to_hidden.weight)
        nn.init.xavier_uniform_(self.hidden_to_output.weight)

    def forward(self, latent, batch_size):
        '''
        Converts latent to hidden to output

        :param latent: latent vector
        :return: outputs consisting of mean and std dev of vector
        '''
        # self.hidden_size
        h_state = self.latent_to_hidden(latent)

        # initialize inputs for each batch
        decoder_inputs = torch.zeros(batch_size, self.sequence_length, 1, requires_grad=True).type(self.dtype).cuda()
        c_0 = torch.zeros(self.hidden_layer_depth, batch_size, self.hidden_size, requires_grad=True).type(
            self.dtype).cuda()

        # RUN RNN Decoder Model
        if isinstance(self.model, nn.LSTM):
            h_0 = torch.stack([h_state for _ in range(self.hidden_layer_depth)])
            decoder_output, _ = self.model(decoder_inputs, (h_0, c_0))
        elif isinstance(self.model, nn.GRU):
            h_0 = torch.stack([h_state for _ in range(self.hidden_layer_depth)])
            decoder_output, _ = self.model(decoder_inputs, h_0)
        else:
            raise NotImplementedError

        out = self.hidden_to_output(decoder_output)
        return out


def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"


