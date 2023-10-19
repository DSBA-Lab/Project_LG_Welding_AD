import torch
import torch.nn as nn
import torch.optim as optim

import time
from torch.autograd import Variable
import torch.nn.init as init

from tqdm import tqdm    
import numpy as np

import pdb
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class LSTMGenerator(nn.Module):
    """An LSTM based generator. It expects a sequence of noise vectors as input.
    Args:
        in_dim: Input noise dimensionality
        out_dim: Output dimensionality
        n_layers: number of lstm layers
        hidden_dim: dimensionality of the hidden layer of lstms
    Input: noise of shape (batch_size, seq_len, in_dim)
    Output: sequence of shape (batch_size, seq_len, out_dim)
    """

    def __init__(self, in_dim, out_dim, device=None):
        super().__init__()
        self.out_dim = out_dim
        self.device = device

        self.lstm0 = nn.LSTM(in_dim, hidden_size=32, num_layers=1, batch_first=True)
        self.lstm1 = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True)

        self.linear = nn.Sequential(nn.Linear(in_features=128, out_features=out_dim), nn.Tanh())

        self.device = device

    def forward(self, input, h_0, c_0):
        batch_size, seq_len = input.size(0), input.size(1)

        recurrent_features, (h_1, c_1) = self.lstm0(input, (h_0, c_0))
        recurrent_features, (h_2, c_2) = self.lstm1(recurrent_features)
        recurrent_features, _ = self.lstm2(recurrent_features)

        outputs = self.linear(recurrent_features.contiguous().view(batch_size * seq_len, 128))
        outputs = outputs.view(batch_size, seq_len, self.out_dim)
        return outputs, recurrent_features


class LSTMDiscriminator(nn.Module):
    """An LSTM based discriminator. It expects a sequence as input and outputs a probability for each element.
    Args:
        in_dim: Input noise dimensionality
        n_layers: number of lstm layers
        hidden_dim: dimensionality of the hidden layer of lstms
    Inputs: sequence of shape (batch_size, seq_len, in_dim)
    Output: sequence of shape (batch_size, seq_len, 1)
    """

    def __init__(self, in_dim, device=None):
        super().__init__()
        self.device = device

        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=100, num_layers=1, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(100, 1), nn.Sigmoid())

    def forward(self, input, h_0, c_0):
        batch_size, seq_len = input.size(0), input.size(1)

        recurrent_features, _ = self.lstm(input, (h_0, c_0))
        outputs = self.linear(recurrent_features.contiguous().view(batch_size * seq_len, 100))
        outputs = outputs.view(batch_size, seq_len, 1)
        return outputs, recurrent_features

class TAnoGAN_MOCAP:
    def __init__(self, params):
        n_feats = params['feature_num']
        device = params['device']
        self.G = LSTMGenerator(n_feats, n_feats, device).to(device)
        self.D = LSTMDiscriminator(n_feats, device).to(device)
        self.lr = 0.0001
        self.beta1 = 0.5

        self.optimizerD = optim.Adam(self.D.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizerG = optim.Adam(self.G.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.seq_len = configs.seq_len
        self.feature_num = configs.feature_num
        self.hidden_size = configs.hidden_size

        self.configs = configs 
        self.device = torch.device('cuda:{}'.format(configs.gpu)) if configs.gpu else 'cpu'
        self.gen = LSTMGenerator(self.feature_num, self.feature_num, self.device)
        self.dis = LSTMDiscriminator(self.feature_num, self.device)

        self.lr = configs.learning_rate

    def fit(self, train_loader, train_epochs, model_optim, criterion, device, ckpt_path, early_stopping):
        time_now = time.time()
        train_steps = len(train_loader)
        optimizer_dis = model_optim(self.gen.parameters(), lr=self.lr)
        optimizer_gen = model_optim(self.dis.parameters(), lr=self.lr)

        for epoch in tqdm(range(train_epochs), desc='Epochs', position=0, leave=True):
            iter_count = 0
            train_loss = []

            self.train()
            epoch_time = time.time()
            self.dis.train()
            self.gen.train()

            for batch_idx, batch in tqdm(enumerate(train_loader), desc='Batches', position=0, leave=False,
                                    total=len(train_loader)):
                iter_count += 1
                batch_reshape = batch['given'].view([batch['given'].shape[0], self.seq_len, self.feature_num]) # (bsz, seq_len, nvars)
                batch=to_device(batch_reshape, device)

                batch_size, window_size, in_dim = batch.size(0), batch.size(1), batch.size(2)

                optimizer_dis.zero_grad()
                optimizer_gen.zero_grad()
                
                h_0 = torch.zeros(1, batch_size, 100).to(device)
                c_0 = torch.zeros(1, batch_size, 100).to(device)

                ### Step 1 : train discriminator ###
                label = torch.ones((batch_size, window_size, 1)).to(device)
                label_size = label.flatten().shape[0]

                output, _ = self.dis(batch, h_0, c_0)
                
                loss_D_real = torch.mean(criterion(output, label))
                loss_D_real.backward()

                optimizer_dis.step()

                hg_0 = torch.zeros(1, batch_size, 32).to(device)
                cg_0 = torch.zeros(1, batch_size, 32).to(device)

                noise = Variable(init.normal(torch.Tensor(batch_size,window_size,in_dim),mean=0,std=0.1)).to(device)
                fake, _ = self.gen(noise, hg_0, cg_0)

                output, _ = self.dis(fake.detach(), h_0, c_0)
                label = torch.zeros((batch_size, window_size, 1)).to(device)
        
                loss_d_fake = torch.mean(criterion(output, label))
                loss_d_fake.backward()

                loss_dis = loss_d_fake + loss_D_real

                optimizer_dis.step()

                ### Step 2 : train generator ###

                optimizer_gen.zero_grad()
                noise = Variable(init.normal(torch.Tensor(batch_size,window_size,in_dim),mean=0,std=0.1)).to(device)
                fake, _ = self.gen(noise, hg_0, cg_0)
                label = torch.ones((batch_size, window_size, 1)).to(device)

                output, _ = self.dis(fake, h_0, c_0)
                loss_g = torch.mean(criterion(output, label))
                loss_g.backward()

                optimizer_gen.step()


                if (batch_idx + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | Discriminator loss: {2:.7f}| Generator loss: {2:.7f}".format(batch_idx + 1, epoch + 1, loss_dis.item(), loss_g.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((train_epochs - epoch) * train_steps - batch_idx)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
            train_loss.append(loss_dis.item()+loss_g.item())

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(epoch + 1, train_steps, train_loss))
            early_stopping(train_loss, self, ckpt_path)

            if early_stopping.early_stop:
                print("Early stopping")
                break      

    def test(self, test_loader, criterion, device, alpha=.5, beta=.5):
        dist = []
        attack = []
        pred=[]

        self.eval()
        torch.backends.cudnn.enabled = False

        for batch_idx, batch in tqdm(enumerate(test_loader), desc='Batches', position=0, leave=True, total=len(test_loader)):

            batch_reshape = batch['given'].view([batch['given'].shape[0], self.seq_len, self.feature_num]) # (bsz, seq_len, nvars)
            batch_x=to_device(batch_reshape,device)

            batch_size, window_size, in_dim = batch_x.size(0), batch_x.size(1), batch_x.size(2)
            
            z = Variable(torch.randn((batch_size, window_size, in_dim), device=device), requires_grad=True)

            # z = Variable(init.normal(torch.zeros(batch_size, window_size, in_dim, device=device),mean = 0, std = 0.1),
            #             requires_grad = True)

            z_optimizer =  torch.optim.Adam([z], lr = 0.01)

            for iter in range(self.configs.iterations):

                h_0 = torch.zeros(1, batch_size, 100).to(device)
                c_0 = torch.zeros(1, batch_size, 100).to(device)
                hg_0 = torch.zeros(1, batch_size, 32).to(device)
                cg_0 = torch.zeros(1, batch_size, 32).to(device)

                fake, _ = self.gen(z, hg_0, cg_0)
                _, x_feature = self.dis(batch_x, h_0, c_0) 
                _, G_z_feature = self.dis(fake, h_0, c_0)

                residual_loss = torch.mean(torch.abs(batch_x-fake), axis = 2) # Residual Loss
                discrimination_loss = torch.mean(torch.abs(x_feature-G_z_feature), axis=2) # Discrimination loss
                loss = (1-self.configs.Lambda)*residual_loss + self.configs.Lambda*discrimination_loss
                #loss = self.Anomaly_score(batch_x, fake, x_feature, G_z_feature, self.configs.Lambda)
                z_optimizer.zero_grad()
                torch.mean(loss).backward()
                z_optimizer.step()

            dist.append(loss.detach().cpu().numpy())
            attack.append(batch['attack'].squeeze().numpy())
                
        dist = np.concatenate(dist).flatten()
        attack = np.concatenate(attack).flatten()
        # pred = np.concatenate(pred)
        scores = dist.copy()

        return scores, attack, pred 


    def Anomaly_score(latent, fake, latent_interm, fake_interm, Lambda_value):
        residual_loss = torch.mean(torch.abs(latent-fake), axis = 2) # Residual Loss
        
        discrimination_loss = torch.mean(torch.abs(latent_interm-fake_interm), axis=2) # Discrimination loss

        total_loss = (1-Lambda_value)*residual_loss + Lambda_value*discrimination_loss
        return total_loss