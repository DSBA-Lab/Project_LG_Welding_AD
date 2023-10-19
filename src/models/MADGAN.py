import torch
import torch.nn as nn 
import time 
import datetime
import numpy as np 
import os 
import sys
import pandas as pd
from sklearn.base import BaseEstimator
from tqdm import tqdm
from tqdm.auto import trange

import pdb
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


class Model(BaseEstimator, nn.Module):
    '''
    Model Training and Validation
    '''       
    def __init__(
        self, 
        configs
    ):
        super(Model, self).__init__()

        self.sequence_length = configs.seq_len  # 100
        # torch.Size([32, 100, 25]) : batch크기, window 크기, feature_num 수
        self.feature_num = configs.feature_num
        self.hidden_size = configs.hidden_size
        self.hidden_layer_depth = configs.num_layers
        self.latent_length = configs.latent_length
        self.batch_size = configs.batch_size
        self.rnn_type = configs.rnn_type
        self.dropout_rate = configs.dropout
        self.dtype = torch.FloatTensor
        self.lr = configs.learning_rate
        
        # madgan 추가 config
        
        self.lam = configs.lam
        self.z_optim_iter = configs.z_optim_iter
        self.dr_loss_method = configs.dr_loss_method
        
    
    
    def fit(self, train_loader, train_epochs, model_optim, criterion, device, ckpt_path, early_stopping):
        self.G = LSTMGenerator(nb_features = len(train_loader.dataset[0]['given'][0]), device = device)
        self.D = LSTMDiscriminator(nb_features = len(train_loader.dataset[0]['given'][0]), device = device)
        
        # optimizer
        self.gen_optim = torch.optim.Adam(self.G.parameters(), lr=0.1)
        self.dis_optim = torch.optim.SGD(self.D.parameters(), lr=0.1) 

        ## scheduler
        self.gen_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.gen_optim, mode='min', factor=0.9, patience=10, verbose=True)
        self.dis_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.dis_optim, mode='min', factor=0.9, patience=10, verbose=True)

        
        def train(self):
            self.D.train()
            self.G.train()
            train_loss = []
            train_dis_loss = []
            train_dis_real_loss = []
            train_dis_fake_loss = []
            train_gen_loss = [] 

            # TODO gen, dis scheduler
            gen_max = 3 
            gen_cnt = 0 
            
            for batch_idx, inputs in enumerate(train_loader):
                inputs = inputs['given']
                inputs = inputs.to(device)
                noise = torch.autograd.Variable(torch.randn(size=inputs.size())).to(self.device)

                # Generator
                # set gen optimizer init
                self.gen_optim.zero_grad()

                # generate
                fakes = self.G(noise)

                # gen loss
                gen_loss = criterion(fakes, inputs)

                # gen update
                # gen_loss.backward()
                gen_loss.mean().backward()
                self.gen_optim.step()

                # Discriminator
                # set dis optimizer init
                self.dis_optim.zero_grad()

                # generate fake inputs
                inputs_fake = self.G(noise)

                # discriminate
                dis_real_outputs = self.D(inputs)
                dis_fake_outputs = self.D(inputs_fake)

                # dis loss
                dis_real_loss = criterion(dis_real_outputs, torch.full(dis_real_outputs.size(), 1, dtype=torch.float, device = device))
                dis_fake_loss = criterion(dis_fake_outputs, torch.full(dis_fake_outputs.size(), 0, dtype=torch.float, device = device))
                dis_loss = dis_real_loss + dis_fake_loss 

                # dis update
                dis_loss.mean().backward()
                self.dis_optim.step()

                # Loss
                train_dis_loss.append(dis_loss.mean().item())
                train_dis_real_loss.append(dis_real_loss.mean().item())
                train_dis_fake_loss.append(dis_fake_loss.mean().item())
                train_gen_loss.append(gen_loss.mean().item())
                loss = dis_loss + gen_loss
                train_loss.append(loss.mean().item())

            train_dis_loss = np.mean(train_dis_loss)
            train_dis_real_loss = np.mean(train_dis_real_loss)
            train_dis_fake_loss = np.mean(train_dis_fake_loss)
            train_gen_loss = np.mean(train_gen_loss)
            train_loss = np.mean(train_loss)

            return train_dis_loss, train_dis_real_loss, train_dis_fake_loss, train_gen_loss, train_loss    
        
        def validation(self):
            self.D.eval()
            self.G.eval()
            val_loss = []
            val_dis_loss = []
            val_dis_real_loss = []
            val_dis_fake_loss = []
            val_gen_loss = [] 

            with torch.no_grad():
                for batch_idx, inputs in enumerate(self.validloader):
                    inputs = inputs['given']
                    inputs = inputs.to(self.device)
                    noise = torch.autograd.Variable(torch.randn(size=inputs.size())).to(self.device)

                    # Generator 
                    gen_outputs = self.G(noise).detach() 

                    # gen loss
                    gen_loss = criterion(gen_outputs, inputs)
                    
                    # Discriminator
                    dis_real_outputs = self.D(inputs).detach()
                    dis_fake_outputs = self.D(gen_outputs).detach()

                    # dis loss
                    dis_real_loss = criterion(dis_real_outputs, torch.full(dis_real_outputs.size(), 1, dtype=torch.float, device=self.device))
                    dis_fake_loss = criterion(dis_fake_outputs, torch.full(dis_fake_outputs.size(), 0, dtype=torch.float, device=self.device))
                    dis_loss = dis_real_loss + dis_fake_loss 
                    
                    # loss
                    val_dis_loss.append(dis_loss.mean().item())
                    val_dis_real_loss.append(dis_real_loss.mean().item())
                    val_dis_fake_loss.append(dis_fake_loss.mean().item())
                    val_gen_loss.append(gen_loss.mean().item())
                    loss = gen_loss + dis_loss 
                    val_loss.append(loss.mean().item())


            val_dis_loss = np.mean(val_dis_loss)
            val_dis_real_loss = np.mean(val_dis_real_loss)
            val_dis_fake_loss = np.mean(val_dis_fake_loss)
            val_gen_loss = np.mean(val_gen_loss)
            val_loss = np.mean(val_loss)

            return val_dis_loss, val_dis_real_loss, val_dis_fake_loss, val_gen_loss, val_loss
        

        self.trainloader = train_loader
        self.validloader = None
        self.device = device
        self.history = {}

    
        # Training time check
        total_start = time.time()

        # Initialize history list
        train_loss_lst, train_gen_loss_lst, train_dis_loss_lst, train_dis_real_loss_lst, train_dis_fake_loss_lst = [], [], [], [], []
        
        if self.validloader:
            val_loss_lst, val_gen_loss_lst, val_dis_loss_lst, val_dis_real_loss_lst, val_dis_fake_loss_lst = [], [], [], [], []
            
        epoch_time_lst = []

        start_epoch = 0
        end_epoch = start_epoch + train_epochs
        for i in range(start_epoch, end_epoch):
            print('Epoch: ',i+1)

            epoch_start = time.time()
            train_dis_loss, train_dis_real_loss, train_dis_fake_loss, train_gen_loss, train_loss = train(self)
            if self.validloader:
                val_dis_loss, val_dis_real_loss, val_dis_fake_loss, val_gen_loss, val_loss = validation(self)        

            # scheduler
            if self.gen_scheduler!=None:
                self.gen_scheduler.step(train_gen_loss)
                # G_last_lr = self.gen_scheduler.state_dict()['_last_lr'][0]
            else:
                pass
                # G_last_lr = self.gen_optim.state_dict()['param_groups'][0]['lr']

            if self.dis_scheduler!=None:
                self.dis_scheduler.step(train_dis_loss)
                # D_last_lr = self.dis_scheduler.state_dict()['_last_lr'][0]
            else:
                pass
                # D_last_lr = self.dis_optim.state_dict()['param_groups'][0]['lr']


            # Save history
            train_loss_lst.append(train_loss)
            train_gen_loss_lst.append(train_gen_loss)
            train_dis_loss_lst.append(train_dis_loss)
            train_dis_real_loss_lst.append(train_dis_real_loss)
            train_dis_fake_loss_lst.append(train_dis_fake_loss)
            
            if self.validloader:
                val_loss_lst.append(val_loss)
                val_gen_loss_lst.append(val_gen_loss)
                val_dis_loss_lst.append(val_dis_loss)
                val_dis_real_loss_lst.append(val_dis_real_loss)
                val_dis_fake_loss_lst.append(val_dis_fake_loss)
            
            
            end = time.time()
            epoch_time = datetime.timedelta(seconds=end - epoch_start)
            epoch_time_lst.append(str(epoch_time))
            
            if self.validloader:
                print(f"Epoch: {i+1}. train_loss: {train_loss}, val_loss: {val_loss}")
            
            else:
                print(f"Epoch: {i+1}. train_loss: {train_loss}")
                early_stopping(train_loss, self, ckpt_path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                
        best_model_path = ckpt_path + '/' + 'checkpoint.pth'
        torch.save(Model.state_dict(self), best_model_path)
        self.load_state_dict(torch.load(best_model_path))

        end = time.time() - total_start
        total_time = datetime.timedelta(seconds=end)
        print('\nFinish Train: Training Time: {}\n'.format(total_time))

        # # Make history 
        # self.history = {}
        # self.history['train'] = []
        # self.history['train'].append({
        #     'loss':train_loss_lst,
        #     'gen_loss':train_gen_loss_lst,
        #     'dis_loss':train_dis_loss_lst,
        #     'dis_real_loss':train_dis_real_loss_lst,
        #     'dis_fake_loss':train_dis_fake_loss_lst
        # })
        
        # if self.validloader:
        #     self.history['validation'] = []
        #     self.history['validation'].append({
        #         'loss':val_loss_lst,
        #         'gen_loss':val_gen_loss_lst,
        #         'dis_loss':val_dis_loss_lst,
        #         'dis_real_loss':val_dis_real_loss_lst,
        #         'dis_fake_loss':val_dis_fake_loss_lst
        #     })

        # self.history['time'] = []
        # self.history['time'].append({
        #     'epoch':epoch_time_lst,
        #     'total':str(total_time)
        # })

    
    def test(self, test_loader, criterion, device):
        
        # test(self, testloader ,lam, optim_iter, method, real_time=False)
        """
        Arguments
        ---------
        - lam: lambda for DR score
        - optim_iter: iteration to obtain optimal Z
        - datadir: data directory
        - method: paper or approximation -> dr loss를 구하는 과정을 paper의 방법과 그대로 할지, 아니면 approximation을 통해 구할지
        """
        self.G.train()
        self.D.eval()

        batch_size = test_loader.batch_size
        window_size = test_loader.dataset.window_size
        slide_size = test_loader.dataset.slide_size
        dr_loss_arr = np.zeros(len(test_loader.dataset))
        dr_loss_list = []
        attack = []
        pred = []
        
        for batch_idx, batch in tqdm(enumerate(test_loader), desc='Batches', position=0, leave=True,
                                         total=len(test_loader)):
            batch_x = batch['given'].float().to(device)
            attack.append(batch['attack'].squeeze().numpy())
        
        attack = np.concatenate(attack).flatten()
        
        for batch_idx, inputs in enumerate(test_loader):
            inputs = inputs['given']
            inputs = inputs.to(device)

            # DR Score
            dr_loss = self.ds_score(inputs = inputs, criterion = criterion)
            dr_loss_list.append(dr_loss)

        
        # dr_loss_arr = np.zeros(len(test_loader.dataset))
        dr_loss_arr = []
        # method -> dr_loss_method 로 변경 필요
        if self.dr_loss_method == 'approximation':
            # method 1 : 근사법
            # 각 윈도우별 dr_score를 확인해본 결과 같은 timestamp에 대한 data면 dr_score의 차이가 크게 없음
            # 따라서 누적으로 window에 dr_score를 추가하는 방식으로 최종 dr_score를 산출
            # 각 window의 value를 dr_score에 추가
            
            
            index = 0
            for batch_idx in range(len(test_loader)):
                for window in dr_loss_list[batch_idx]:
                    dr_loss_arr.append(np.array(window))
                    index +=1
                print(index)
                
            dr_loss_arr = np.concatenate(dr_loss_arr).flatten()

                        
        elif self.dr_loss_method == 'paper':
            
            # method 2 : Paper에 나와있는 방법
            # 모든 window 내의 data들에 대해 시점을 파악하고, 같은 시점에 대한 dr_score들을 모아 묶어서 평균을 취한 값을 
            # 해당 timestamp의 최종 dr_score로 산출
            # 시간이 엄청 오래 걸림 2080ti 기준 test data 80,000개 기준 60,000초 정도...

            dr_score_list = []

            for i in range(len(test_loader.dataset)):
                print(i)
                dr_score = 0
                lct = 0
                for batch_idx in range(len(test_loader)):
                    
                    if batch_idx == len(test_loader)-1:
                        this_batch_size = len(test_loader.dataset) - batch_size*(len(test_loader)-1)
                        
                    else:
                        this_batch_size = batch_size
                        
                    for j in range(this_batch_size):
                        for s in range(window_size):
                            order = (batch_size * batch_idx + j) * slide_size + s
                            if order == i:
                                dr_score = dr_score + dr_loss_list[batch_idx][j][s]
                                lct = lct +1
                
                dr_score_list.append(dr_score/lct)


            dr_loss_arr = np.array(dr_score_list)
        
        else:
            # 잘못된 method 입력시 오류 출력
            
            print("Invalid method. Your input should be paper or approximation")
                    

        return dr_loss_arr, attack, pred
        
    def ds_score(self, inputs, criterion):
        """
        Arguments
        ---------
        - inputs: input data (batch size x sequence length X the number of features)
        - lam: lambda for DR score
        - iterations: iteration to obtain optimal Z
        
        return
        ---------
        dr_score (float)
        """
        
        # self.lam = configs.lam
        # self.z_optim_iter = configs.z_optim_iter
        # self.dr_loss_method = configs.dr_loss_method
        
        
        # get optimal Z
        optimal_z = self.optim_z(inputs, iterations = self.z_optim_iter, criterion = criterion)
        
        # Generator 
        fake = self.G(optimal_z.to(self.device))
        gen_loss = criterion(fake, inputs)
        gen_loss = gen_loss.mean(dim=-1).detach().cpu().numpy()

        # Discriminator
        dis_loss = self.D(inputs)
        dis_loss = dis_loss.mean(dim=-1).detach().cpu().numpy()

        dr_loss = self.lam * gen_loss + (1 - self.lam) * dis_loss

        return dr_loss


    def optim_z(self, inputs, iterations, criterion):
        """
        Arguments
        ---------
        - inputs: input data (batch size x sequence length X the number of features)
        - iterations: iteration to obtain optimal Z
        """
        criterion.reduction = 'mean'

        best_loss = np.inf
        optimal_z = 0
        
        noise = torch.autograd.Variable(torch.nn.init.normal_(torch.zeros(inputs.size()),mean=0,std=0.1),
                                        requires_grad=True)
        z_optimizer = torch.optim.Adam([noise], lr=0.01)
        
        for i in range(iterations):
            gen_outputs = self.G(noise.to(self.device))
            gen_loss = criterion(gen_outputs, inputs)
            gen_loss.backward()
            
            z_optimizer.step()
            
            if gen_loss.item() < best_loss:
                best_loss = gen_loss.item()
                optimal_z = noise
        
        criterion.reduction = 'none'

        return optimal_z
        
class LSTMGenerator(nn.Module):
    """
    LSTM based generator. 
    """

    def __init__(self, nb_features, device = None):
        """
        Arguments
        ---------

        nb_features: the number of features
        """
        super().__init__()
        self.device = device
        self.nb_features = nb_features
        self.hidden_size = 100 # MAD-GAN uses 100 hidden size
        self.lstm_depth = 3 # MAD-GAN uses 3 layers in LSTM

        self.lstm = nn.LSTM(input_size  = self.nb_features, 
                            hidden_size = self.hidden_size, 
                            num_layers  = self.lstm_depth, 
                            batch_first = True).to(self.device)
        
        self.linear = nn.Sequential(
            nn.Linear(in_features  = self.hidden_size, 
                      out_features = self.nb_features), 
            nn.Tanh()
        ).to(self.device)

    def forward(self, data):
        batch_size, seq_len, _ = data.size()
        
        outputs, _ = self.lstm(data)
        
        outputs = self.linear(outputs.contiguous().view(batch_size*seq_len, self.hidden_size))
        outputs = outputs.view(batch_size, seq_len, -1)
        return outputs


class LSTMDiscriminator(nn.Module):
    """
    LSTM based discriminator
    """

    def __init__(self, nb_features, device=None):
        super().__init__()
        self.nb_features = nb_features
        self.device = device
        self.hidden_size = 100 # MAD-GAN uses 100 hidden size
        
        self.lstm = nn.LSTM(input_size  = self.nb_features, 
                            hidden_size = self.hidden_size, 
                            num_layers  = 1, 
                            batch_first = True).to(self.device)

        self.linear = nn.Sequential(
            nn.Linear(in_features  = self.hidden_size, 
                      out_features = 1), 
            nn.Sigmoid()
        ).to(self.device)

    def forward(self, data):
        batch_size, seq_len, _ = data.size()

        recurrent_features, _ = self.lstm(data)
        outputs = self.linear(recurrent_features.contiguous().view(batch_size*seq_len, self.hidden_size))
        outputs = outputs.view(batch_size, seq_len, -1)
        return outputs
