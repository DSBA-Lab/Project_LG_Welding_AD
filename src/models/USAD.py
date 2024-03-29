import torch
import torch.nn as nn
from torch import optim
import numpy as np
from tqdm import tqdm
import time

class Model(nn.Module):
  def __init__(self, configs):
    super().__init__()
    # 기존에 4 # window size * feature 수
    self.seq_len = configs.seq_len
    self.feature_num = configs.feature_num
    self.hidden_size = configs.hidden_size
    self.w_size = self.seq_len * self.feature_num 
    self.z_size = self.seq_len * self.hidden_size
    self.encoder = Encoder(self.w_size, self.z_size)
    self.decoder1 = Decoder(self.z_size, self.w_size)
    self.decoder2 = Decoder(self.z_size, self.w_size)
    self.lr = configs.learning_rate


  def fit(self, train_loader, train_epochs, model_optim, criterion, device, ckpt_path, early_stopping):
    time_now = time.time()
    train_steps = len(train_loader)
    optimizer1 = model_optim(list(self.encoder.parameters())+list(self.decoder1.parameters()), lr=self.lr)
    optimizer2 = model_optim(list(self.encoder.parameters())+list(self.decoder2.parameters()), lr=self.lr)
    for epoch in tqdm(range(train_epochs), desc='Epochs', position=0, leave=True):
      iter_count = 0
      train_loss = []

      self.train()
      epoch_time = time.time()
      for batch_idx, batch in tqdm(enumerate(train_loader), desc='Batches', position=0, leave=False,
                                        total=len(train_loader)):
        iter_count += 1
        batch_reshape = batch['given'].view([batch['given'].shape[0],self.w_size]) # (bsz, w_size)
        batch=to_device(batch_reshape, device)
        # batch = torch.tensor(batch)
          
        #Train AE1
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        # w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        loss1 = 1/(epoch+1)*torch.mean((batch-w1)**2)+(1-1/(epoch+1))*torch.mean((batch-w3)**2)
        # loss2 = 1/(epoch+1)*torch.mean((batch-w2)**2)-(1-1/(epoch+1))*torch.mean((batch-w3)**2)
        loss1.backward()
        optimizer1.step()
        optimizer1.zero_grad()
          
        #Train AE2
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        # loss1 = 1/(epoch+1)*torch.mean((batch-w1)**2)+(1-1/(epoch+1))*torch.mean((batch-w3)**2)
        loss2 = 1/(epoch+1)*torch.mean((batch-w2)**2)-(1-1/(epoch+1))*torch.mean((batch-w3)**2)
        loss2.backward()
        optimizer2.step()
        optimizer2.zero_grad()

        if (batch_idx + 1) % 100 == 0:
          print("\titers: {0}, epoch: {1} | loss1: {2:.7f}| loss2: {2:.7f}".format(batch_idx + 1, epoch + 1, loss1.item(), loss2.item()))
          speed = (time.time() - time_now) / iter_count
          left_time = speed * ((train_epochs - epoch) * train_steps - batch_idx)
          print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
          iter_count = 0
          time_now = time.time()

      train_loss.append(loss1.item()+loss2.item())
      print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
      train_loss = np.average(train_loss)
      print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(epoch + 1, train_steps, train_loss))
      early_stopping(train_loss, self, ckpt_path)
      if early_stopping.early_stop:
        print("Early stopping")
        break      
        
    best_model_path = ckpt_path + '/' + 'checkpoint.pth'
    self.load_state_dict(torch.load(best_model_path))

    return 
  
  
  def test(self, test_loader, criterion, device, alpha=.5, beta=.5):
    dist = []
    attack = []
    pred=[]
    self.eval()
    with torch.no_grad():
      for batch_idx, batch in tqdm(enumerate(test_loader), desc='Batches', position=0, leave=True,
                                          total=len(test_loader)):
                           
        batch_reshape = batch['given'].view([batch['given'].shape[0],self.w_size])
        batch_x=to_device(batch_reshape,device)
        w1=self.decoder1(self.encoder(batch_x))
        w2=self.decoder2(self.encoder(w1))

        # 다시 input된 timestamp 단위의 형태로 변환
        w1=w1.view(batch_x.shape[0], self.seq_len, self.feature_num)
        w2=w2.view(batch_x.shape[0], self.seq_len, self.feature_num)
        batch_x=batch_x.view(batch_x.shape[0], self.seq_len, self.feature_num)
        # pred.append([w1.cpu().detach().numpy(), w2.cpu().detach().numpy()])

        # criterion(MSE)에 차원을 유지하는 option을 가해 각 timestamp별로 score를 구함
        score = alpha * criterion(w1, batch_x) + beta * criterion(w2, batch_x)
        dist.append(np.mean(score.detach().cpu().numpy(), axis=2))

        attack.append(batch['attack'].squeeze().numpy())
    dist = np.concatenate(dist).flatten()
    attack = np.concatenate(attack).flatten()
    # pred = np.concatenate(pred)
    scores = dist.copy()

    return scores, attack, pred 
  

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class Encoder(nn.Module):
  def __init__(self, in_size, latent_size):
    super().__init__()
    self.linear1 = nn.Linear(in_size, int(in_size/2))
    self.linear2 = nn.Linear(int(in_size/2), int(in_size/4))
    self.linear3 = nn.Linear(int(in_size/4), latent_size)
    self.relu = nn.ReLU(True)
        
  def forward(self, w):
    out = self.linear1(w)
    out = self.relu(out)
    out = self.linear2(out)
    out = self.relu(out)
    out = self.linear3(out)
    z = self.relu(out)
    return z
    
class Decoder(nn.Module):
  def __init__(self, latent_size, out_size):
    super().__init__()
    self.linear1 = nn.Linear(latent_size, int(out_size/4))
    self.linear2 = nn.Linear(int(out_size/4), int(out_size/2))
    self.linear3 = nn.Linear(int(out_size/2), out_size)
    self.relu = nn.ReLU(True)
    self.sigmoid = nn.Sigmoid()
        
  def forward(self, z):
    out = self.linear1(z)
    out = self.relu(out)
    out = self.linear2(out)
    out = self.relu(out)
    out = self.linear3(out)
    w = self.sigmoid(out)
    return w
    