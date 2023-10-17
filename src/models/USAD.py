import torch
import torch.nn as nn
from torch import optim
import numpy as np

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Model(nn.Module):
  def __init__(self, configs):
    super().__init__()
    self.z_size = configs.window_size * configs.hidden_size# 기존에 4 # window size * feature 수
    self.w_size = configs.window_size * configs.feature_num 
    self.encoder = Encoder(self.w_size, self.z_size)
    self.decoder1 = Decoder(self.z_size, self.w_size)
    self.decoder2 = Decoder(self.z_size, self.w_size)
 
  # def evaluate(model, val_loader, n, w_size):
  #   # outputs = [model.validation_step(torch.tensor(to_device(batch['given'].view([batch['given'].shape[0],w_size]),device)), n) for batch in val_loader]
  #   outputs = [model.validation_step(to_device(batch['given'].view([batch['given'].shape[0],w_size]),device), n) for batch in val_loader]
  #   return model.validation_epoch_end(outputs)

  def _select_optimizer(self):
      if self.optim == 'adamw':
          model_optim = optim.AdamW
      elif self.optim == 'adam':
          model_optim = optim.Adam
      elif self.optim == 'sgd':
          model_optim = optim.SGD
      return model_optim

  def fit(self, train_loader, train_epochs):
   
    # history = []
    model_optim = self._select_optimizer()
    optimizer1 = model_optim(list(self.encoder.parameters())+list(self.decoder1.parameters()))
    optimizer2 = model_optim(list(self.encoder.parameters())+list(self.decoder2.parameters()))
    for epoch in range(train_epochs):
        for batch in train_loader:
          
          batch_reshape = batch['given'].view([batch['given'].shape[0],self.w_size]) # (bsz, w_size)
          batch=to_device(batch_reshape, device)
          # batch = torch.tensor(batch)
            
          #Train AE1
          # loss1,loss2 = self.training_step(batch,epoch+1)
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
          # loss1,loss2 = self.training_step(batch,epoch+1)
          z = self.encoder(batch)
          w1 = self.decoder1(z)
          w2 = self.decoder2(z)
          w3 = self.decoder2(self.encoder(w1))
          # loss1 = 1/(epoch+1)*torch.mean((batch-w1)**2)+(1-1/(epoch+1))*torch.mean((batch-w3)**2)
          loss2 = 1/(epoch+1)*torch.mean((batch-w2)**2)-(1-1/(epoch+1))*torch.mean((batch-w3)**2)
          loss2.backward()
          optimizer2.step()
          optimizer2.zero_grad()
            
        # result = evaluate(self, val_loader, epoch+1, w_size)
        # self.epoch_end(epoch, result)
        # history.append(result)

    return # history
  
  
  def test(self, test_loader, alpha=.5, beta=.5):
    results=[]
    for batch in test_loader:
      batch_reshape = batch['given'].view([batch['given'].shape[0],self.w_size])
      batch=to_device(batch_reshape,device)
      # batch = torch.tensor(batch)
      w1=self.decoder1(self.encoder(batch))
      w2=self.decoder2(self.encoder(w1))
      results.append(alpha*torch.mean((batch-w1)**2,axis=1)+beta*torch.mean((batch-w2)**2,axis=1))
      y_pred=np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
                                    results[-1].flatten().detach().cpu().numpy()])
    return y_pred
  
  # def epoch_end(self, epoch, result):
  #   print("Epoch [{}], val_loss1: {:.4f}, val_loss2: {:.4f}".format(epoch, result['val_loss1'], result['val_loss2']))
    

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
    