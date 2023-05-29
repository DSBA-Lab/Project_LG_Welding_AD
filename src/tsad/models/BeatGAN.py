import os,pickle,time,sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def weights_init(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        # mod.weight.data.normal_(0.0, 0.02)
        nn.init.xavier_normal_(mod.weight.data)
        # nn.init.kaiming_uniform_(mod.weight.data)

    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)
    elif classname.find('Linear') !=-1 :
        torch.nn.init.xavier_uniform(mod.weight)
        mod.bias.data.fill_(0.01)

class Generator(nn.Module):
    def __init__(self, nc):
        super(Generator, self).__init__()

        self.encoder = nn.Sequential(
            # input is (nc) x 64
            nn.Linear(nc * 25, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 32),
            nn.Tanh(),
            nn.Linear(32, 10),

        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 32),
            nn.Tanh(),
            nn.Linear(32, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, nc * 25),
            nn.Tanh(),
        )

    def forward(self, input):
        input=input.view(input.shape[0],-1)
        z = self.encoder(input)
        output = self.decoder(z)
        output=output.view(output.shape[0],100,-1)
        return output

class Discriminator(nn.Module):
    def __init__(self, nc):
        super(Discriminator, self).__init__()

        self.features = nn.Sequential(
            # input is (nc) x 64
            nn.Linear(nc * 25 , 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 32),
            nn.Tanh(),

        )

        self.classifier=nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        input=input.view(input.shape[0],-1)
        features = self.features(input)
        # features = self.feat(features.view(features.shape[0],-1))
        # features=features.view(out_features.shape[0],-1)
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)

        return classifier, features


class BeatGAN_MOCAP:
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

        self.batchsize = 64
        self.nz = 10
        self.niter = 300
        self.nc= 100
        self.lr=0.0001
        self.beta1=0.5


        self.G = Generator(self.nc).to(device)
        self.G.apply(weights_init)


        self.D = Discriminator(self.nc).to(device)
        self.D.apply(weights_init)

        # Loss Functions
        self.bce_criterion = nn.BCELoss()
        self.l1_criterion=nn.L1Loss()
        self.mse_criterion=nn.MSELoss()

        self.optimizerD = optim.Adam(self.D.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizerG = optim.Adam(self.G.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        self.total_steps = 0
        self.cur_epoch = 0


        self.real_label = 1
        self.fake_label = 0

        # -- Discriminator attributes.
        self.err_d_real = None
        self.err_d_fake = None

        # -- Generator attributes.
        self.err_g = None


    def train_epoch(self):
        self.G.train()
        self.D.train()

        for train_x,train_y in self.dataloader["train"]:
            train_x=train_x.to(self.device)
            train_y=train_y.to(self.device)
            for i in range(1):
                self.D.zero_grad()
                batch_size=train_x.shape[0]
                self.y_real_, self.y_fake_ = torch.ones(batch_size).to(self.device), torch.zeros(batch_size).to(
                    self.device)
                # Train with real
                out_d_real, _ = self.D(train_x)
                # Train with fake
                fake = self.G(train_x)
                out_d_fake, _ = self.D(fake)
                # --

                self.err_d_real = self.bce_criterion(out_d_real,self.y_real_)
                self.err_d_fake = self.bce_criterion(out_d_fake,self.y_fake_)

                self.err_d = self.err_d_real + self.err_d_fake
                self.err_d.backward(retain_graph=True)
                self.optimizerD.step()


            self.G.zero_grad()

            _, feat_fake = self.D(fake)
            _, feat_real = self.D(train_x)


            self.err_g_adv = self.mse_criterion(feat_fake, feat_real)  # loss for feature matching
            self.err_g_rec = self.mse_criterion(fake, train_x)  # constrain x' to look like x

            self.err_g = self.err_g_rec+0.01*self.err_g_adv
            self.err_g.backward()
            self.optimizerG.step()


    def predict(self,dataloader_,scale=True):
        with torch.no_grad():
            # Create big error tensor for the test set.
            test_pair=[]


            self.an_scores = torch.zeros(size=(len(dataloader_.dataset),), dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(size=(len(dataloader_.dataset),), dtype=torch.long,    device=self.device)
            # self.dis_feat = torch.zeros(size=(len(dataloader_.dataset), self.opt.ndf*16*10), dtype=torch.float32,
            #                             device=self.device)


            for i, data in enumerate(dataloader_, 0):
                test_x=data[0].to(self.device)
                test_y=data[1].to(self.device)

                fake = self.G(test_x)



                error = torch.mean(
                    torch.pow((test_x.view(test_x.shape[0], -1) - fake.view(fake.shape[0], -1)), 2),
                    dim=1)

                self.an_scores[i*self.batchsize : i*self.batchsize+error.size(0)] = error.reshape(error.size(0))
                self.gt_labels[i*self.batchsize : i*self.batchsize+error.size(0)] = test_y.reshape(error.size(0))
                # self.dis_feat[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0), :] = d_feat.reshape(
                #     error.size(0), self.opt.ndf*16*10)


            # Scale error vector between [0, 1]
            if scale:
                self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (torch.max(self.an_scores) - torch.min(self.an_scores))

            y_=self.gt_labels.cpu().numpy()
            y_pred=self.an_scores.cpu().numpy()
            # print(y_pred)

            return y_,y_pred