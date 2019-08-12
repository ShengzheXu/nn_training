"""A module for a mixture density network layer

For more info on MDNs, see _Mixture Desity Networks_ by Bishop, 1994.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import torch.nn.functional as F
import math
from scipy import integrate
import numpy as np


ONEOVERSQRT2PI = 1.0 / math.sqrt(2*math.pi)

class MDN(nn.Module):
    """A mixture density network layer

    The input maps to the parameters of a MoG probability distribution, where
    each Gaussian has O dimensions and diagonal covariance.

    Arguments:
        in_features (int): the number of dimensions in the input
        out_features (int): the number of dimensions in the output
        num_gaussians (int): the number of Gaussians per output dimensions

    Input:
        minibatch (BxD): B is the batch size and D is the number of input
            dimensions.

    Output:
        (pi, sigma, mu) (BxG, BxGxO, BxGxO): B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions for each
            Gaussian. Pi is a multinomial distribution of the Gaussians. Sigma
            is the standard deviation of each Gaussian. Mu is the mean of each
            Gaussian.
    """
    def __init__(self, in_height, emb_size, num_filters, out_features, num_gaussians):
        super(MDN, self).__init__()
        self.emb_size = emb_size
        self.in_height = in_height
        self.window_sizes = [1, 2, 3, 4, 5]
        self.num_filters = num_filters

        in_features = num_filters * len(self.window_sizes)
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        self.pi = nn.Sequential(
            nn.Linear(in_features, num_gaussians),
            nn.Softmax(dim=1)
        )
        self.sigma = nn.Sequential(
            nn.Linear(in_features, out_features*num_gaussians),
            # nn.Softmax(dim=1)
        )
        self.mu = nn.Sequential(
            nn.Linear(in_features, out_features*num_gaussians),
            # nn.Softmax(dim=1)
        )

        # conv part
        self.convs = nn.ModuleList([
            nn.Conv2d(1, self.num_filters, [h, self.emb_size], padding=(h - 1, 0))
            for h in self.window_sizes
        ])
        self.bn_input = nn.BatchNorm1d(self.in_height*self.emb_size, momentum=0.5)
        self.dropout = nn.Dropout(p=0.2)

        # self.conv1 = nn.Sequential(nn.Conv1d(10, 6, 3)
#         self.convs = nn.ModuleList([
#                 nn.Sequential(nn.Conv1d(in_channels=10, 
#                                         out_channels=2, 
#                                         kernel_size=h),
# #                              nn.BatchNorm1d(num_features=config.feature_size), 
#                               nn.ReLU(),
#                               nn.MaxPool1d(kernel_size=50-h+1))
#                      for h in self.window_sizes
#                     ])

    def forward(self, minibatch):
        # input batch * 50 * 10
        # batch_size x text_len x embedding_size  -> batch_size x embedding_size x text_len
        # print('forward1', minibatch.size())
        # minibatch = minibatch.permute(0, 2, 1)
        # minibatch = self.conv1(minibatch)
        # out = out.view(-1, out.size(1))
        
        # out = [conv(minibatch) for conv in self.convs]  #out[i]:batch_size x feature_size*1
        #for o in out:
        #    print('o',o.size())  # 32*100*1
        # out = torch.cat(out, dim=1)  # cat on 2nd dim，exo 5*2*1, 5*3*1 -> 5*5*1
        #print(out.size(1)) # 32*400*1
        # out = out.view(-1, out.size(1)) 
        # print('forward2', out.size())  # 32*400

        # ===============================
        # x = self.embedding(x)           # [B, T, E]
        # minibatch [B, 50, E]
        # Apply a convolution + max pool layer for each window size
        x = torch.unsqueeze(minibatch, 1)       # [B, C, T, E] Add a channel dim.
        xs = []
        for conv in self.convs:
            # x2 = F.relu(conv(x))        # [B, F, T, 1]
            x2 = F.tanh(conv(x))
            x2 = torch.squeeze(x2, -1)  # [B, F, T]
            x2 = F.max_pool1d(x2, x2.size(2))  # [B, F, 1]
            xs.append(x2)
        x = torch.cat(xs, 2)            # [B, F, window]

        # FC
        out = x.view(x.size(0), -1)       # [B, F * window]
        # logits = self.fc(x)             # [B, class]

        # Prediction
        # probs = F.softmax(logits)       # [B, class]
        # classes = torch.max(probs, 1)[1]# [B]
        # dropout
        out = self.dropout(out)
        # print('traing label:', self.training)
        
        # minibatch in_feature 160 * 50 
        pi = self.pi(out)
        sigma = torch.exp(self.sigma(out))
        sigma = sigma.view(-1, self.num_gaussians, self.out_features)
        mu = self.mu(out)
        mu = mu.view(-1, self.num_gaussians, self.out_features)
        return pi, sigma, mu

def pdf_func(sigma, mu, target):
    return ONEOVERSQRT2PI * torch.exp(-0.5 * ((target - mu) / sigma)**2) / sigma 

def gaussian_probability(sigma, mu, target):
    """Returns the probability of `data` given MoG parameters `sigma` and `mu`.
    
    Arguments:
        sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
            size, G is the number of Gaussians, and O is the number of
            dimensions per Gaussian.
        mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions per Gaussian.
        data (BxI): A batch of data. B is the batch size and I is the number of
            input dimensions.

    Returns:
        probabilities (BxG): The probability of each point in the probability
            of the distribution in the corresponding sigma/mu index.
    """
    target = target.unsqueeze(1).expand_as(sigma)
    ret = ONEOVERSQRT2PI * torch.exp(-0.5 * ((target - mu) / sigma)**2) / sigma
    # ret = integrate.quad(pdf_func, target, target+)
    # print('ret', ret.size())
    # print('2dret', torch.prod(ret, 2).size())
    # print(not torch.isnan(ret).any())
    if (torch.isnan(ret).any()):
        print('sigma', sigma)
        print('mu', mu)
        print('ret', ret)
        input()
    return torch.prod(ret, 2)


def mdn_loss(pi, sigma, mu, target):
    """Calculates the error, given the MoG parameters and the target

    The loss is the negative log likelihood of the data given the MoG
    parameters.
    """
    # print('pi', pi.size(), 'sigma', sigma.size(), 'target', target.size())
    prob = pi * gaussian_probability(sigma, mu, target)
    # print('prob', prob)
    nll = -torch.log(torch.sum(prob, dim=1)+1e-10)
    # prob = torch.log(prob)
    # print('prob', prob.size(), prob)
    # nll = -torch.logsumexp(prob, dim=1)
    # print('nll', nll.size(), nll) # print('nll', nll) # print('mean',torch.mean(nll))
    # if np.isnan(torch.mean(nll).data.numpy()):
    #     print('pi', pi)
    #     print('sigma', sigma)
    #     print('mu', mu)
    #     print('prob', prob)
    #     print(target)
    #     input()
    return torch.mean(nll)


def sample(pi, sigma, mu):
    """Draw samples from a MoG.
    """
    # print("sample: pi:", pi.size(), pi)
    categorical = Categorical(pi)
    pis = list(categorical.sample().data)
    sample = Variable(sigma.data.new(sigma.size(0), sigma.size(2)).normal_())
    for i, idx in enumerate(pis):
        sample[i] = sample[i].mul(sigma[i,idx]).add(mu[i,idx])
    return sample