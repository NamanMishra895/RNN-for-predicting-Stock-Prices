# -*- coding: utf-8 -*-
import torch
import math
from torch import nn
from torch.nn import functional as F
from torch.nn import init, Parameter
from torch.autograd import Variable


class NoisyLinear(nn.Linear):
  def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
    super(NoisyLinear, self).__init__(in_features, out_features, bias=True)  # TODO: Adapt for no bias
    # µ^w and µ^b reuse self.weight and self.bias
    self.sigma_init = sigma_init
    self.sigma_weight = Parameter(torch.Tensor(out_features, in_features))  # σ^w
    self.sigma_bias = Parameter(torch.Tensor(out_features))  # σ^b
    self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
    self.register_buffer('epsilon_bias', torch.zeros(out_features))
    self.reset_parameters()

  def reset_parameters(self):
    if hasattr(self, 'sigma_weight'):  # Only init after all params added (otherwise super().__init__() fails)
      init.uniform_(self.weight, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
      init.uniform_(self.bias, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
      init.constant_(self.sigma_weight, self.sigma_init)
      init.constant_(self.sigma_bias, self.sigma_init)

  def forward(self, input):
    return F.linear(input, self.weight + self.sigma_weight * Variable(self.epsilon_weight), self.bias + self.sigma_bias * Variable(self.epsilon_bias))

  def sample_noise(self):
    self.epsilon_weight = torch.randn(self.out_features, self.in_features)
    self.epsilon_bias = torch.randn(self.out_features)

  def remove_noise(self):
    self.epsilon_weight = torch.zeros(self.out_features, self.in_features)
    self.epsilon_bias = torch.zeros(self.out_features)

class ActorCritic(nn.Module):
  def __init__(self, observation_space, action_space, hidden_size, sigma_init, no_noise):
    super(ActorCritic, self).__init__()
    self.state_size = observation_space.shape[0]
    self.action_size = action_space.n
    self.no_noise = no_noise
    self.conv1 = nn.Conv2d(self.state_size, 32, 3, stride=2, padding=1) # first convolution
    self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1) # second convolution
    self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1) # third convolution
    self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1) # fourth convolution
    self.lstm = nn.LSTMCell(320, hidden_size)
    self.fc_actor = NoisyLinear(hidden_size, self.action_size, sigma_init=sigma_init)
    self.fc_critic = NoisyLinear(hidden_size, self.action_size, sigma_init = sigma_init)

  def forward(self, x, h):
    x = F.elu(self.conv1(x)) # forward propagating the signal from the input images to the 1st convolutional layer
    x = F.elu(self.conv2(x)) # forward propagating the signal from the 1st convolutional layer to the 2nd convolutional layer
    x = F.elu(self.conv3(x)) # forward propagating the signal from the 2nd convolutional layer to the 3rd convolutional layer
    x = F.elu(self.conv4(x))
    x = x.view(x.size(0),-1)
    h = self.lstm(x, h)  # h is (hidden state, cell state)
    x = h[0]
    policy = F.softmax(self.fc_actor(x), dim=1).clamp(max=1 - 1e-20)  # Prevent 1s and hence NaNs
    Q = self.fc_critic(x)
    V = (Q * policy).sum(1, keepdim=True)  # V is expectation of Q under π
    return policy, Q, V, h

  def sample_noise(self):
    if not self.no_noise:
      self.fc_actor.sample_noise()
      self.fc_critic.sample_noise()

  def remove_noise(self):
    if not self.no_noise:
      self.fc_actor.remove_noise()
      self.fc_critic.remove_noise()









