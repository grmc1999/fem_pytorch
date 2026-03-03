import torch
from torch import nn

class model_poisson(torch.nn.Module):
  def __init__(self,input):
    super(model_poisson,self).__init__()
    self.input = input
    self.act = nn.Tanh()
    self.linear = nn.Linear(input,int(input*0.25))
    self.linear_x = nn.Linear(int(input*0.25),input)
    self.linear_y = nn.Linear(int(input*0.25),input)
  def forward(self,x,y):
    x = self.act(self.linear(x))
    y = self.act(self.linear(y))
    return self.linear_x(x) + self.linear_y(y)

class model_diffusion(torch.nn.Module):
  def __init__(self,input):
    super(model_diffusion,self).__init__()
    self.input = input
    self.act = nn.Tanh()
    self.linear = nn.Linear(input + 1,int(input*0.25))
    self.linear_x = nn.Linear(int(input*0.25),input)
    self.linear_y = nn.Linear(int(input*0.25),input)
  def forward(self,x,y,t):
    x = self.act(self.linear(torch.concatenate((x,t), axis = 1)))
    y = self.act(self.linear(torch.concatenate((y,t), axis = 1)))
    return self.linear_x(x) + self.linear_y(y)