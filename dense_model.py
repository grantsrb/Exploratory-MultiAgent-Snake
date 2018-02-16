import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
import matplotlib.pyplot as plt

"""
Description: A simple feedforward model with batchnorm.
"""

class Model(nn.Module):
    def __init__(self, input_space, output_space, env_type='snake-v0', view_net_input=False):
        super(Model, self).__init__()
        self.hidden_dim = 200
        self.input_space = input_space
        self.output_space = output_space

        self.entry = nn.Linear(input_space[-1], self.hidden_dim)
        self.bnorm1 = nn.BatchNorm1d(self.hidden_dim)
        self.hidden = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.bnorm2 = nn.BatchNorm1d(self.hidden_dim)

        self.action_out = nn.Linear(self.hidden_dim, output_space)
        self.value_out = nn.Linear(self.hidden_dim, 1)

        self.env_type = env_type
        if env_type == "Pong-v0":
            self.view_shape = (80,80)
        else:
            self.view_shape = (60,60)
        self.view_net_input = view_net_input
        self.viewer = None

    def forward(self, x):
        fx = x.data[:,0,:] - .5*x.data[:,1,:]
        #if self.viewer is None and x.shape[0] == 2:
        #        self.viewer  = plt.imshow(fx[0].numpy().reshape(self.view_shape))
        #elif x.shape[0] == 2:
        #    self.viewer.set_data(fx[0].numpy().reshape(self.view_shape))
        #    plt.pause(0.5)
        #    plt.draw()

        fx = F.relu(self.entry(Variable(fx)))
        fx = self.bnorm1(fx)
        fx = F.relu(self.hidden(fx))
        fx = self.bnorm2(fx)
        action = self.action_out(fx)
        value = self.value_out(fx)
        return value, action

    def check_grads(self):
        """
        Checks all gradients for NaN values. NaNs have a way of sneaking into pytorch...
        """
        for param in self.parameters():
            if torch.sum(param.data != param.data) > 0:
                print("NaNs in Grad!")

    def req_grads(self, grad_on):
        """
        Used to turn off and on all gradient calculation requirements for speed.

        grad_on - bool denoting whether gradients should be calculated
        """
        for p in self.parameters():
            p.requires_grad = grad_on

    @staticmethod
    def preprocess(pic, head_color):
        new_pic = np.zeros(pic.shape[:2],dtype=np.float32)
        new_pic[:,:][pic[:,:,0]==1] = 1
        new_pic[:,:][pic[:,:,0]==255] = -1.25
        new_pic[:,:][pic[:,:,1]==head_color[1]] = 1.25
        new_pic[:,:][pic[:,:,1]==255] = 0
        new_pic[:,:][pic[:,:,2]==255] = .33
        pic = new_pic
        return pic.ravel()[None]
