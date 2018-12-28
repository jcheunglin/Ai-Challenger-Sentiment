#!/usr/bin/env python
# encoding: utf-8
"""
@author: johnny
@time: 2018/12/20 21:59
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class _Attention(nn.Module):
    """
    implementation hierachical attention
    """

    def __init__(self,h_dim=100,query_dim=50):
        super(_Attention,self).__init__()
        self.key_projection = nn.Linear(h_dim,query_dim)
        self.query_vec = nn.Parameter(torch.rand(1,query_dim),requires_grad=True)

    def forward(self, rnn_ouput):
        """
        :param rnn_ouput: shape (batch_size,time_step,2*rnn_units)
        :return: shape (batch_size,2*rnn_units
        """
        batch_size,time_step,h_dim = rnn_ouput.size()

        key = torch.tanh(self.key_projection(rnn_ouput))    # (batch_size,time_step,key_dim)
        query = self.query_vec.repeat(batch_size, 1).unsqueeze(2)  #(batch_size,key_dim,1)
        att_score = F.softmax(torch.bmm(key, query), 1)  # [batch_size,time_step,1]
        value = rnn_ouput.transpose(1, 2)  # shape (batch_size,h_dim,time_step)
        att_oup = torch.bmm(value,att_score)  # shape(batch_size,h_dim,1]

        return att_oup.squeeze()

if __name__=="__main__":
    rnn = torch.rand((32,48,100))
    net = _Attention()
    att_out = net(rnn)
    print(att_out.size())