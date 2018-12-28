#!/usr/bin/env python
# encoding: utf-8
"""
@author: johnny
@time: 2018/10/28 15:40
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiGru(nn.Module):
    def __init__(self,opt):
        super(BiGru,self).__init__()
        self.tasks = opt.tasks
        # embedding
        self.embedding = nn.Embedding.from_pretrained(opt.vectors,freeze=True)

        # semantic encoder
        self.base_encoder = nn.GRU(input_size=opt.embed_dim, hidden_size=opt.base_hidden_size, num_layers=opt.base_num_layers,
                            bias=True, batch_first=True, dropout=opt.dropout, bidirectional=True)

        # task encoder
        # task attention
        self.att_v = nn.ParameterDict()

        for task, childs in opt.tasks.items():
            self.add_module(name=task+'_encoder',module=nn.GRU(input_size=opt.base_hidden_size*2, hidden_size=\
                opt.task_hidden_size, num_layers=opt.task_num_layers,bias=True, batch_first=True, dropout=opt.dropout,
                                                                      bidirectional=True))

            for child_task, output_size in childs.items():
                # attention
                self.add_module(name=child_task+'_attention',module=nn.Linear(opt.task_hidden_size,opt.task_hidden_size))

                # attention v
                v = nn.Parameter(torch.rand(opt.task_hidden_size))
                stdv = 1. / math.sqrt(v.size(0))
                v.data.uniform_(-stdv, stdv)
                self.att_v[child_task+'_v'] = v

                # output layer
                self.add_module(name=child_task+'_linear',module=nn.Linear(opt.task_hidden_size,opt.output_size))
        # drop out
        self.dropout = nn.Dropout(p=opt.dropout)

        # # activate func
        # self.act = nn.LeakyReLU(negative_slope=0.05,inplace=True)
        # custom vectors
        self.embedding.weight.requires_grad = False
        self.embedding.weight.copy_(opt.vectors)
        # if opt.frezze:
        #     self.embedding.weight.requires_grad = False

    def forward(self,x):
        x_embed = self.embedding(x)
        x_embed = self.dropout(x_embed)
        x_base_encode, _ = self.base_encoder(x_embed)
        x_base_encode = self.dropout(x_base_encode)

        outputs = []
        for task,child in self.tasks.items():
            task_encoder = getattr(self,task+'_encoder')
            task_code,_ = task_encoder(x_base_encode)  # task encoder
            b_size,seq_len,hidden_size = task_code.size()
            task_code_=task_code.view(b_size,seq_len,2,hidden_size//2).sum(2)  # sum the bi-directional output
            task_code_ = self.dropout(task_code_)

            for child_task, output_size in child.items():

                task_att = getattr(self,child_task+'_attention')
                att_w = F.tanh(task_att(task_code_)) # [B, T, H]

                att_v = self.att_v[child_task+'_v'].repeat(b_size,1).unsqueeze(2)  # [B, H, 1]
                att_oup = F.softmax(torch.bmm(att_w,att_v),1)  # [B,T,1]

                #  task_code [B,T,H] -->[B,H,T] for batch mat multiply
                att_oup = torch.bmm(task_code_.transpose(1,2),att_oup)
                att_oup = self.dropout(att_oup)

                task_linear = getattr(self,child_task+'_linear')  # classifier
                task_outp = task_linear(att_oup.squeeze())
                outputs.append(task_outp)
        return outputs

