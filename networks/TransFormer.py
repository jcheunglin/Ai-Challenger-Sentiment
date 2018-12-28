#!/usr/bin/env python
# encoding: utf-8
"""
@author: johnny
@time: 2018/12/24 15:31
"""
import torch
import torch.nn as nn
from networks.Modules import PosEmbedding, EncoderLayer
from networks.Attention import _Attention
from config import Configure
import torch.nn.functional as F


class Bert(nn.Module):
    """
    implementation Transformer for text classification
    """
    def __init__(self,opt,base_num_layers=4,n_heads=8,maxlen=1000):
        super(Bert,self).__init__()
        self.tasks = opt.tasks
        self.pos_embedding = PosEmbedding(maxlen=maxlen,d_model=opt.embed_dim,pad_idx=0)
        self.word_embedding = nn.Embedding.from_pretrained(opt.vectors,freeze=True)
        self.layers = nn.ModuleList([EncoderLayer(d_model=opt.embed_dim,n_heads=n_heads,d_k=32,d_v=32,d_hidden=256,dropout=opt.dropout) for _ in range(base_num_layers)])

        for task, childs in opt.tasks.items():
            # self.add_module(name=task+'_encoder',module=EncoderLayer(d_model=opt.embed_dim,n_heads=n_heads,d_k=32,d_v=32,d_hidden=256,dropout=opt.dropout))

            for child_task, output_size in childs.items():
                # conv
                self.add_module(name=child_task+'_att',module=_Attention(opt.embed_dim,query_dim=64))

                # output layer
                self.add_module(name=child_task+'_linear',module=nn.Linear(opt.embed_dim,opt.output_size))
        self.dropout = nn.Dropout(opt.dropout)

    def forward(self, x,pos_x):
        out = self.word_embedding(x)+self.pos_embedding(pos_x)
        out = self.dropout(out)
        for layer in self.layers:
            out = layer(out)

        outputs = []
        for task, child in self.tasks.items():
            # task_encoder = getattr(self, task + '_encoder')
            # task_code = task_encoder(out)  # task encoder

            for child_task, output_size in child.items():
                # attention
                task_attention = getattr(self, child_task + '_att')
                att_x = F.relu(task_attention(out))
                att_x = self.dropout(att_x)
                # clf
                task_linear = getattr(self, child_task + '_linear')  # classifier
                task_outp = task_linear(att_x)
                outputs.append(task_outp)

        return outputs

