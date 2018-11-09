#!/usr/bin/env python
# encoding: utf-8
"""
@author: johnny
@time: 2018/11/6 17:40
"""
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

class BiGruCNN(nn.Module):
    def __init__(self,opt):
        super(BiGruCNN,self).__init__()
        self.tasks = opt.tasks
        # embedding
        self.embedding = nn.Embedding(num_embeddings=opt.vocab_size,embedding_dim=opt.embed_dim)

        # semantic encoder
        self.base_encoder = nn.GRU(input_size=opt.embed_dim, hidden_size=opt.base_hidden_size, num_layers=opt.base_num_layers,
                            bias=True, batch_first=True, dropout=opt.dropout, bidirectional=True)

        # task encoder

        for task, childs in opt.tasks.items():
            self.add_module(name=task+'_encoder',module=nn.GRU(input_size=opt.base_hidden_size*2, hidden_size=\
                opt.task_hidden_size, num_layers=opt.task_num_layers,bias=True, batch_first=True, dropout=opt.dropout,
                                                                      bidirectional=True))

            for child_task, output_size in childs.items():
                # conv
                self.add_module(name=child_task+'_cnn',module=nn.Conv2d(in_channels=1,out_channels=opt.task_hidden_size,kernel_size=(3,opt.task_hidden_size)))

                # output layer
                self.add_module(name=child_task+'_linear',module=nn.Linear(opt.task_hidden_size*2,opt.output_size))
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

            # cnn
            b_size,seq_len,hidden_size = task_code.size()
            task_code_=task_code.view(b_size,seq_len,2,hidden_size//2).sum(2)  # sum the bi-directional output
            task_code_ = self.dropout(task_code_)
            task_code_ = task_code_.unsqueeze(1)
            for child_task, output_size in child.items():

                # cnn
                task_cnn = getattr(self,child_task+'_cnn')
                cnn_x = F.relu(task_cnn(task_code_).squeeze(3))

                # pool
                max_pool_x = F.max_pool1d(cnn_x,cnn_x.size()[2]).squeeze(2)
                avg_pool_x = F.avg_pool1d(cnn_x,cnn_x.size()[2]).squeeze(2)
                pool_x = torch.cat([max_pool_x,avg_pool_x],1)

                # clf
                task_linear = getattr(self,child_task+'_linear')  # classifier
                task_outp = task_linear(pool_x)
                outputs.append(task_outp)
        return outputs