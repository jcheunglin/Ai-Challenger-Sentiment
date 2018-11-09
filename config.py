#!/usr/bin/env python
# encoding: utf-8
"""
@author: johnny
@time: 2018/10/28 18:35
"""
import torch
import pandas as pd
import numpy as np
train_set = pd.read_csv('./processed/cut_valid_set.csv',index_col='id',encoding='utf-8').drop('content',1)
embedding = torch.load('./processed/embedding.pkl')

output_size = 4
task_list = ['location', 'service', 'price', 'environment', 'dish', 'other']
task_name = train_set.columns.tolist()
tasks = {}.fromkeys(task_list)
# task_loss = {}.fromkeys(range(len(task_name)))

for task in tasks.keys():
    tasks[task]={}
    for name in task_name:
        if task in name:
            tasks[task][name]=output_size

# for task in range(len(task_name)):
#     weight=train_set[task_name[task]].value_counts(sort=False).values
#     weight_=torch.from_numpy((1+np.log(weight.max()/weight)).astype(np.float32))
#     if torch.cuda.is_available():
#         weight_ = weight_.cuda()
#     task_loss[task] = torch.nn.CrossEntropyLoss(weight=weight_)

class Configure(object):
    """
    configure for net work
    """
    vocab_size, embed_dim = embedding.size()
    vectors = embedding
    base_hidden_size = 100
    base_num_layers = 1
    dropout = 0.
    task_hidden_size=50
    task_num_layers =2
    output_size=4
    tasks = tasks
    task_loss = torch.nn.CrossEntropyLoss()
    device = 3