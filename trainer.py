#!/usr/bin/env python
# encoding: utf-8
"""
@author: johnny
@time: 2018/10/28 18:34
"""
import torch
import logging
from tqdm import tqdm
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from data.dataloader import generate_loader
from config import Configure
from networks.BiGru import BiGru
from framework import MyFrame
from utils import initLogging

name = 'HAN_'
mylog = 'logs/'+name+'.log'
path = 'weights/'+name+'.pkl'
initLogging(mylog)
device=3
total_epochs =30
valid_best_score=0.

train_loader, valid_loader, testa_loader = generate_loader(train_bs=32)
opt = Configure()
net = BiGru
loss_func = CrossEntropyLoss(size_average=True)
solver = MyFrame(net=net,loss=loss_func,opt=opt,lr=1e-3,device=device)
# solver.load(path)

no_optim_round =0
for epoch in range(total_epochs):
    # train
    solver.train_mode()
    train_loss=0.
    for X,y in tqdm(train_loader):
        solver.set_input(X,y)
        step_loss = solver.optimize()
        train_loss += step_loss
    train_epoch_loss = train_loss/len(train_loader)
    logging.info('epoch  %d train_loss:%.5f'%(epoch+1,train_epoch_loss))
    # elval
    valid_score = solver.eval(valid_loader)

    # save
    if valid_score > valid_best_score:
        logging.info('epoch %d valid_score improve %.5f >>>>>>>>> %.5f save model at : %s' % (epoch+1,valid_best_score,valid_score,path))
        solver.save(path)
        valid_best_score = valid_score
        no_optim_round = 0

    else:
        no_optim_round +=1
        logging.info('epoch %d valid_score %.5f '%(epoch,valid_score))

    if no_optim_round>4:
        logging.info('early stop at %d epoch' % (epoch+1))
        break

    if no_optim_round>2:
        # update lr
        if solver.old_lr < 5e-7:
            break
        solver.load(path)
        solver.update_lr(5.0, factor=True, mylog=mylog)





