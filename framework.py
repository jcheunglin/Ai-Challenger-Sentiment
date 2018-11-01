#!/usr/bin/env python
# encoding: utf-8
"""
@author: johnny
@time: 2018/10/29 12:48
"""
import torch
from torch.autograd import Variable as V
from sklearn.metrics import f1_score

class MyFrame(object):

    def __init__(self, net, loss,lr=2e-4,device=0,opt=None,rmp=False,):
        self.device = device
        self.opt = opt
        self.net = net(opt)
        self.loss = loss
        if torch.cuda.is_available():
            self.net.cuda(device=self.device)
        # self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        if rmp:
            # self.optimizer = torch.optim.SGD(params=self.net.parameters(), lr=lr,momentum=0.9,nesterov=True)
            self.optimizer = torch.optim.RMSprop(params=self.net.parameters(), lr=lr)
        self.old_lr = lr
    def set_input(self, X_batch, y_batch=None):
        self.X = X_batch
        self.y = y_batch

    def train_mode(self):
        self.net.train()
        return self

    def eval(self,valid_loader):
        self.net.eval()
        score_dict = {}.fromkeys(range(20))
        eval_dict = {}.fromkeys(range(20))

        for key in eval_dict.keys():
            eval_dict[key]={'pred':[],'label':[]}

        for valid_x, valid_y in valid_loader:
            valid_x,valid_y = valid_x.cuda(self.device),valid_y.cuda(self.device)
            batch_pred = self.net(valid_x)
            for idx in range(20):
                pred = batch_pred[idx].cpu().argmax(1).tolist()
                label = valid_y[:,idx].cpu().tolist()

                eval_dict[idx]['pred'] += pred
                eval_dict[idx]['label'] += label
        for idx in range(20):
            score_dict[idx]= f1_score(eval_dict[idx]['label'],eval_dict[idx]['pred'],average='macro')

        valid_score = sum(score_dict.values())/20
        return valid_score


    def forward(self, volatile=False):
        if torch.cuda.is_available():
            self.X = V(self.X.cuda(device=self.device), volatile=volatile)
            if self.y is not None:
                self.y = V(self.y.cuda(device=self.device), volatile=volatile)
        # self.X = V(self.X, volatile=volatile)
        # self.y = V(self.y, volatile=volatile)

    def optimize(self):
        self.forward()
        self.optimizer.zero_grad()
        pred = self.net(self.X)
        # loss_func = self.opt.task_loss
        loss = self.loss(pred[0], self.y[:, 0])/20
        for idx in range(1,20):
            # loss_func = self.opt.task_loss
            loss += self.loss(pred[idx],self.y[:,idx])/20

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

    def update_lr(self, new_lr, mylog, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        self.old_lr = new_lr
