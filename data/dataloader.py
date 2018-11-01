#!/usr/bin/env python
# encoding: utf-8
"""
@author: johnny
@time: 2018/10/28 10:06
"""
import pandas as pd
import numpy as np
import torch
from torchtext import data
from data.batchwrapper import BatchWrapper

train_set=pd.read_csv('./processed/cut_train_set.csv',index_col='id',encoding='utf-8')

def generate_loader(train_bs=32,valid_bs=128,test_bs=128,device=-1,min_frq=5,max_size=100000,use_test=True,shuffle=True,load_vec=True):
    # fields
    comment = data.Field(sequential=True, tokenize=lambda x: x.split(' '), use_vocab=True, lower=True, batch_first=True)
    label = data.Field(sequential=False, use_vocab=False, batch_first=True)

    fields = [('id', None), ('comment', comment)]
    for col in train_set.drop('content', 1).columns:
        fields.append((col, label))

    train, valid = data.TabularDataset.splits(
        path='./',  # the root directory where the data lies
        train='processed/cut_train_set.csv', validation="processed/cut_valid_set.csv",
        format='csv',
        skip_header=True,
        # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
        fields=fields)
    test = data.TabularDataset(path='./processed/cut_testa_set.csv', format='csv',
                               skip_header=True, fields=[('id', None), ('comment', comment)])
    if use_test:
        comment.build_vocab(train, valid, test, min_freq=min_frq, max_size=max_size)
    else:
        comment.build_vocab(train, valid, min_freq=min_frq, max_size=max_size)

    # load custom vectors
    if load_vec:
        cust_vectors = torch.load('./processed/embedding.pkl')
        comment.vocab.set_vectors(stoi=comment.vocab.stoi, vectors=cust_vectors, dim=200)

    else:
        cust_vectors = dict(zip(comment.vocab.stoi.values(), [torch.zeros(200) for i in range(len(comment.vocab.stoi))]))
        with open('../raw_data/Tencent_AILab_ChineseEmbedding.txt', 'r', encoding='utf-8') as r:
            r.readline()
            line = r.readline()
            while line:
                values = line.rstrip().rsplit(' ')
                word = values[0]
                if word in comment.vocab.stoi.keys():
                    vector = torch.from_numpy(np.asarray(values[1:], dtype='float32'))
                    cust_vectors[comment.vocab.stoi[word]] = vector
                line = r.readline()
        comment.vocab.set_vectors(stoi=comment.vocab.stoi, vectors=cust_vectors, dim=200)
        torch.save(comment.vocab.vectors,'./processed/embedding.pkl')

    train_iter, val_iter = data.BucketIterator.splits(
        (train, valid),  # we pass in the datasets we want the iterator to draw data from
        batch_sizes=(train_bs, valid_bs),
        shuffle=shuffle,
        device=device,  # if you want to use the GPU, specify the GPU number here
        sort_key=lambda x: len(x.comment),
        # the BucketIterator needs to be told what function it should use to group the data.
        sort_within_batch=True,
        repeat=False  # we pass repeat=False because we want to wrap this Iterator layer.
    )
    test_iter = data.Iterator(test, batch_size=test_bs, device=device, sort=False, sort_within_batch=True, repeat=False)

    train_dl = BatchWrapper(train_iter, "comment", train_set.drop('content', 1).columns)
    valid_dl = BatchWrapper(val_iter, "comment", train_set.drop('content', 1).columns)
    test_dl = BatchWrapper(test_iter, "comment", None)

    return train_dl,valid_dl,test_dl

if __name__=='__main__':
    train_dl,valid_dl,testa_dl=generate_loader()