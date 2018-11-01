#!/usr/bin/env python
# encoding: utf-8
'''
@author: johnny
@time: 2018/10/28 9:56
'''

import pandas as pd
import jieba
import time

jieba.load_userdict('../raw_data/word.txt')

def process():
    train_set = pd.read_csv('../raw_data/sentiment_analysis_trainingset.csv',index_col='id')
    valid_set = pd.read_csv('../raw_data/sentiment_analysis_validationset.csv',index_col='id')
    testa_set = pd.read_csv('../raw_data/sentiment_analysis_testa.csv',index_col='id')

    clean_train_comment= map(lambda comment: comment.strip('\"').replace('\n','').replace('\t','').replace('\r','').lower(),train_set['content'])
    clean_valid_comment= map(lambda comment: comment.strip('\"').replace('\n','').replace('\t','').replace('\r','').lower(),valid_set['content'])
    clean_testa_comment= map(lambda comment: comment.strip('\"').replace('\n','').replace('\t','').replace('\r','').lower(),testa_set['content'])

    stop_sigs=['?','、','。','“','”','《','》','：','；','？','．','，']
    cut_train_comment = [' '.join([word for word in jieba.cut(comment) if word not in stop_sigs])  for comment in clean_train_comment]
    cut_valid_comment = [' '.join([word for word in jieba.cut(comment) if word not in stop_sigs])  for comment in clean_valid_comment]
    cut_testa_comment = [' '.join([word for word in jieba.cut(comment) if word not in stop_sigs])  for comment in clean_testa_comment]

    train_set['content']=cut_train_comment
    valid_set['content']=cut_valid_comment
    testa_set['content']=cut_testa_comment

    train_set.to_csv('../processed/cut_train_set.csv',index=True,encoding='utf-8',header=True)
    valid_set.to_csv('../processed/cut_valid_set.csv',index=True,encoding='utf-8',header=True)
    testa_set.to_csv('../processed/cut_testa_set.csv',index=True,encoding='utf-8',header=True)


if __name__ == '__main__':
    start_t = time.time()
    process()
    end_t = time.time()
    print('cut all comment  using %.3f s'%(end_t-start_t))