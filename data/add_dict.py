#!/usr/bin/env python
# encoding: utf-8
'''
@author: johnny
@time: 2018/10/28 9:44
'''

import time

def generate_dict():
    start_t = time.time()
    with open('../raw_data/Tencent_AILab_ChineseEmbedding.txt', 'r', encoding='utf-8') as r:
        with open('../raw_data/word.txt', 'w', encoding='utf-8') as w:
            r.readline()
            line = r.readline()
            while line:
                word = line.strip().split(' ')[0]
                w.write(word + '\n')
                line = r.readline()
    end_t = time.time()
    print('finished generate word dict using %.3f s'%(end_t-start_t))


if __name__ == '__main__':
    generate_dict()