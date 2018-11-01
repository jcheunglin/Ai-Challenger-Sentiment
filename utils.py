#!/usr/bin/env python
# encoding: utf-8
"""
@author: johnny
@time: 2018/10/29 13:41
"""
import logging
from sklearn.metrics import f1_score


def initLogging(logFilename):
  """Init for logging
  """
  logging.basicConfig(
                    level    = logging.DEBUG,
                    format='%(asctime)s-%(levelname)s-%(message)s',
                    datefmt  = '%y-%m-%d %H:%M',
                    filename = logFilename,
                    filemode = 'w');
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
  console.setFormatter(formatter)
  logging.getLogger('').addHandler(console)


def eval(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')