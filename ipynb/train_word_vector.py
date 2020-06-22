# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 14:27:19 2020

@author: 86188
"""
import os
import sys
import numpy as np 
import pandas as pd
import logging
import gc
import tqdm
import pickle
import json
import time
import tempfile
from gensim.models import Word2Vec
from gensim.test.utils import common_texts, get_tmpfile

def initiate_logger(log_path):
    """
    Initialize a logger with file handler and stream handler
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-s: %(message)s', datefmt='%H:%M:%S')
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.info('===================================')
    logger.info('Begin executing at {}'.format(time.ctime()))
    logger.info('===================================')
    return logger

def train(target, embed_size, logger=None):
    """
    Train a Word2Vec Model and save the model artifact
    """
    global corpus_dic, embed_path
    assert target in corpus_dic

    start = time.time()
    with open(corpus_dic[target], 'rb') as f:
        corpus = pickle.load(f)
    if logger: logger.info('{} corpus is loaded after {:.2f}s'.format(target.capitalize(), time.time()-start))

    model = Word2Vec(sentences=corpus, size=embed_size, window=10, sg=1, hs=1, min_count=1, workers=16)
    if logger: logger.info('{} w2v training is done after {:.2f}s'.format(target.capitalize(), time.time()-start))

    save_path = os.path.join(embed_path, '{}_sg_embed_s{}_'.format(target, embed_size))
    with tempfile.NamedTemporaryFile(prefix=save_path, delete=False) as tmp:
        tmp_file_path = tmp.name
        model.save(tmp_file_path)
    if logger: logger.info('{} w2v model is saved to {} after {:.2f}s'.format(target.capitalize(), tmp_file_path, time.time()-start))

    return tmp_file_path

def save_wv(target, embed_size, logger=None):
    global embed_path, w2v_registry
    assert target in w2v_registry

    start = time.time()
    model = Word2Vec.load(w2v_registry[target])
    save_path = os.path.join(embed_path, '{}_sg_embed_s{}_wv_'.format(target, embed_size))
    with tempfile.NamedTemporaryFile(prefix=save_path, delete=False) as tmp:
        tmp_file_path = tmp.name
        model.wv.save(tmp_file_path)
    if logger: logger.info('{} word vector is saved to {} after {:.2f}s'.format(target.capitalize(), tmp_file_path, time.time()-start))

    return tmp_file_path