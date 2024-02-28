import os
import pandas as pd
import numpy as np
import copy
import random
import re
from datetime import datetime
from os import listdir
from os.path import isfile
from nltk.stem.porter import PorterStemmer
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix

def load_data(data_path):
    def sparse_to_dense(sparse_r_d, vocab_size):
        r_d = [0.0 for _ in range(vocab_size)]
        indices_td_idfs = sparse_r_d.split()
        for index_tf_idf in indices_td_idfs:
            index = int(index_tf_idf.split(':')[0])
            tf_idf = float(index_tf_idf.split(':')[1])
            r_d[index] = tf_idf
        return np.array(r_d)
    
    with open(data_path) as f:
        d_lines = f.read().splitlines()
    with open('Session_1/TF-IDF/20news-bydate/words_idfs.txt') as f:
        vocab_size = len(f.read().splitlines())

    data = []
    labels = []
    for data_id, d in enumerate(d_lines):
        features = d.split('<fff>')
        label, doc_id = int(features[0]), int(features[1])
        r_d = sparse_to_dense(sparse_r_d=features[2], vocab_size=vocab_size)
        data.append(r_d)
        labels.append(label)
    return data, labels

def Kmean_with_scikit_learn():
    print("XXX")
    data, labels = load_data(data_path='Session_1/TF-IDF/20news-bydate/full-tf-idf.txt')
    
    X = csr_matrix(data)
    print("X = ")
    print(X)
    kmeans = KMeans(
        n_clusters=20,
        init = 'random',
        n_init=5,
        tol=1e-3,
        random_state=2024
    ).fit(X)
    labels = kmeans.labels_
    print("labels = ")
    print(labels.size)
    print(labels)

if __name__ == "__main__":
    Kmean_with_scikit_learn()
    