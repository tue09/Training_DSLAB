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

class Member():
    def __init__(self, r_d, label=None, doc_id=None):
        self._r_d = r_d # td_idf
        self._label = label # newsgroup (label)
        self._doc_id = doc_id # document id
    
class Cluster():
    def __init__(self):
        self._centroid = None # centroid of cluster
        self._members = [] # member list in this cluster

    def reset_members(self):
        self._members = []

    def add_member(self, member):
        self._members.append(member)

class Kmeans():
    def __init__(self, num_clusters):
        self._num_cluters = num_clusters # number of cluster
        self._cluster = [Cluster() for _ in range(self._num_clusters)] # list of clusters
        self._E = [] # list of centroids
        self._S = 0 # total similarity

    def load_data(self, data_path):
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

        self._data = []
        self._label_count = defaultdict(int)
        for data_id, d in enumerate(d_lines):
            features = d.split('<fff>')
            label, doc_id = int(features[0]), int(features[1])
            self._label_count[label] += 1
            r_d = sparse_to_dense(sparse_r_d=features[2], vocab_size=vocab_size)
            self._data.append(Member(r_d, label, doc_id))

    def random_init(self, seed_value):
        pass

    def compute_similarity(self, member, centroid):
        pass

    def select_cluster_for(self, member):
        pass

    def update_centroid_of(self, cluster):
        pass

    def stopping_condition(self, criterion, threshold):
        pass

    def run(self, seed_value, criterion, threshold):
        pass

    def compute_purity(self):
        pass

    def compute_NMI(self):
        pass

if __name__=='__main__':
    KM = Kmeans(num_clusters=5)
    KM.load_data('Session_1/TF-IDF/20news-bydate/full-tf-idf.txt')
