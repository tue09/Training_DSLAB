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
        self._r_d = r_d # td_idf (d : document)
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
        self._num_clusters = num_clusters # number of cluster
        self._clusters = [Cluster() for _ in range(self._num_clusters)] # list of clusters
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
        # np.choice(range, size_of_array, no_repeat_element)
        centroid_id = np.random.choice(len(self._data), (self._num_clusters), replace=False)
        for i in range(self._num_clusters):
            self._clusters[i]._centroid = np.array(self._data[centroid_id[i]]._r_d)
            self._E += [self._clusters[i]._centroid]
    
    def compute_similarity(self, member:Member, centroid:np.ndarray):
        return cosine_similarity(member._r_d.reshape(1,-1), centroid.reshape(1,-1))
    

    def select_cluster_for(self, member):
        best_fit_cluster = None
        max_similarity = -1
        for cluster in self._clusters:
            similarity = self.compute_similarity(member, cluster._centroid)
            if similarity > max_similarity:
                max_similarity = similarity
                best_fit_cluster = cluster
        best_fit_cluster.add_member(member)
        return max_similarity

    def update_centroid_of(self, cluster: Cluster):
        member_r_ds = [member._r_d for member in cluster._members]
        avg_r_d = np.mean(member_r_ds, axis = 0)
        sqrt_sqr = np.sqrt(np.sum(avg_r_d**2))
        new_centroid = np.array([value/sqrt_sqr for value in avg_r_d])
        cluster._centroid = new_centroid

    def stopping_condition(self, criterion, threshold):
        def not_in_list(x, listt):
            for i in listt:
                if np.array_equal(x, i):
                    return False
            return True
        
        criteria = ['centroid', 'similarity', 'max_iters']
        assert criterion in criteria
        if criterion == 'max_iters':
            return True if self._iter >= threshold else False
        elif criterion == 'centroid':
            E_new = [cluster._centroid for cluster in self._clusters]
            E_new_minus_E = [centroid for centroid in E_new if not_in_list(centroid, self._E)]
            self._E = E_new
            return True if len(E_new_minus_E) <= threshold else False
        else:
            new_S_minus_S = self._new_S - self._S
            self._S = self._new_S
            return True if new_S_minus_S <= threshold else False

    def run(self, seed_value, criterion, threshold):
        self.random_init(seed_value)
        self._iter = 0
        while True:
            for cluster in self._clusters:
                cluster.reset_members()
            self._new_S = 0
            for member in self._data:
                max_S = self.select_cluster_for(member)
                self._new_S += max_S
            for cluster in self._clusters:
                self.update_centroid_of(cluster)
            self._iter += 1
            if self.stopping_condition(criterion, threshold):
                break

    def compute_purity(self):
        majority_sum = 0
        for cluster in self._clusters:
            member_labels = [member._label for member in cluster._members]
            max_count = max([member_labels.count(label) for label in range(20)])
            majority_sum += max_count
        return majority_sum * 1./len(self._data)

    def compute_NMI(self):
        I_value, H_omega, H_C, N = 0., 0., 0., len(self._data)
        for cluster in self._clusters:
            wk = len(cluster._members)*1.
            H_omega += -wk/N*np.log10(wk/N)
            member_labels = [member._label for member in cluster._members]
            for label in range(20):
                wk_cj = member_labels.count(label)*1.
                cj = self._label_count[label]
                I_value += wk_cj/N*np.log10(N*wk_cj/(wk*cj)+1e-12)
            for label in range(20):
                cj = self._label_count[label]*1.
                H_C += -cj / N*np.log10(cj/N)
            return I_value*2./(H_omega+H_C)

if __name__=='__main__':
    KM = Kmeans(num_clusters=5)
    KM.load_data('Session_1/TF-IDF/20news-bydate/full-tf-idf.txt')
    KM.run(seed_value = 42, criterion='similarity', threshold=20)
        
    with open('Session_2/result_k_means.txt', 'w') as f:
        f.write(f'KMeans:\n')
        f.write(f'Purity: {KM.compute_purity()}\n')
        f.write(f'NMI: {KM.compute_NMI()}\n')
