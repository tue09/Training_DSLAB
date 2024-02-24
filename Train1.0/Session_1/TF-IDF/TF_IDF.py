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


random.seed(1)

class TF_IDF:
    def __init__(self) -> None:
        pass
    
def collect_data():    
    def gather_20newsgroups_data():
        path = "Session_1/TF-IDF/20news-bydate/"
        dirs = [path + dir_name + '/'
                for dir_name in listdir(path)
                if not isfile(path + dir_name)]
        train_dir, test_dir = (dirs[0], dirs[1]) if 'train' in dirs[0] else (dirs[1], dirs[0])
        list_newsgroups = [newsgroup for newsgroup in listdir(train_dir)]
        list_newsgroups.sort()
        return train_dir, test_dir, list_newsgroups

    def collect_data_from(parent_dir, newsgroup_list):
        data = []
        for group_id, newsgroup in enumerate(newsgroup_list):
            label = group_id
            dir_path = parent_dir + '/' + newsgroup + '/'
            files = [(filename, dir_path + filename)
                    for filename in listdir(dir_path)
                    if isfile(dir_path + filename)]
            files.sort()
            for filename, filepath in files:
                with open(filepath) as f:
                    text = f.read().lower()
                    words = [stemmer.stem(word) for word in re.split('\W+', text) if word not in stop_words]
        return data
    
    with open('Session_1/TF-IDF/stopwords.txt', encoding='unicode_escape') as f:
        stop_words = f.read().splitlines()
    stemmer = PorterStemmer()

    train_dir, test_dir, list_newsgroups = gather_20newsgroups_data()
    train_data = collect_data_from(parent_dir=train_dir, newsgroup_list=list_newsgroups)
    test_data  = collect_data_from(parent_dir=test_dir, newsgroup_list=list_newsgroups)
    full_data = train_data + test_data
    with open('Session_1/TF-IDF/20news-bydate/20news-train-processed.txt', 'w') as f:
        f.write('\n'.join(train_data))
    with open('Session_1/TF-IDF/20news-bydate/20news-test-processed.txt', 'w') as f:
        f.write('\n'.join(test_data))
    with open('Session_1/TF-IDF/20news-bydate/20news-full-processed.txt', 'w') as f:
        f.write('\n'.join(full_data))

if __name__=="__main__":
    current_directory = os.getcwd()
    print("Current Directory:", current_directory)
    
    col_data = collect_data()
    
