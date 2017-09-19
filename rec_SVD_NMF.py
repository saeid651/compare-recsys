# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 11:03:38 2017

Recommender system using SVD and NMF

@author: Saeid Parvandeh
"""
import os.path
from surprise import Reader
from surprise import SVD
from surprise import NMF
from surprise import Dataset
from surprise import evaluate, print_perf

from pprint import PrettyPrinter
pp = PrettyPrinter(indent=2)

# path to dataset file
file_path = os.path.expanduser('/home/saeid/Documents/SVD-NMF/D11-02/new-D01')    

# As we're loading a custom dataset, we need to define a reader. In the
# movielens-100k dataset, each line has the following format:
# 'user item rating timestamp', separated by '\t' characters.
reader = Reader(line_format='user item rating', sep='\t')

data = Dataset.load_from_file(file_path, reader=reader)
data.split(n_folds=5)

# We'll use the famous SVD algorithm.
SVD_algo = SVD()
NMF_algo = NMF()

# Evaluate performances of our algorithm on the dataset.
perf = evaluate(SVD_algo, data, measures=['RMSE', 'MAE'])

print_perf(perf)