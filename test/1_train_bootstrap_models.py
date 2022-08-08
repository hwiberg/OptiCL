import pandas as pd
from imp import reload
import numpy as np
import math
from sklearn.utils.extmath import cartesian
import time
import sys
import os
import opticl
# import opticl.embed_mip as em
from pyomo import environ
from pyomo.environ import *
np.random.seed(0)

bs= 50
viol_rule = 0.5

version = 'TPDP_robust'
outcome = 'palatability'
Γ = 0.5 ## what is this for?
threshold = 0.5

nutr_val = pd.read_excel('../notebooks/WFP/processed-data/Syria_instance.xlsx', sheet_name='nutr_val', index_col='Food')
nutr_req = pd.read_excel('../notebooks/WFP/processed-data/Syria_instance.xlsx', sheet_name='nutr_req', index_col='Type')
cost_p = pd.read_excel('../notebooks/WFP/processed-data/Syria_instance.xlsx', sheet_name='FoodCost', index_col='Supplier').iloc[0,:]
dataset = pd.read_csv('../notebooks/WFP/processed-data/WFP_dataset.csv').sample(frac=1)

y = dataset['label']
X = dataset.drop(['label'], axis=1, inplace=False)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)


data = X_train
gr = True

for alg in ['linear','cart','gbm','svm','rf','mlp']:
    print("Algorithm = %s" % alg)
    print("Bootstrap iterations = %d" % bs)
    code_version = '%s_bs_%d' % (alg, bs)
    outcome_list = {'palatability': {'lb':threshold, 'ub':None, 'objective_weight':0,'group_models':gr,
    'task_type': 'continuous', 'alg_list':[alg], 'bootstrap_iterations':bs,
                                       'X_train':X_train, 'y_train':y_train, 'X_test':X_test, 'y_test':y_test,
                                       'dataset_path':'../notebooks/WFP/processed-data/WFP_dataset.csv'}}
    performance = opticl.train_ml_models(outcome_list, version)
    performance.to_csv('results/%s/%s_performance.csv' % (alg, code_version))
