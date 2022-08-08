import pandas as pd
from imp import reload
import numpy as np
import math
from sklearn.utils.extmath import cartesian
import time
import sys
import os
import opticl
np.random.seed(0)

alg = sys.argv[1]
bs= int(sys.argv[2])
n_iterations = 25

print("Algorithm = %s" % alg)
print("Bootstrap iterations = %d" % bs)

sols = []

for viol_rule in ['average',0.0,0.1,0.25,0.5]:
    if viol_rule == 'average':
        gr_method = 'average'
        max_viol = None
        gr_string = 'average'
    else: 
        gr_method = 'violation'
        max_viol = float(viol_rule)
        gr_string = 'violation_%.2f' % max_viol
    code_version = '%s_bs_%d_group_%s' % (alg, bs, gr_string)
    for seed in range(n_iterations):
        x = pd.read_csv('experiments/solution_%s.csv' % (code_version))
        sols.append(x)

df_sols = pd.concat(sols)

summ = df_sols.groupby(['algorithm','bootstraps','viol_rule'])[['objective_function','real_palat','pred_palat','violation','time']].mean()
print(summ)
