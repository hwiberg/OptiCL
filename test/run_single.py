import pandas as pd
from imp import reload
import numpy as np
import math
from sklearn.utils.extmath import cartesian
import time
import sys
import os
import itertools
import opticl
# import opticl.embed_mip as em
from pyomo import environ
from pyomo.environ import *
from test_mip_single import *

np.random.seed(0)

alg_list = ['cart','linear','gbm','svm','rf','mlp']
violation_list = ['average',0.0,0.1,0.25,0.5]
bs_list = [2,5,10,25,50]

param_list = list(itertools.product(*[alg_list, bs_list, violation_list]))

# idx = int(sys.argv[1])
# alg, bs, viol_rule = param_list[idx]
# df = run_experiment(alg, bs, viol_rule, n_iterations = 100)

# failed_ids = [55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
#  68, 69, 70, 71, 72, 73, 74, 115, 116, 117, 118, 119, 120, 121, 122, 
#  123, 124, 140, 144, 145, 147, 148, 149]

failed_ids_25 = [140, 144, 115, 116, 117, 118, 119,
failed_ids_50 = [145, 147, 148, 149, 120, 121, 122, 123, 124]
failed_ids_gbm = [62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,]

# patch_list = list(itertools.product(*[failed_ids, range(100)]))
# idx = int(sys.argv[1])
# param_idx, seed = patch_list[idx]
# alg, bs, viol_rule = param_list[param_idx]
# df = run_experiment(alg, bs, viol_rule, fixed_seed = seed)

seed = int(sys.argv[1])
for idx in failed_ids_v2:
	print("Failed ID = %d" % idx)
	alg, bs, viol_rule = param_list[idx]
	df = run_experiment(alg, bs, viol_rule, fixed_seed = seed)


print("Completed successfully")
