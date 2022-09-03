
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

failed_ids_gbm = [62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,]

failed_seeds = [1, 5, 8, 9, 10, 16, 18, 19, 20, 21, 22, 23, 25, 27, 28, 29, 30, 31, 34, 35, 36, 37, 40, 42, 43, 45, 54, 56, 57, 58, 59, 61, 62, 63, 64, 65, 67, 72, 73, 76, 77, 81, 83, 85, 89, 90, 93, 97, 98, 99]

job_list = list(itertools.product(*[failed_ids_gbm, failed_seeds]))

idx, seed = job_list[int(sys.argv[1])]
print("Failed ID = %d" % idx)
print("Seed = %d" % seed)
alg, bs, viol_rule = param_list[idx]
df = run_experiment(alg, bs, viol_rule, fixed_seed = seed)


print("Completed successfully")
