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

alg_list = ['rf','gbm','mlp']
violation_list = ['average',0.0,0.1,0.25,0.5]
bs_list = [2,5,10,25]
seeds_list = [0,20,40,60,80]

param_list = list(itertools.product(*[alg_list, bs_list, violation_list, seeds_list]))

idx = int(sys.argv[1])
alg, bs, viol_rule, seed_start = param_list[idx]

for seed in range(seed_start, seed_start+20):
    print("Seed = %d" % seed)
    df = run_experiment(alg, bs, viol_rule, fixed_seed = seed)

print("Completed successfully")
