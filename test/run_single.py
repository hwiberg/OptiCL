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
np.random.seed(0)

alg_list = ['cart','linear','gbm','svm','rf','mlp']
violation_list = ['average',0.0,0.1,0.25,0.5]
bs_list = [0,5,10,25,50]

param_list = list(itertools.product(*[alg_list, bs_list, violation_list]))

idx = sys.args[1]
alg, bs, viol_rule = param_list[idx]

df = run_experiment(alg, bs, viol_rule, n_iterations = 100)

print("Completed successfully")