
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

alg_list = ['gbm','rf','mlp']
violation_list = [0.1]
bs_list = [25]
seed_list = range(100)

param_list = list(itertools.product(*[alg_list, bs_list, violation_list, seed_list]))

# failed_ids_gbm = [62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,]

# failed_ids = [0,1,2]
#failed_ids = [21, 235, 242, 293, 297, 305, 342, 362, 376, 389, 393, 409, 442, 493, 509, 542, 572, 593]
idx = int(sys.argv[1])
# idx = failed_ids[int(sys.argv[1])]

print("JOB ID = %d" % idx)
alg, bs, viol_rule, seed = param_list[idx]
print("Params = (%s, %s, %s); seed = %d" % (alg, bs, viol_rule, seed))
df = run_experiment(alg, bs, viol_rule, fixed_seed = seed, time_limit = 14400, print_output = True)
print("Completed successfully")
