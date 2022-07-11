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


code_version = 'refactor'
alg = 'cart'
outcome = 'palatability_2'


def normalize(y):
    minimum = 71.969  
    maximum = 444.847  
    return 1 - (y - minimum)/(maximum - minimum)

def check_violation(threshold, solution):
    # Cereals & Grains
    group1 = [1, 11, 12, 14, 15, 22, 23]
    group1_names = [list(solution.keys())[i] for i in group1]
    values1 = [solution[x] for x in group1_names]
    food_in_group1 = sum(values1)*100
#     print(f'food_in_group1 {food_in_group1}')
    idealG1 = 400
    distG1 = food_in_group1 - idealG1
    # Pulses & Vegetables
    group2 = [0, 6, 10, 13]
    group2_names = [list(solution.keys())[i] for i in group2]
    values2 = [solution[x] for x in group2_names]
    food_in_group2 = sum(values2)*100
#     print(f'food_in_group2 {food_in_group2}')
    idealG2 = 65
    distG2 = food_in_group2 - idealG2
    # Oils & Fats
    group3 = [21]
    group3_names = [list(solution.keys())[i] for i in group3]
    values3 = [solution[x] for x in group3_names]
    food_in_group3 = sum(values3)*100
#     print(f'food_in_group3 {food_in_group3}')
    idealG3 = 27
    distG3 = food_in_group3 - idealG3
    # Mixed & Blended Foods
    group4 = [5, 24, 16, 17, 18, 19]
    group4_names = [list(solution.keys())[i] for i in group4]
    values4 = [solution[x] for x in group4_names]
    food_in_group4 = sum(values4)*100
#     print(f'food_in_group4 {food_in_group4}')
    idealG4 = 45
    distG4 = food_in_group4 - idealG4
    # Meat & Fish & Dairy
    group5 = [2, 3, 4, 7, 8]
    group5_names = [list(solution.keys())[i] for i in group5]
    values5 = [solution[x] for x in group5_names]
    food_in_group5 = sum(values5)*100
#     print(f'food_in_group5 {food_in_group5}')
    idealG5 = 30
    distG5 = food_in_group5 - idealG5
    real_palatability = np.round(math.sqrt(distG1 ** 2 + (5.7 * distG2) ** 2 + (16.6 * distG3) ** 2
                                      + (4.4 * distG4) ** 2 + (6.6 * distG5) ** 2), 3)
    real_palatability_norm = normalize(real_palatability)
    return 1-int(real_palatability_norm>=threshold), real_palatability_norm

def init_conceptual_model(cost_p):
    N = list(nutr_val.index)  # foods
    M = nutr_req.columns  # nutrient requirements
    model = ConcreteModel('TPDP')
    '''
    Decision variables
    '''
    model.x = Var(N, domain=NonNegativeReals)  # variables controlling the food basket
    '''
    Objective function.
    '''
    def obj_function(model):
        return sum(cost_p[food].item()*model.x[food] for food in N)
    model.OBJ = Objective(rule=obj_function, sense=minimize)
    '''
    Nutrients requirements constraint.
    '''
    def constraint_rule1(model, req):
        return sum(model.x[food] * nutr_val.loc[food, req] for food in N) >= nutr_req[req].item()
    model.Constraint1 = Constraint(M, rule=constraint_rule1)
    '''
    Sugar constraint
    '''
    def constraint_rule2(model):
        return model.x['Sugar'] == 0.2
    model.Constraint2 = Constraint(rule=constraint_rule2)
    '''
    Salt constraint
    '''
    def constraint_rule3(model):
        return model.x['Salt'] == 0.05
    model.Constraint3 = Constraint(rule=constraint_rule3)
    return model

nutr_val = pd.read_excel('../notebooks/WFP/processed-data/Syria_instance.xlsx', sheet_name='nutr_val', index_col='Food')
nutr_req = pd.read_excel('../notebooks/WFP/processed-data/Syria_instance.xlsx', sheet_name='nutr_req', index_col='Type')
cost_p = pd.read_excel('../notebooks/WFP/processed-data/Syria_instance.xlsx', sheet_name='FoodCost', index_col='Supplier').iloc[0,:]
dataset = pd.read_csv('../notebooks/WFP/processed-data/WFP_dataset.csv').sample(frac=1)

y = dataset['label']
X = dataset.drop(['label'], axis=1, inplace=False)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

threshold = 0.5
trust_region = True
lb = threshold; ub = np.nan;
weight_objective = 0; 
task = 'continuous'
data = X_train
Γ = 0.5
alg_list = [alg]


version = 'TPDP_v1'
outcome_list = {'palatability_2': {'outcome_type': ['constraint', None], 'task_type': 'continuous', 
'alg_list':alg_list, 
                                   'X_train':X_train, 'y_train':y_train, 'X_test':X_test, 'y_test':y_test}}
constraints_embed = ['palatability_2']
objectives_embed = {}
performance = pd.read_csv('../notebooks/WFP/results/%s/%s_%s_performance.csv' % (alg,version, outcome))
performance.dropna(axis='columns')
performance['task'] = 'continuous'
performance.loc[:,'save_path'] = '../notebooks/WFP/' + performance['save_path']

# columns_df = ['algorithm','iteration','price_matrix']+list(X.columns)+['objective_function', 'real_palat', 'pred_palat', 'violation', 'time']
# solutions_df = pd.DataFrame(columns = columns_df)

model_master = opticl.model_selection(performance.query('alg == "%s"' % alg), 
                                  outcome_list)
model_master['lb'] = threshold
model_master['ub'] = None

model_master[['lb', 'ub', 'SCM_counterfactuals', 'features', 'trust_region', 'dataset_path',
              'clustering_model', 'max_violation', 'enlargement', 'var_features', 'contex_features']] = None

model_master['features'] = model_master['features'].astype('object')
model_master['var_features'] = model_master['var_features'].astype('object')
model_master['contex_features'] = model_master['contex_features'].astype('object')
model_master['enlargement'] = model_master['contex_features'].astype('object')

model_master.loc[0, 'lb'] = 0.5
model_master.loc[0, 'ub'] = None
model_master.loc[0, 'SCM_counterfactuals'] = None
model_master.at[0, 'features'] = [col for col in X.columns]
model_master.loc[0, 'trust_region'] = True
model_master.loc[0, 'dataset_path'] = '../notebooks/WFP/processed-data/WFP_dataset.csv'
model_master.loc[0, 'clustering_model'] = None
model_master.loc[0, 'max_violation'] = None
model_master.at[0, 'var_features'] = [col for col in X.columns]
model_master.at[0, 'contex_features'] = {}  # example: {'contextual_feat_name_1': 1, contextual_feat_name_2': 5}
model_master.at[0, 'enlargement'] = [0]

## duplicate model to test ensemble constraint
model_master.loc[1,:] = model_master.loc[0,:]

opticl.check_model_master(model_master)

i = 1
np.random.seed(i)
price_random = pd.Series(np.random.random(len(cost_p))*1000)
price_random.index = cost_p.index

conceptual_model= init_conceptual_model(price_random)
MIP_final_model = opticl.optimization_MIP(conceptual_model, model_master)
MIP_final_model.write('mip_%s_%s.lp' % (code_version, alg))
opt = SolverFactory('gurobi')
start_time = time.time()
results = opt.solve(MIP_final_model) 
computation_time = time.time() - start_time
pred_palat = value(MIP_final_model.y[outcome])

print("\nPredicted palatability: %.3f" % pred_palat)
violation_bool, real_palat = check_violation(threshold,  MIP_final_model.x.get_values())
## Save solutions
solution = MIP_final_model.x.get_values()

print("\nSolution: ")
for i in solution.keys():
    if solution[i] >= 1e-5:
        print("%s: %.3f" % (i, solution[i]))

result_save = pd.DataFrame({'variable':list(solution.keys()), 'value':list(solution.values())})
result_save.loc[len(result_save.index)] = [outcome,pred_palat]
result_save.to_csv('results_%s_%s.csv' % (code_version, alg), index = False)

# solution['algorithm'] = alg
# solution['iteration'] =i
# solution['price_matrix'] = price_random.to_dict()
# solution['violation'] = violation_bool
# solution['real_palat'] = real_palat
# solution['pred_palat'] = pred_palat
# solution['objective_function'] = value(MIP_final_model.OBJ)
# solution['time'] = computation_time
# solutions_df = solutions_df.append(solution, ignore_index=True)
# print(f"The predicted palatability of the optimal solution is {pred_palat}")