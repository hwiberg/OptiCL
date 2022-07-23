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


code_version = 'refactor_july23_cart'
alg_list = ['cart']
bs=5
version = 'TPDP_v1'
outcome = 'palatability'
Γ = 0.5 ## what is this for?
threshold = 0.5
gr=True
gr_method = 'violation'
max_viol = 0.5


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

data = X_train
outcome_list = {'palatability': {'lb':threshold, 'ub':None, 'objective_weight':0,'group_models':gr,
'task_type': 'continuous', 'alg_list':alg_list, 'bootstrap_iterations':bs,
                                   'X_train':X_train, 'y_train':y_train, 'X_test':X_test, 'y_test':y_test,
                                   'dataset_path':'../notebooks/WFP/processed-data/WFP_dataset.csv'}}

performance = opticl.train_ml_models(outcome_list, version)

# performance = pd.read_csv('../notebooks/WFP/results/%s/%s_%s_performance.csv' % (alg,version, outcome))
# performance.dropna(axis='columns')
# performance['task'] = 'continuous'
# performance.loc[:,'outcome_label'] = performance['outcome']
# columns_df = ['algorithm','iteration','price_matrix']+list(X.columns)+['objective_function', 'real_palat', 'pred_palat', 'violation', 'time']
# solutions_df = pd.DataFrame(columns = columns_df)

mm = opticl.initialize_model_master(outcome_list)
mm.loc[outcome,'group_method'] = gr_method
mm.loc[outcome,'max_violation'] = max_viol
model_master = opticl.model_selection(mm, performance)

# model_master.at[outcome,'model'] = {
#     '../notebooks/WFP/results/cart/TPDP_v1_palatability_model.csv':'cart',
#     '../notebooks/WFP/results/linear/TPDP_v1_palatability_model.csv':'linear'
#     }

## duplicate model to test ensemble constraint
opticl.check_model_master(model_master)


i = 1
np.random.seed(i)
price_random = pd.Series(np.random.random(len(cost_p))*1000)
price_random.index = cost_p.index

conceptual_model= init_conceptual_model(price_random)
MIP_final_model = opticl.optimization_MIP(conceptual_model, model_master)
MIP_final_model.write('mip_%s.lp' % (code_version))
opt = SolverFactory('gurobi')
start_time = time.time()
results = opt.solve(MIP_final_model) 
computation_time = time.time() - start_time
pred_palat = value(MIP_final_model.y[outcome])

print("\nPredicted palatability: %.3f" % pred_palat)

print("\nPredictions - individual models: ")
sol_y = MIP_final_model.y.get_values()
for i in sol_y.keys():
    print("%s: %.3f" % (i, sol_y[i]))
    try:
        print("%s violation: %.3f" % (i, MIP_final_model.y_viol.get_values()[i]))
    except:
        pass


violation_bool, real_palat = check_violation(threshold,  MIP_final_model.x.get_values())
## Save solutions
solution = MIP_final_model.x.get_values()
print("\nSolution: ")
for i in solution.keys():
    if solution[i] >= 1e-5:
        print("%s: %.3f" % (i, solution[i]))

# for c in MIP_final_model.component_objects(environ.Constraint, active=True):
#     print(c)
#     # MIP_final_model.c.lslack()
#     # MIP_final_model.c.uslack()

# MIP_final_model.lowerBoundViolpalatability['palatability_0'].uslack()
# MIP_final_model.lowerBoundViolpalatability['palatability_1'].uslack()

if gr:
    MIP_final_model.GroupAvgpalatability.pprint()
    if gr_method == 'violation':
        MIP_final_model.constraintViolpalatability.pprint()

result_save = pd.DataFrame({'variable':list(solution.keys()), 'value':list(solution.values())})
result_save.loc[len(result_save.index)] = [outcome,pred_palat]
result_save.to_csv('results_%s.csv' % (code_version), index = False)

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