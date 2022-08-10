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

# alg = sys.argv[1]
# bs= int(sys.argv[2])
# # violation_list = ['average',0.0,0.1,0.25,0.5]
# viol_rule = 0.5



# summ = solutions_df.groupby(['algorithm','bootstraps','viol_rule'])[['objective_function','real_palat','pred_palat','violation','time']].mean()
# print(summ)



def run_experiment(alg, bs, viol_rule, n_iterations = 100, fixed_seed = None):
    if alg == 'ensemble':
        alg_list = ['cart','linear','gbm','svm','rf','mlp']
        print("Ensemble with methods: ", alg_list)
    else: 
        alg_list = [alg]

    print("Algorithm = %s" % alg)
    print("Bootstrap iterations = %d" % bs)
    print("Violation rule = %s" % str(viol_rule))
    code_version = '%s_bs_%d_group_%s' % (alg, bs, str(viol_rule))

    version = 'TPDP_v1'
    outcome = 'palatability'
    Γ = 0.5 ## what is this for?
    threshold = 0.5
    gr=True

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

    def init_simple_model():
        N = range(5)  # foods
        model = ConcreteModel('SimpleModel')
        model.x = Var(N, domain=NonNegativeReals)  # variables controlling the food basket
        def obj_function(model):
            return sum(food*model.x[food] for food in N)
        model.OBJ = Objective(rule=obj_function, sense=minimize)
        def constraint_rule(model):
            return sum(model.x[food] for food in N) == 1
        model.Constraint = Constraint(rule=constraint_rule)
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
    
    perf_list = []
    for idx in range(bs):
        perf_list.append(pd.read_csv('results/%s/%s_%s_s%d_performance.csv' % (alg, version, outcome, idx)))
    performance = pd.concat(perf_list)
    print(performance)

    columns_df = ['algorithm','iteration','price_matrix']+list(X.columns)+['objective_function', 'real_palat', 'pred_palat', 'violation', 'time']
    solutions_df = pd.DataFrame(columns = columns_df)

    ####### Solve test problem to start up pyomo
    simple_model = init_simple_model()
    opt = SolverFactory('gurobi')
    start_time = time.time()
    results = opt.solve(simple_model) 
    computation_time = time.time() - start_time
    print("Simple problem runtime = %.3f" % computation_time)
    ###### Done with test instance

    print("\nPreparing model master")
    if viol_rule == 'average':
        gr_method = 'average'
        max_viol = None
        print("Group method = %s" % (gr_method))
        gr_string = 'average'
    else: 
        gr_method = 'violation'
        max_viol = float(viol_rule)
        print("Group method = %s (violation limit = %.2f)" % (gr_method, max_viol))
        gr_string = 'violation_%.2f' % max_viol


    mm = opticl.initialize_model_master(outcome_list)
    mm.loc[outcome,'group_method'] = gr_method
    mm.loc[outcome,'max_violation'] = max_viol
    model_master = opticl.model_selection(mm, performance)
    model_master.to_csv('experiments/model_master_%s.csv' % (code_version), index = True)

    opticl.check_model_master(model_master)

    if fixed_seed == None:
        iter_list = range(n_iterations)
    else:
        iter_list = range(fixed_seed, fixed_seed+1)
        code_version = code_version+'_seed%s' % fixed_seed
    for seed in iter_list:
        print("\nRunning iteration %d" % seed)
        np.random.seed(seed)
        price_random = pd.Series(np.random.random(len(cost_p))*1000)
        price_random.index = cost_p.index

        conceptual_model= init_conceptual_model(price_random)
        MIP_final_model = opticl.optimization_MIP(conceptual_model, model_master)
        # MIP_final_model.write('experiments/mip_%s_seed_%d.lp' % (code_version, seed))
        opt = SolverFactory('gurobi')
        start_time = time.time()
        results = opt.solve(MIP_final_model) 
        computation_time = time.time() - start_time
        pred_palat = value(MIP_final_model.y[outcome])


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

        if gr:
            MIP_final_model.GroupAvgpalatability.pprint()
            if gr_method == 'violation':
                MIP_final_model.constraintViolpalatability.pprint()

        solution['algorithm'] = alg
        solution['bootstraps'] = bs
        solution['viol_rule'] = viol_rule
        solution['iteration'] = seed
        solution['price_matrix'] = price_random.to_dict()
        solution['violation'] = violation_bool
        solution['real_palat'] = real_palat
        solution['pred_palat'] = pred_palat
        solution['objective_function'] = value(MIP_final_model.OBJ)
        solution['time'] = computation_time
        solutions_df = solutions_df.append(solution, ignore_index=True)

        print("\nPredicted palatability: %.3f" % pred_palat)
        print("Real palatability: %.3f" % real_palat)
        print("Total cost: %.3f" % value(MIP_final_model.OBJ))
        print("Runtime: %.3f" % computation_time)

    solutions_df.to_csv('experiments/solution_%s.csv' % code_version, index = False)
    return solutions_df