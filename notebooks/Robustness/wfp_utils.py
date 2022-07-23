import pandas as pd
from imp import reload
import numpy as np
import math
from sklearn.utils.extmath import cartesian
import time
import sys
import os
sys.path.append(os.path.abspath('../../src'))  # TODO: has to be changed
import opticl
from pyomo import environ
from pyomo.environ import *

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
    distG5 = food_in_group5 - idealG
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
