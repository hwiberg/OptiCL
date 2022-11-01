from pyomo import environ
from pyomo.environ import *
from pyomo.opt import *
import numpy as np
import random
import math
import pandas as pd
import random

def access_model_info(model, var_name):
    var = []
    for v in model.getVars():
        if var_name in v.varName:
            var.append(v.x)
    return var


import gurobipy as gp
from gurobipy import Model, GRB, quicksum


def diet_model(C, nutr_matrix, nutr_req, group1, group2, group3, group4, group5):
    # https://www.unhcr.org/45fa745b2.pdf
    # Macro-category bounds
    # Cereals & Grains: max = 450, min 350, mean 400 --> +/- 50%
    maxG1 = 600
    minG1 = 200
    idealG1 = 400

    # Pulses & Vegetables: max = 100, min 50, mean 66 --> +/- 50%
    maxG2 = 100
    minG2 = 30
    idealG2 = 65

    # Oils & Fats: max = 30, min 25, mean 27 --> +/- 50%
    maxG3 = 40
    minG3 = 15
    idealG3 = 27.5

    # Mixed & Blended Foods: max = 50, min 40, mean 45 --> +/- 50%
    maxG4 = 90
    minG4 = 0
    idealG4 = 45

    # Meat & Fish & Dairy: max = 30, min 10, mean 20 --> +/- 50%
    maxG5 = 60.0
    minG5 = 0.0
    idealG5 = 30

    model = Model('diet')
    model.Params.OutputFlag = 0

    N = nutr_matrix.shape[0]
    M = len(nutr_req)

    x = model.addVars(N, vtype=GRB.CONTINUOUS, name='x')
    s = model.addVars(M, vtype=GRB.CONTINUOUS, name='slack', ub=1, lb=0)

    # Objective function
    model.modelSense = GRB.MINIMIZE
    #     model.setObjective(quicksum(s[i] for i in range(M)))
    model.setObjective(quicksum(C[j] * x[j] for j in range(N)) + 10000*quicksum(s[i] for i in range(M)))
    # model.setObjective(quicksum(C[j] * x[j] for j in range(N)))

    # Nutrient requirements constraints
    model.addConstrs(quicksum(x[j] * nutr_matrix[j, i] for j in range(N)) >= nutr_req[i] * (1 - s[i]) for i in range(M))
    # model.addConstrs(quicksum(x[j] * nutr_matrix[j, i] for j in range(N)) >= nutr_req[i] for i in range(M))

    # Marco-categories constraints
    #     model.addConstr(quicksum(x[i]*100 for i in group1) >= minG1)
    #     model.addConstr(quicksum(x[i]*100 for i in group1) <= maxG1)
    #     model.addConstr(quicksum(x[i]*100 for i in group2) >= minG2)
    #     model.addConstr(quicksum(x[i]*100 for i in group2) <= maxG2)
    #     model.addConstr(quicksum(x[i]*100 for i in group3) >= minG3)
    #     model.addConstr(quicksum(x[i]*100 for i in group3) <= maxG3)
    #     model.addConstr(quicksum(x[i]*100 for i in group4) >= minG4)
    #     model.addConstr(quicksum(x[i]*100 for i in group4) <= maxG4)
    #     model.addConstr(quicksum(x[i]*100 for i in group5) >= minG5)
    #     model.addConstr(quicksum(x[i]*100 for i in group5) <= maxG5)

    g1_sum = random.uniform(minG1, maxG1)
    g2_sum = random.uniform(minG2, maxG2)
    g3_sum = random.uniform(minG3, maxG3)
    g4_sum = random.uniform(minG4, maxG4)
    g5_sum = random.uniform(minG5, maxG5)

    model.addConstr(quicksum(x[i] * 100 for i in group1) == g1_sum)
    model.addConstr(quicksum(x[i] * 100 for i in group2) == g2_sum)
    model.addConstr(quicksum(x[i] * 100 for i in group3) == g3_sum)
    model.addConstr(quicksum(x[i] * 100 for i in group4) == g4_sum)
    model.addConstr(quicksum(x[i] * 100 for i in group5) == g5_sum)

    # Sugar constraint
    model.addConstr(x[20] == 0.2)
    # Salt constraint
    model.addConstr(x[9] == 0.05)

    model.addConstr(quicksum(s[i] for i in range(M)) <= 0.15)

    status = model.optimize()
    return model, idealG1 - g1_sum, idealG2 - g2_sum, idealG3 - g3_sum, idealG4 - g4_sum, idealG5 - g5_sum


nutr_matrix_pd = pd.read_csv('nutritional_values.csv',header=None)
nutr_matrix = np.array(nutr_matrix_pd)
# FOODS: "Beans", "Bulgur", "Cheese",	"Fish",	"Meat",	"Corn-soya blend (CSB)",	"Dates",	"Dried skim milk (
# enriched) (DSM)",	"Milk",	"Salt",	"Lentils",	"Maize",	"Maize meal",	"Chickpeas",	"Rice",
# "Sorghum/millet",	"Soya-fortified bulgur wheat",	"Soya-fortified maize meal",	"Soya-fortified sorghum grits",
# "Soya-fortified wheat flour",	"Sugar",	"Oil",	"Wheat",	"Wheat flour",	"Wheat-soya blend (WSB)"
# nutr_req___ = np.array([2100, 52.5, 89.25, 1100, 22, 500, 0.9, 1.4, 12, 160, 28, 150])  # original
nutr_req = np.array([2100, 52.5, 35, 1100, 22, 500, 0.9, 1.4, 12, 160, 28, 150])  # oil threshold moved to 35

# Real cost
# C = np.array([800, 450, 15000, 900, 1200, 800, 500, 1600, 1200, 800, 500, 300, 300, 550, 575, 320, 1100, 900, 1300, 900, 1000, 1400, 300, 300, 850])

# Cereals & Grains
group1 = [1, 11, 12, 14, 15, 22, 23]

# Pulses & Vegetables
group2 = [0, 6, 10, 13]

# Oils & Fats
group3 = [21]

# Mixed & Blended Foods
group4 = [5, 24, 16, 17, 18, 19]

# Meat & Fish & Dairy
group5 = [2, 3, 4, 7, 8]

for i in range(300000):
    C_rand = np.random.rand(25)
    model_, distG1, distG2, distG3, distG4, distG5 = diet_model(C_rand, nutr_matrix, nutr_req, group1, group2, group3, group4, group5)
    NameError = "Unable to retrieve attribute 'x'"
    if i % 10000 == 0:
        print(i)
    try:
        # Extract solution
        solution_ = access_model_info(model_, "x")

        solution = pd.DataFrame([solution_], columns=["Beans", "Bulgur", "Cheese",	"Fish",	"Meat",	"Corn-soya blend (CSB)",	"Dates",	"Dried skim milk (enriched) (DSM)",	"Milk",	"Salt",	"Lentils",	"Maize",	"Maize meal",	"Chickpeas",	"Rice",	"Sorghum/millet",	"Soya-fortified bulgur wheat",	"Soya-fortified maize meal",	"Soya-fortified sorghum grits",	"Soya-fortified wheat flour",	"Sugar",	"Oil",	"Wheat",	"Wheat flour",	"Wheat-soya blend (WSB)"
        ])

        obj_val = model_.objVal

        f = open("WFP_palatability.csv", "a")
        row = ""

        '''
        Evaluation function 
        '''
        for food in solution_:
            row += str(np.round(food, 6)) + ','
        row += str(np.round(math.sqrt(distG1 ** 2 + (5.7 * distG2) ** 2 + (16 * distG3) ** 2
                                      + (4.4 * distG4) ** 2 + (6.6 * distG5) ** 2), 10)) + '\n'
        f.write(row)
        f.close()
    except Exception:
        print("Optimization model infeasible")
        pass


# max = 211.869