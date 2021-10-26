import pandas as pd
from gurobipy import Model, GRB, quicksum, tupledict
import math
import numpy as np
import os
import time
from tqdm import tqdm


def optimization_LP(conceptual_model,
                    x,  ## decision variables (already attached to model)
                    model_master,  ## master table that specifies learned functions for constraints (and parameters)
                    data,  ## dataset
                    tr=True,  ## bool variable for the use of trust region constraints
                    clustering_model=None):  ## trained clustering algorithm using the entire data

    def optimization_leaf(model, x, tree_constraints, data, check_leaf):
        '''
        This function solves one LP for each feasible leaf of the tree
        '''
        model.Params.OutputFlag = 0
        features = data.columns
        # Palatability Constraints
        if check_leaf:
            x_temp = {}
            for f in features:
                x_temp[f] = model.getVarByName('x['+f+']')
            for j, row in tree_constraints.iterrows():
                model.addConstr(quicksum(x_temp[i] * row[i] for i in features) <= row['threshold'])
        else:
            for j, row in tree_constraints.iterrows():
                model.addConstr(quicksum(x[i] * row[i] for i in features) <= row['threshold'])
        model.optimize()
        if check_leaf and model.status != 2:
            print('This branch of the tree leads to an infeasible solution')
        return model

    def constraints_tr(model, samples, data, clustering_model):
        print("Generating Trust Region Constraints")
        ## Feasibility (Convex Hull)
        lam = model.addVars(samples, vtype=GRB.CONTINUOUS, lb=0, ub=1)
        if clustering_model is not None:
            print('Using the clustering algorithm')
            n_clusters = np.unique(clustering_model.labels_)
            print(n_clusters)
            z = model.addVars(n_clusters, vtype=GRB.BINARY)
            for label in np.unique(clustering_model.labels_):
                cluster = data[clustering_model.labels_ == label]
                cluster_samples = cluster.index
                model.addConstr(quicksum(lam[k] for k in cluster_samples) == z[label])
            model.addConstrs(x[i] == quicksum(lam[k] * data.loc[k, i] for k in samples) for i in data.columns)
            model.addConstr(quicksum(z[label] for label in np.unique(clustering_model.labels_)) == 1)
        else:
            model.addConstr(quicksum(lam[k] for k in samples) == 1)
            model.addConstrs(x[i] == quicksum(lam[k] * data.loc[k, i] for k in samples) for i in data.columns)

    # We consider only feasible leaves of the tree. 
    tree_table = pd.read_csv(model_master['save_path'][0])
    if model_master['lb'] is not None:
        threshold = model_master['lb'][0]
        feasible_tree_table = tree_table[tree_table['prediction'] >= threshold]
    else:
        threshold = model_master['ub'][0]
        feasible_tree_table = tree_table[tree_table['prediction'] <= threshold]

    samples = data.index
    if tr:
        constraints_tr(conceptual_model, samples, data, clustering_model)

    best_obj_value = math.inf  # initialize the best objective value to infinite since we are minimizing
    best_leaf = -1
    print(f"{len(np.unique(feasible_tree_table['ID']))} LPs must be solved")
    computation_time = 0
    highest_computation_time = -math.inf
    for count, leaf in enumerate(np.unique(feasible_tree_table['ID'])):
        print(f'Solving LP {count}...')
        conceptual_model.update()
        conceptual_model_temp = conceptual_model.copy()
        tree_constraints = feasible_tree_table[feasible_tree_table['ID'] == leaf]
        start_time = time.time()
        model = optimization_leaf(conceptual_model_temp, x, tree_constraints, data, True)
        computation_time += time.time() - start_time
        if computation_time >= highest_computation_time:
            highest_computation_time = computation_time
        try:
            obj_value = model.objval
            # Update the best objective value
            if obj_value < best_obj_value:
                best_obj_value = obj_value
                best_leaf = leaf
        except Exception:
            pass
    try:
        tree_constraints = feasible_tree_table[feasible_tree_table['ID'] == best_leaf]
        model = optimization_leaf(conceptual_model, x, tree_constraints, data, False)
    except Exception:
        print("No feasible solution")
        return 0
    return model, computation_time, highest_computation_time
