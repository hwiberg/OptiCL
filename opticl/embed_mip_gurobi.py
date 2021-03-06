import pandas as pd
import numpy as np
import math
from gurobipy import Model, GRB, quicksum, tupledict
from sklearn.cluster import KMeans, DBSCAN

def optimization_MIP(model, 
                    x,  ## decision variables (already attached to model)
                    model_master,  ## master table that specifies learned functions for constraints (and parameters)
                    data,  ## dataframe holding all data to be used for convex hull
                    max_violation = None,  ## parameter for RF models - to be deprecated in favor of specifying within model_master
                    tr = True,  ## bool variable for the use of trust region constraints
                    clustering_model = None):  ## trained clustering algorithm using the entire data

    def constraints_tree(model, outcome, tree_table, lb = None, ub = None, M =  1e5, weight_objective=0):
        ## Add y[outcome] to track predicted value
        y[outcome] = model.addVar(vtype=GRB.CONTINUOUS, name = 'y_%s' % outcome, lb = -float('inf'))
        ## Leaf-level information (set leaf ID as index)
        leaf_values = tree_table.loc[:,['ID','prediction']].drop_duplicates().set_index('ID')
        # Row-level information: row = single constraint (multiple rows can correspond to single leaf)
        intercept = tree_table['threshold']
        coeff = tree_table.drop(['ID', 'threshold', 'prediction'], axis=1, inplace=False).reset_index(drop=True)
        l_ids = tree_table['ID']
        n_constr = coeff.shape[0]
        L = np.unique(tree_table['ID'])
        for leaf in L:
            l[(outcome,leaf)] = model.addVar(vtype=GRB.BINARY, name='l_%s_%d' % (outcome, leaf))

        model.addConstrs(quicksum(x[i]*coeff.loc[j, i] for i in N) <= intercept.iloc[j]
                         + M*(1-l[(outcome,l_ids.iloc[j])]) for j in range(n_constr))
        model.addConstr(y[outcome]==quicksum(leaf_values.loc[j, 'prediction']*l[(outcome,j)] for j in L))
        model.addConstr(quicksum(l[(outcome,j)] for j in L)==1)

        if weight_objective != 0:
            obj.addTerms(weight_objective, y[outcome])
        else:
            if not pd.isna(ub):
                model.addConstr(y[outcome]<=ub)
            elif not pd.isna(lb):
                model.addConstr(y[outcome]>=lb)

    def constraints_gbm(model, outcome, gbm_table, ub=None, lb=None, weight_objective=0):
        gbm_table['Tree_id'] = [outcome + '_' + str(i) for i in gbm_table['Tree_id']]
        T = np.unique(gbm_table['Tree_id'])

        ## For each tree in the forest, add tree to model and define outcome y
        for t in T:
            tree_table = gbm_table.loc[gbm_table['Tree_id']==t,:].drop(['Tree_id', 'initial_prediction', 'learning_rate'], axis=1, inplace=False)
            # don't set LB, UB, or objective for individual trees
            constraints_tree(model, t, tree_table, lb=None, ub=None, weight_objective=0)

        ## Compute average (as y[outcome]), either for avg. constraint or objective
        y[outcome] = model.addVar(vtype=GRB.CONTINUOUS, name = 'y_%s' % outcome, lb = -float('inf'))
        model.addConstr(y[outcome] == np.unique(gbm_table['initial_prediction']).item() + np.unique(gbm_table['learning_rate']).item()*quicksum(y[j] for j in T))
        if weight_objective != 0:
            # objective function based on the average value of trees predictions
            obj.addTerms(weight_objective, y[outcome])
        else:
            if not pd.isna(ub):
                model.addConstr(y[outcome] <= ub)
            if not pd.isna(lb):
                model.addConstr(y[outcome] >= lb)

    def constraints_rf(model, outcome, forest_table, ub = None, lb = None, max_violation = None, weight_objective=0):
        forest_table['Tree_id'] = [outcome + '_' + str(i) for i in forest_table['Tree_id']]
        T = np.unique(forest_table['Tree_id'])
        # print(T)
        
        ## For each tree in the forest, add tree to model and define outcome y
        for t in T:
            # print(t)
            tree_table = forest_table.loc[forest_table['Tree_id']==t,:].drop('Tree_id', axis=1)

            # don't set LB, UB, or objective for individual trees
            constraints_tree(model, t, tree_table, lb = None, ub = None, weight_objective = 0)

        ## Compute average (as y[outcome]), either for avg. constraint or objective
        y[outcome] = model.addVar(vtype=GRB.CONTINUOUS, name = 'y_%s' % outcome,  lb = -float('inf'))
        model.addConstr(y[outcome] == 1/len(T)*quicksum(y[j] for j in T))

        if weight_objective != 0:
            # print('##############')
            # objective function based on the average value of trees predictions
            obj.addTerms(weight_objective, y[outcome])
        else:
            if pd.isna(max_violation):
                if not pd.isna(ub):
                    # print('##############')
                    model.addConstr(y[outcome] <= ub)
                if not pd.isna(lb):
                    # print('##############')
                    model.addConstr(y[outcome] >= lb)
            else:
                ## Apply constraint to full set of trees
                y_viol = model.addVars(T, vtype=GRB.BINARY, name='y_viol')

                # Identify violating trees
                if not pd.isna(ub):
                    model.addConstrs(1/100*(y[j] - ub) <= y_viol[j] for j in T)
                if not pd.isna(lb):
                    model.addConstrs(1/100*(lb - y[j]) <= y_viol[j] for j in T)

                # Constrain proportion of trees that violate bound to be at most max_violation
                model.addConstr(1/len(T)*quicksum(y_viol[j] for j in T) <= max_violation)

    def constraints_linear(model, outcome, coefficients, lb=None, ub=None, weight_objective=0):
        ## Add y[outcome] to track predicted value
        y[outcome] = model.addVar(vtype=GRB.CONTINUOUS, name='y_%s' % outcome, lb = -float('inf'))

        # Row-level information: row = single constraint (multiple rows can correspond to single leaf)
        intercept = coefficients['intercept'][0]
        coeff = coefficients.drop(['intercept'], axis=1, inplace=False).loc[0, :]

        model.addConstr(y[outcome] == quicksum(x[i] * coeff.loc[i] for i in N) + intercept)

        if weight_objective != 0:
            obj.addTerms(weight_objective, y[outcome])
        else:
            if not pd.isna(ub):
                model.addConstr(y[outcome] <= ub)
            elif not pd.isna(lb):
                model.addConstr(y[outcome] >= lb)

    def constraints_mlp(model, outcome, weights, lb=None, ub=None, weight_objective=0, M_l = -1e5, M_u = 1e5):
        ## Add y[outcome] to track predicted value
        y[outcome] = model.addVar(vtype=GRB.CONTINUOUS, name='y_%s' % outcome, lb = -float('inf'))

        max_layer = max(weights['layer'])

        # Recursively generate constraints linking nodes between layers, starting from input
        v_input = x.values()
        nodes_input = range(len(x))
        for l in range(max_layer+1):
            df_layer = weights.query('layer == %d' % l)
            max_nodes = [k for k in df_layer.columns if 'node_' in k]
            # coeffs_layer = np.array(df_layer.iloc[:, range(len(max_nodes))].dropna(axis=1))
            coeffs_layer = np.array(df_layer.loc[:, max_nodes].dropna(axis=1))
            intercepts_layer = np.array(df_layer['intercept'])
            nodes = df_layer['node']

            if l == max_layer:
                node = nodes.iloc[0] #only one node in last layer
                model.addConstr(y[outcome]==quicksum(v_input[i] * coeffs_layer[node,i] for i in nodes_input) + intercepts_layer[node])

            else:
                # Save v_pos for input to next layer
                v_pos_list = []
                for node in nodes: 
                    ## Initialize variables
                    v = model.addVar(vtype=GRB.CONTINUOUS, name='v_%s_%d_%d' % (outcome, l, node), lb = 0)
                    v_ind = model.addVar(vtype=GRB.BINARY, name='v_ind_%s_%d_%d' % (outcome, l, node))

                    v_pos_list.append(v)

                    model.addConstr(v >= quicksum(v_input[i] * coeffs_layer[node,i] for i in nodes_input) + intercepts_layer[node])

                    model.addConstr(v <= M_u*(v_ind)) # binding if sum < 0
                    model.addConstr(v <= quicksum(v_input[i] * coeffs_layer[node,i] for i in nodes_input) + intercepts_layer[node]\
                                    - M_l*(1-v_ind)) # binding if sum >= 0
                ## Prepare nodes_input for next layer
                nodes_input = nodes 
                v_input = v_pos_list

        if weight_objective != 0:
            obj.addTerms(weight_objective, y[outcome])
        else:
            if not pd.isna(ub):
                model.addConstr(y[outcome] <= ub)
            elif not pd.isna(lb):
                model.addConstr(y[outcome] >= lb)

    def constraints_tr(model, samples, data, clustering_model):
        # print("Generating Trust Region Constraints")
        ## Feasibility (Convex Hull)
        lam = model.addVars(samples, vtype=GRB.CONTINUOUS, name = ['lambda_%s' % str(x) for x in samples], lb=0, ub=1)
        if clustering_model is not None:
            print('Using the clustering algorithm')
            n_clusters = np.unique(clustering_model.labels_)
            print(n_clusters)
            z = model.addVars(n_clusters, name = ['cluster_%d' % x for x in n_clusters], vtype=GRB.BINARY)
            for label in np.unique(clustering_model.labels_):
                cluster = data[clustering_model.labels_ == label]
                cluster_samples = cluster.index
                model.addConstr(quicksum(lam[k] for k in cluster_samples) == z[label])
            model.addConstrs(x[i] == quicksum(lam[k] * data.loc[k, i] for k in samples) for i in data.columns)
            model.addConstr(quicksum(z[label] for label in np.unique(clustering_model.labels_)) == 1)
        else:
            model.addConstr(quicksum(lam[k] for k in samples) == 1)
            model.addConstrs(x[i] == quicksum(lam[k] * data.loc[k, i] for k in samples) for i in data.columns)

    ## Decision variable indices
    N = x.keys()
    samples = data.index

    if tr:
        constraints_tr(model, samples, data, clustering_model)

    ## Embed learned models (specified in model_master)
    y = {}
    l = tupledict()
    obj = model.getObjective()

    for i, row in model_master.iterrows():
        # if row['objective']!=0:
        #     print('Embedding Objective function')
        # else:
        #     print(f"Embedding Constraints for {row['outcome']}")
        mtype = row['model_type']
        mfile =  pd.read_csv(row['save_path'])
        if mtype in ['cart','oct']:
            constraints_tree(model, row['outcome'], mfile, lb = row['lb'], ub = row['ub'], weight_objective=row['objective'])
        elif mtype == 'rf':
            constraints_rf(model, row['outcome'], mfile, lb = row['lb'], ub = row['ub'], max_violation = max_violation, weight_objective=row['objective'])
        elif mtype == 'gbm':
            constraints_gbm(model, row['outcome'], mfile, lb = row['lb'], ub = row['ub'], weight_objective=row['objective'])
        elif mtype == 'mlp':
            constraints_mlp(model, row['outcome'], mfile, lb = row['lb'], ub = row['ub'], weight_objective=row['objective'])
        elif mtype in ['linear','svm']:
            constraints_linear(model, row['outcome'], mfile, lb = row['lb'], ub = row['ub'], weight_objective=row['objective'])
        # print(obj)

    model.setObjective(obj)
    return model
    

## Helper function to handle submodels in ensemble 
def expand_outcomes(model_master):
    outcome_list = []
    for i, row in model_master.iterrows():
        if row['model_type'] == 'rf':
            outcome_list.extend([row['outcome'] + '_' + str(i) for i in range(row['submodels'])])
        else: 
            outcome_list.extend([row['outcome']])
    return outcome_list

def model_selection(performance, constraints_embed = [], objectives_embed = {}, scores = False):
    ## If don't specify any constraints/objectives, assume all constraints
    outcomes_embed = constraints_embed + list(objectives_embed.keys())
    # print(outcomes_embed)
    if outcomes_embed == []:
        outcomes_embed = np.unique(performance['outcome'])
    performance_embed = performance.loc[performance.loc[:,'outcome'].isin(outcomes_embed),:]
    # performance['score'] = performance['auc_test'].combine_first(performance['test_r2'])
    performance_embed.loc[:,'score'] = performance_embed.loc[:,'valid_score']
    performance_embed.sort_values(['outcome', 'score'], ascending = [True, False], axis=0, inplace = True)
    if scores:
        model_master = performance_embed.groupby('outcome').first().reset_index().\
            rename({'alg':'model_type'},axis=1)
    else:
        model_master = performance_embed.groupby('outcome').first().reset_index().\
            rename({'alg':'model_type'},axis=1).loc[:,['outcome','model_type','save_path']]
    model_master['objective'] = model_master['outcome'].apply(lambda x: objectives_embed[x]\
                                                              if x in objectives_embed.keys()\
                                                             else 0)
    print(model_master)
    return model_master

def check_model_master(model_master):
    obj_cnt = sum(model_master['objective']!=0)
    if obj_cnt == 0:
        print("No learned objective")
    else:
        obj = list(model_master.query('objective != 0')['outcome'])
        print(f"Learn objective for {obj}; will add to manually set objective.")

    for i, row in model_master.query('objective!=0').iterrows():
        print(f"\nEmbedding objective term for {row['outcome']} using {row['model_type']} model.")
        print(f"Outcome weight = {row['objective']}")

    for i, row in model_master.query('objective==0').iterrows():
        constraint_lb = f"{round(row['lb'],3)} <= " if not pd.isna(row['lb']) else ""
        constraint_ub = f" <= {round(row['ub'],3)}" if not pd.isna(row['ub']) else ""
        if (constraint_lb + constraint_ub) != "":
            print(f"\nEmbedding constraint for {row['outcome']} using {row['model_type']} model.")
            print(constraint_lb + row['outcome'] + constraint_ub)