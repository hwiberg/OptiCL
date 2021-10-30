import pandas as pd
import numpy as np
import math
from sklearn.cluster import KMeans, DBSCAN
from pyclustering.cluster.xmeans import xmeans, kmeans_plusplus_initializer
from pyomo import environ
from pyomo.environ import *


def optimization_MIP(model,
                     x,  ## decision variables (already attached to model)
                     model_master,  ## master table that specifies learned functions for constraints (and parameters)
                     data,  ## dataframe holding all data to be used for convex hull
                     max_violation=None, ## parameter for RF models - to be deprecated in favor of specifying within model_master
                     tr=True,  ## bool variable for the use of trust region constraints
                     clustering_model=None):  ## trained clustering algorithm using the entire data

    def constraints_tree(model, outcome, tree_table, lb=None, ub=None, M=1e5, weight_objective=0):
        ## Add y[outcome] to track predicted value
        ## Leaf-level information (set leaf ID as index)
        leaf_values = tree_table.loc[:, ['ID', 'prediction']].drop_duplicates().set_index('ID')
        # Row-level information: row = single constraint (multiple rows can correspond to single leaf)
        intercept = tree_table['threshold']
        coeff = tree_table.drop(['ID', 'threshold', 'prediction'], axis=1, inplace=False).reset_index(drop=True)
        l_ids = tree_table['ID']
        n_constr = coeff.shape[0]
        L = np.unique(tree_table['ID'])

        def constraintsTree_1(model, j):
            return sum(model.x[i]*coeff.loc[j, i] for i in N) <= intercept.iloc[j] + M*(1-model.l[(outcome,str(l_ids.iloc[j]))])

        def constraintsTree_2(model):
            return sum(model.l[(outcome, str(i))] for i in L) == 1

        def constraintTree(model):
            return model.y[outcome] == sum(leaf_values.loc[i, 'prediction'] * model.l[(outcome, str(i))] for i in L)

        model.add_component(outcome+'_1', Constraint(range(n_constr), rule=constraintsTree_1))
        model.add_component('DT'+outcome, Constraint(rule=constraintTree))
        model.add_component(outcome+'_2', Constraint(rule=constraintsTree_2))

        if weight_objective != 0:
            model.OBJ.set_value(expr=model.OBJ.expr + weight_objective * model.y[outcome])
        else:
            if not pd.isna(ub):
                model.add_component('ub_'+outcome, Constraint(expr=model.y[outcome] <= ub))
                model.upperBound = Constraint(expr=model.y[outcome] <= ub)
            elif not pd.isna(lb):
                model.add_component('lb_' + outcome, Constraint(expr=model.y[outcome] >= lb))

    def constraints_gbm(model, outcome, gbm_table, ub=None, lb=None, weight_objective=0):
        gbm_table['Tree_id'] = [outcome + '_' + str(i) for i in gbm_table['Tree_id']]
        T = np.unique(gbm_table['Tree_id'])

        ## For each tree in the forest, add tree to model and define outcome y
        for i, t in enumerate(T):
            tree_table = gbm_table.loc[gbm_table['Tree_id'] == t, :].drop(
                ['Tree_id', 'initial_prediction', 'learning_rate'], axis=1, inplace=False)
            # don't set LB, UB, or objective for individual trees
            constraints_tree(model, t, tree_table, lb=None, ub=None, weight_objective=0)

        # ## Compute average (as y[outcome]), either for avg. constraint or objective
        def constraint_gbm(model):
            return model.y[outcome] == np.unique(gbm_table['initial_prediction']).item() + np.unique(gbm_table['learning_rate']).item() * quicksum(model.y[j] for j in T)

        model.add_component('GBM'+outcome, Constraint(rule=constraint_gbm))
        if weight_objective != 0:
            model.OBJ.set_value(expr=model.OBJ.expr + weight_objective * model.y[outcome])
        else:
            if not pd.isna(ub):
                model.add_component('ub_' + outcome, Constraint(expr=model.y[outcome] <= ub))
                model.upperBound = Constraint(expr=model.y[outcome] <= ub)
            elif not pd.isna(lb):
                model.add_component('lb_' + outcome, Constraint(expr=model.y[outcome] >= lb))

    def constraints_rf(model, outcome, forest_table, ub=None, lb=None, max_violation=None, weight_objective=0):
        forest_table['Tree_id'] = [outcome + '_' + str(i) for i in forest_table['Tree_id']]
        T = np.unique(forest_table['Tree_id'])

        ## For each tree in the forest, add tree to model and define outcome y
        for i, t in enumerate(T):
            tree_table = forest_table.loc[forest_table['Tree_id'] == t, :].drop('Tree_id', axis=1)
            # don't set LB, UB, or objective for individual trees
            constraints_tree(model, t, tree_table, lb=None, ub=None, weight_objective=0)

        ## Compute average (as y[outcome]), either for avg. constraint or objective
        model.add_component('RF'+outcome, Constraint(rule=model.y[outcome] == 1 / len(T) * quicksum(model.y[j] for j in T)))
        if weight_objective != 0:
            model.OBJ.set_value(expr=model.OBJ.expr + weight_objective * model.y[outcome])
        else:
            if not pd.isna(ub):
                model.add_component('ub_' + outcome, Constraint(expr=model.y[outcome] <= ub))
                model.upperBound = Constraint(expr=model.y[outcome] <= ub)
            elif not pd.isna(lb):
                model.add_component('lb_' + outcome, Constraint(expr=model.y[outcome] >= lb))
            else:
                # Identify violating trees
                if not pd.isna(ub):
                    def constraint_upperBoundViol(model, j):
                        return 1 / 100 * (model.y[j] - ub) <= model.y_viol[(outcome, str(j))]
                    model.add_component('upperBoundViol'+outcome, Constraint(T, rule=constraint_upperBoundViol))
                if not pd.isna(lb):
                    def constraint_lowerBoundViol(model, j):
                        return 1 / 100 * (lb - model.y[j]) <= model.y_viol[(outcome, str(j))]
                    model.add_component('lowerBoundViol' + outcome, Constraint(T, rule=constraint_lowerBoundViol))
                # Constrain proportion of trees that violate bound to be at most max_violation
                model.add_component('constraintViol'+outcome, Constraint(rule=1 / len(T) * sum(model.y_viol[(outcome, str(j))] for j in T) <= max_violation))

    def constraints_linear(model, outcome, coefficients, lb=None, ub=None, weight_objective=0):
        # Row-level information: row = single constraint (multiple rows can correspond to single leaf)
        intercept = coefficients['intercept'][0]
        coeff = coefficients.drop(['intercept'], axis=1, inplace=False).loc[0, :]
        model.add_component('LR'+outcome, Constraint(expr=model.y[outcome] == sum(model.x[i] * coeff.loc[i] for i in N) + intercept))

        if weight_objective != 0:
            model.OBJ.set_value(expr=model.OBJ.expr + weight_objective * model.y[outcome])
        else:
            if not pd.isna(ub):
                model.add_component('ub_' + outcome, Constraint(expr=model.y[outcome] <= ub))
                model.upperBound = Constraint(expr=model.y[outcome] <= ub)
            elif not pd.isna(lb):
                model.add_component('lb_' + outcome, Constraint(expr=model.y[outcome] >= lb))

    def constraints_mlp(model, outcome, weights, lb=None, ub=None, weight_objective=0, M_l=-1e5, M_u=1e5):
        ## Add y[outcome] to track predicted value
        max_layer = max(weights['layer'])

        # Recursively generate constraints linking nodes between layers, starting from input
        v_input = list(x.values())
        nodes_input = range(len(x))
        for l in range(max_layer + 1):
            df_layer = weights.query('layer == %d' % l)
            max_nodes = [k for k in df_layer.columns if 'node_' in k]
            # coeffs_layer = np.array(df_layer.iloc[:, range(len(max_nodes))].dropna(axis=1))
            coeffs_layer = np.array(df_layer.loc[:, max_nodes].dropna(axis=1))
            intercepts_layer = np.array(df_layer['intercept'])
            nodes = df_layer['node']

            if l == max_layer:
                node = nodes.iloc[0]  # only one node in last layer
                model.add_component('MLP'+outcome, Constraint(rule=model.y[outcome] == sum(v_input[i] * coeffs_layer[node, i] for i in nodes_input) + intercepts_layer[
                        node]))
            else:
                # Save v_pos for input to next layer
                v_pos_list = []
                for node in nodes:
                    ## Initialize variables

                    v_pos_list.append(model.v[(outcome, l, node)])

                    model.add_component('constraint_1_'+str(node)+outcome,
                                        Constraint(rule=model.v[(outcome, l, node)] >= sum(v_input[i] * coeffs_layer[node, i] for i in nodes_input) + intercepts_layer[node]))
                    model.add_component('constraint_2_' + str(node) + outcome,
                                        Constraint(rule=model.v[(outcome, l, node)] <= M_u * (model.v_ind[(outcome, l, node)])))
                    model.add_component('constraint_3_' + str(node) + outcome,
                                        Constraint(rule=model.v[(outcome, l, node)] <= sum(v_input[i] * coeffs_layer[node, i] for i in nodes_input) + intercepts_layer[node] - M_l * (1 - model.v_ind[(outcome, l, node)])))
                ## Prepare nodes_input for next layer
                nodes_input = nodes
                v_input = v_pos_list

        if weight_objective != 0:
            model.OBJ.set_value(expr=model.OBJ.expr + weight_objective * model.y[outcome])
        else:
            if not pd.isna(ub):
                model.add_component('ub_' + outcome, Constraint(expr=model.y[outcome] <= ub))
                model.upperBound = Constraint(expr=model.y[outcome] <= ub)
            elif not pd.isna(lb):
                model.add_component('lb_' + outcome, Constraint(expr=model.y[outcome] >= lb))

    def constraints_tr(model, samples, data, clustering_model):
        print(f'Generating constraints for the trust region using {len(samples)} samples.')
        model.lam = Var(samples, domain=Reals, name=['lambda_%s' % str(x) for x in samples], bounds=(0,1))

        def constraint_CTR1(model, i):
            return model.x[i] == sum(model.lam[k] * data.loc[k, i] for k in samples)

        if clustering_model is not None:
            print('Using the clustering algorithm')
            n_clusters = np.unique(clustering_model.labels_)
            model.u = Var(n_clusters, name=['cluster_%d' % x for x in n_clusters], domain=Binary)

            def constraint_CTR2(model, label):
                cluster = data[clustering_model.labels_ == label]
                cluster_samples = cluster.index
                return sum(model.lam[k] for k in cluster_samples) == model.u[label]

            model.ConstraintClusteredTrustRegion1 = Constraint(np.unique(clustering_model.labels_), rule=constraint_CTR2)
            model.ConstraintClusteredTrustRegion2 = Constraint(data.columns, rule=constraint_CTR1)
            model.ConstraintClusteredTrustRegion3 = Constraint(rule=sum(model.u[label] for label in np.unique(clustering_model.labels_)) == 1)
        else:
            model.add_component('ConstraintClusteredTrustRegion1', Constraint(rule=sum(model.lam[k] for k in samples) == 1))
            model.add_component('ConstraintClusteredTrustRegion2', Constraint(data.columns, rule=constraint_CTR1))

        print('... Trust region defined.')

    ## Decision variable indices
    N = data.columns
    samples = data.index

    if tr:
        constraints_tr(model, samples, data, clustering_model)

    for i, row in model_master.iterrows():
        if i == 0:
            model.y = Var(Any, dense=False, domain=Reals)
            model.l = Var(Any, dense=False, domain=Binary)
            model.y_viol = Var(Any, dense=False, domain=Binary)
            model.v = Var(Any, dense=False, domain=NonNegativeReals)
            model.v_ind = Var(Any, dense=False, domain=Binary)
        if row['objective']!=0:
            print('Embedding Objective function')
        else:
            print(f"Embedding Constraints for {row['outcome']}")
        mtype = row['model_type']
        mfile = pd.read_csv(row['save_path'])
        if mtype in ['cart', 'oct']:
            constraints_tree(model, row['outcome'], mfile, lb=row['lb'], ub=row['ub'],
                             weight_objective=row['objective'])
        elif mtype == 'rf':
            constraints_rf(model, row['outcome'], mfile, lb=row['lb'], ub=row['ub'], max_violation=max_violation,
                           weight_objective=row['objective'])
        elif mtype == 'gbm':
            constraints_gbm(model, row['outcome'], mfile, lb=row['lb'], ub=row['ub'], weight_objective=row['objective'])
        elif mtype == 'mlp':
            constraints_mlp(model, row['outcome'], mfile, lb=row['lb'], ub=row['ub'], weight_objective=row['objective'])
        elif mtype in ['linear', 'svm']:
            constraints_linear(model, row['outcome'], mfile, lb=row['lb'], ub=row['ub'],
                               weight_objective=row['objective'])
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


def model_selection(performance, constraints_embed=[], objectives_embed={}, scores=False):
    ## If don't specify any constraints/objectives, assume all constraints
    outcomes_embed = constraints_embed + list(objectives_embed.keys())
    # print(outcomes_embed)
    if outcomes_embed == []:
        outcomes_embed = np.unique(performance['outcome'])
    performance_embed = performance.loc[performance.loc[:, 'outcome'].isin(outcomes_embed), :]
    # performance['score'] = performance['auc_test'].combine_first(performance['test_r2'])
    performance_embed.loc[:, 'score'] = performance_embed.loc[:, 'valid_score']
    performance_embed.sort_values(['outcome', 'score'], ascending=[True, False], axis=0, inplace=True)
    if scores:
        model_master = performance_embed.groupby('outcome').first().reset_index(). \
            rename({'alg': 'model_type'}, axis=1)
    else:
        model_master = performance_embed.groupby('outcome').first().reset_index(). \
                           rename({'alg': 'model_type'}, axis=1).loc[:, ['outcome', 'model_type', 'save_path']]
    model_master['objective'] = model_master['outcome'].apply(lambda x: objectives_embed[x] \
        if x in objectives_embed.keys() \
        else 0)
    print(model_master)
    return model_master


def check_model_master(model_master):
    obj_cnt = sum(model_master['objective'] != 0)
    if obj_cnt == 0:
        print("No learned objective")
    else:
        obj = list(model_master.query('objective != 0')['outcome'])
        print(f"Learn objective for {obj}; will add to manually set objective.")

    for i, row in model_master.query('objective!=0').iterrows():
        print(f"\nEmbedding objective term for {row['outcome']} using {row['model_type']} model.")
        print(f"Outcome weight = {row['objective']}")

    for i, row in model_master.query('objective==0').iterrows():
        constraint_lb = f"{round(row['lb'], 3)} <= " if not pd.isna(row['lb']) else ""
        constraint_ub = f" <= {round(row['ub'], 3)}" if not pd.isna(row['ub']) else ""
        if (constraint_lb + constraint_ub) != "":
            print(f"\nEmbedding constraint for {row['outcome']} using {row['model_type']} model.")
            print(constraint_lb + row['outcome'] + constraint_ub)


def train_clustering_algorithm(X, algorithm, **kwargs):
    assert algorithm in ['kmean', 'xmean', 'dbscan', 'birch']  # to be extended
    if algorithm == 'kmean':
        clustering = KMeans(**kwargs).fit(X)
    elif algorithm == 'dbscan':
        clustering = DBSCAN(**kwargs).fit(X)
    elif algorithm == 'xmean':
        try:
            amount_initial_centers = kwargs['amount_initial_centers']
        except KeyError:
            amount_initial_centers = 2
        initial_centers = kmeans_plusplus_initializer(X, amount_initial_centers).initialize()
        # Create instance of X-Means algorithm. The algorithm will start analysis from 2 clusters, the maximum
        # number of clusters that can be allocated is 20.
        clustering = xmeans(X, initial_centers, **kwargs)
        clustering.process()
    return clustering
