import pandas as pd
import numpy as np
from pyomo import environ
from pyomo.environ import *
from scipy.stats import f


def optimization_MIP(model,
                     x,  ## decision variables (already attached to model)
                     model_master,  ## master table that specifies learned functions for constraints (and parameters)
                     data,  ## dataframe holding all data to be used for convex hull
                     max_violation=None, ## parameter for RF model allowable violation proportion (between 0-1)
                     tr=True,  ## bool variable for the use of trust region constraints
                     enlarge_tr= [0, 0, 0], ## enlargement option: 0-No enlargement, 1-CH enlargement, 2-Mahalanobis distance; enlargement constraint:  0-constraint, 1-objective penalty; constraint ub/penalty multiplier/alpha
                     clustering_model=None):  ## trained clustering algorithm using the entire data (only active if tr = True)

    def logistic_x(proba):
        if proba == 0:
            proba = 0.00001
        if proba == 1:
            proba = 0.99999
        return - np.log(1 / proba - 1)

    def constraints_linear(model, outcome, task, coefficients, lb=None, ub=None, weight_objective=0, SCM=None, features=None):
        '''
        Embed a trained linear predictive model for 'outcome' into the master 'model'.
        'Coefficients' is a model file generated by the constraint_extrapolation_skEN() function.
        'lb/ub' specify the lower/upper bounds if 'outcome' is to be incorporated as a constraint.
        'weight_objective' specifies the weight to use if incorporating 'outcome' as a term in the objective.
        '''
        # Row-level information: row = single constraint (multiple rows can correspond to single leaf)
        intercept = coefficients['intercept'][0]
        coeff = coefficients.drop(['intercept'], axis=1, inplace=False).loc[0, :]
        model.add_component('LR'+outcome, Constraint(expr=model.y[outcome] == sum(model.x[i] * coeff.loc[i] for i in pd.DataFrame(coeff).index) + intercept))

        if weight_objective != 0:
            model.OBJ.set_value(expr=model.OBJ.expr + weight_objective * model.y[outcome])
        elif not pd.isna(SCM):
            model.add_component('scm_' + outcome, Constraint(expr=model.y[outcome] == SCM + model.x[outcome]))
        else:
            if not pd.isna(ub):
                if task == 'binary':
                    ub = logistic_x(proba=ub)
                model.add_component('ub_' + outcome, Constraint(expr=model.y[outcome] <= ub))
            if not pd.isna(lb):
                if task == 'binary':
                    lb = logistic_x(proba=lb)
                model.add_component('lb_' + outcome, Constraint(expr=model.y[outcome] >= lb))

    def constraints_svm(model, outcome, task, coefficients, lb=None, ub=None, weight_objective=0, SCM=None, features=None):
        '''
        Embed a trained SVM predictive model for 'outcome' into the master 'model'.
        'Coefficients' is a model file generated by the constraint_extrapolation_skSVM() function.
        'lb/ub' specify the lower/upper bounds if 'outcome' is to be incorporated as a constraint.
        'weight_objective' specifies the weight to use if incorporating 'outcome' as a term in the objective.
        '''
        # Row-level information: row = single constraint (multiple rows can correspond to single leaf)
        intercept = coefficients['intercept'][0]
        coeff = coefficients.drop(['intercept'], axis=1, inplace=False).loc[0, :]

        # Set y to decision function
        model.add_component('SVM'+outcome, Constraint(expr=model.y[outcome] == sum(model.x[i] * coeff.loc[i] for i in features) + intercept))

        # Set y to binary: 1 if expr >= 0, else 0
        # model.add_component('SVM_lb'+outcome, Constraint(expr=model.y[outcome] >= 1/M*(sum(model.x[i] * coeff.loc[i] for i in features) + intercept)))

        if weight_objective != 0:
            model.OBJ.set_value(expr=model.OBJ.expr + weight_objective * model.y[outcome])
        elif not pd.isna(SCM):
            model.add_component('scm_' + outcome, Constraint(expr=model.y[outcome] == SCM + model.x[outcome]))
        else:
            if task == "continuous":
                if not pd.isna(ub):
                    model.add_component('ub_' + outcome, Constraint(expr=model.y[outcome] <= ub))
                if not pd.isna(lb):
                    model.add_component('lb_' + outcome, Constraint(expr=model.y[outcome] >= lb))
            elif task == "binary":
                model.add_component('lb_' + outcome, Constraint(expr=model.y[outcome] >= 0))

    def constraints_tree(model, outcome, tree_table, lb=None, ub=None, M=1e5, weight_objective=0, SCM=None, features=None):
        '''
        Embed a trained decision tree predictive model for 'outcome' into the master 'model'.
        'tree_table' is a model file generated by the constraint_extrapolation_skTree() function, where each row encodes a single constraint (multiple rows can correspond to single leaf)
        'lb/ub' specify the lower/upper bounds if 'outcome' is to be incorporated as a constraint.
        'weight_objective' specifies the weight to use if incorporating 'outcome' as a term in the objective.
        'M' is an upper bound on the value at any node.
        '''
        leaf_values = tree_table.loc[:, ['ID', 'prediction']].drop_duplicates().set_index('ID')
        # Row-level information:
        intercept = tree_table['threshold']
        coeff = tree_table.drop(['ID', 'threshold', 'prediction'], axis=1, inplace=False).reset_index(drop=True)
        l_ids = tree_table['ID']
        n_constr = coeff.shape[0]
        L = np.unique(tree_table['ID'])
        def constraintsTree_1(model, j):
            return sum(model.x[i]*coeff.loc[j, i] for i in features) <= intercept.iloc[j] + M*(1-model.l[(outcome,str(l_ids.iloc[j]))])

        def constraintsTree_2(model):
            return sum(model.l[(outcome, str(i))] for i in L) == 1

        def constraintTree(model):
            return model.y[outcome] == sum(leaf_values.loc[i, 'prediction'] * model.l[(outcome, str(i))] for i in L)

        model.add_component(outcome+'_1', Constraint(range(n_constr), rule=constraintsTree_1))
        model.add_component('DT'+outcome, Constraint(rule=constraintTree))
        model.add_component(outcome+'_2', Constraint(rule=constraintsTree_2))

        if weight_objective != 0:
            model.OBJ.set_value(expr=model.OBJ.expr + weight_objective * model.y[outcome])
        elif not pd.isna(SCM):
            model.add_component('scm_' + outcome, Constraint(expr=model.y[outcome] == SCM + model.x[outcome]))
        else:
            if not pd.isna(ub):
                model.add_component('ub_'+outcome, Constraint(expr=model.y[outcome] <= ub))
            if not pd.isna(lb):
                model.add_component('lb_' + outcome, Constraint(expr=model.y[outcome] >= lb))

    def constraints_rf(model, outcome, forest_table, ub=None, lb=None, max_violation=None, weight_objective=0, SCM=None, features=None):
        '''
        Embed a trained random forest predictive model for 'outcome' into the master 'model'.
        'forest_table' is a model file generated by the constraint_extrapolation_skRF() function, where each row encodes a single constraint (multiple rows can correspond to single leaf)
        'lb/ub' specify the lower/upper bounds if 'outcome' is to be incorporated as a constraint.
        'weight_objective' specifies the weight to use if incorporating 'outcome' as a term in the objective.
        'max_violation' specifies the allowable violation proportion for a constraint (e.g. 0.2 -> 20% of trees can violate the chosen lb/ub)
        '''
        forest_table['Tree_id'] = [outcome + '_' + str(i) for i in forest_table['Tree_id']]
        T = np.unique(forest_table['Tree_id'])

        ## For each tree in the forest, add tree to model and define outcome y
        for i, t in enumerate(T):
            tree_table = forest_table.loc[forest_table['Tree_id'] == t, :].drop('Tree_id', axis=1)
            # don't set LB, UB, or objective for individual trees
            constraints_tree(model, t, tree_table, lb=None, ub=None, weight_objective=0, SCM=None, features=features)

        ## Compute average (as y[outcome]), either for avg. constraint or objective
        model.add_component('RF'+outcome, Constraint(rule=model.y[outcome] == 1 / len(T) * quicksum(model.y[j] for j in T)))
        if weight_objective != 0:
            model.OBJ.set_value(expr=model.OBJ.expr + weight_objective * model.y[outcome])
        elif not pd.isna(SCM):
            model.add_component('scm_' + outcome, Constraint(expr=model.y[outcome] == SCM + model.x[outcome]))
        else:
            if pd.isna(max_violation):
                # Constrain average values
                if not pd.isna(ub):
                    model.add_component('ub_' + outcome, Constraint(expr=model.y[outcome] <= ub))
                if not pd.isna(lb):
                    model.add_component('lb_' + outcome, Constraint(expr=model.y[outcome] >= lb))
            else:
                # Constrain proportion of trees (1 - max_violation)
                if not pd.isna(ub):
                    def constraint_upperBoundViol(model, j):
                        return 1 / 100 * (model.y[j] - ub) <= model.y_viol[(outcome, str(j))]
                    model.add_component('upperBoundViol'+outcome, Constraint(T, rule=constraint_upperBoundViol))
                if not pd.isna(lb):
                    def constraint_lowerBoundViol(model, j):
                        return 1 / 100 * (lb - model.y[j]) <= model.y_viol[(outcome, str(j))]
                    model.add_component('lowerBoundViol' + outcome, Constraint(T, rule=constraint_lowerBoundViol))
                model.add_component('constraintViol'+outcome, Constraint(rule=1 / len(T) * sum(model.y_viol[(outcome, str(j))] for j in T) <= max_violation))

    def constraints_gbm(model, outcome, task, gbm_table, ub=None, lb=None, weight_objective=0, SCM=None, features=None):
        '''
        Embed a trained gradient-boosting machine model for 'outcome' into the master 'model'.
        'gbm_table' is a model file generated by the constraint_extrapolation_skGBM() function, where each row encodes a single constraint (multiple rows can correspond to single leaf)
        'lb/ub' specify the lower/upper bounds if 'outcome' is to be incorporated as a constraint.
        'weight_objective' specifies the weight to use if incorporating 'outcome' as a term in the objective.
        '''
        gbm_table['Tree_id'] = [outcome + '_' + str(i) for i in gbm_table['Tree_id']]
        T = np.unique(gbm_table['Tree_id'])

        ## For each tree in the forest, add tree to model and define outcome y
        for i, t in enumerate(T):
            tree_table = gbm_table.loc[gbm_table['Tree_id'] == t, :].drop(
                ['Tree_id', 'initial_prediction', 'learning_rate'], axis=1, inplace=False)
            # don't set LB, UB, or objective for individual trees
            constraints_tree(model, t, tree_table, lb=None, ub=None, weight_objective=0, SCM=None, features=features)

        # ## Compute average (as y[outcome]), either for avg. constraint or objective
        def constraint_gbm(model):
            return model.y[outcome] == np.unique(gbm_table['initial_prediction']).item() + np.unique(gbm_table['learning_rate']).item() * quicksum(model.y[j] for j in T)

        model.add_component('GBM'+outcome, Constraint(rule=constraint_gbm))
        if task == 'binary':
            lb = logistic_x(proba=lb) if lb != None else None
            ub = logistic_x(proba=ub) if ub != None else None
        if weight_objective != 0:
            model.OBJ.set_value(expr=model.OBJ.expr + weight_objective * model.y[outcome])
        elif not pd.isna(SCM):
            model.add_component('scm_' + outcome, Constraint(expr=model.y[outcome] == SCM + model.x[outcome]))
        else:
            if not pd.isna(ub):
                if task == 'binary':
                    ub = logistic_x(proba=ub)
                model.add_component('ub_' + outcome, Constraint(expr=model.y[outcome] <= ub))
            if not pd.isna(lb):
                if task == 'binary':
                    lb = logistic_x(proba=lb)
                model.add_component('lb_' + outcome, Constraint(expr=model.y[outcome] >= lb))

    def constraints_mlp(model, data, outcome, task, weights, lb=None, ub=None, weight_objective=0, SCM=None, features=None, M_l=-1e5, M_u=1e5):
        '''
        Embed a trained multi-layer perceptron model for 'outcome' into the master 'model'.
        'weights' is a model file generated by the constraint_extrapolation_skMLP() function, where each row encodes the coefficients for a single node/layer pair
        'lb/ub' specify the lower/upper bounds if 'outcome' is to be incorporated as a constraint.
        'weight_objective' specifies the weight to use if incorporating 'outcome' as a term in the objective.
        'M_l' and 'M_u' are lower/upper bounds on the value at any node/layer pair.
        '''
        # Recursively generate constraints linking nodes between layers, starting from input
        nodes_input = range(len(features))
        # v_input = [x[f'col{i}'] for i in nodes_input]
        v_input = [x[i] for i in features]
        max_layer = max(weights['layer'])
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

                    model.add_component('constraint_1_' + str(l) + '_'+str(node)+outcome,
                                        Constraint(rule=model.v[(outcome, l, node)] >= sum(v_input[i] * coeffs_layer[node, i] for i in nodes_input) + intercepts_layer[node]))
                    model.add_component('constraint_2_' + str(l)+'_' + str(node) + outcome,
                                        Constraint(rule=model.v[(outcome, l, node)] <= M_u * (model.v_ind[(outcome, l, node)])))
                    model.add_component('constraint_3_' + str(l)+'_' + str(node) + outcome,
                                        Constraint(rule=model.v[(outcome, l, node)] <= sum(v_input[i] * coeffs_layer[node, i] for i in nodes_input) + intercepts_layer[node] - M_l * (1 - model.v_ind[(outcome, l, node)])))
                ## Prepare nodes_input for next layer
                nodes_input = nodes
                v_input = v_pos_list

        if weight_objective != 0:
            model.OBJ.set_value(expr=model.OBJ.expr + weight_objective * model.y[outcome])
        elif not pd.isna(SCM):
            model.add_component('scm_' + outcome, Constraint(expr=model.y[outcome] == SCM + model.x[outcome]))
        else:
            if not pd.isna(ub):
                if task == 'binary':
                    ub = logistic_x(proba=ub)
                model.add_component('ub_' + outcome, Constraint(expr=model.y[outcome] <= ub))
            if not pd.isna(lb):
                if task == 'binary':
                    lb = logistic_x(proba=lb)
                model.add_component('lb_' + outcome, Constraint(expr=model.y[outcome] >= lb))

    def constraints_tr(model, data, clustering_model, enlarge):
        '''
        Add constraints for the trust region using the convex hull of 'data'.
        If a clustering_model is specified, the solution is constrained to lie within the convex hull of a single cluster.
        '''
        samples = data.index
        print(f'Generating constraints for the trust region using {len(samples)} samples.')
        model.lam = Var(samples, domain=Reals, name=['lambda_%s' % str(x) for x in samples], bounds=(0,1))

        if clustering_model is not None:
            print('Using the clustering algorithm')
            if enlarge[0]: print('Enlargement not supported with clustering')
            n_clusters = np.unique(clustering_model.labels_)
            model.u = Var(n_clusters, name=['cluster_%d' % x for x in n_clusters], domain=Binary)

            def constraint_CTR1(model, i):
                return model.x[i] == sum(model.lam[k] * data.loc[k, i] for k in samples)

            def constraint_CTR2(model, label):
                cluster = data[clustering_model.labels_ == label]
                cluster_samples = cluster.index
                return sum(model.lam[k] for k in cluster_samples) == model.u[label]

            model.ConstraintClusteredTrustRegion1 = Constraint(data.columns, rule=constraint_CTR1)
            model.ConstraintClusteredTrustRegion2 = Constraint(n_clusters, rule=constraint_CTR2)
            model.ConstraintClusteredTrustRegion3 = Constraint(rule=sum(model.u[label] for label in n_clusters) == 1)
        else:
            if enlarge[0] != 0:
                if enlarge[0] == 2:
                    print('Mahalanobis distance')
                    mu = np.array(np.mean(X, axis=0))
                    cov = np.matrix(np.cov(X.T))
                    if elarge[1] == 0:
                        n = len(data.columns)
                        dfn = n
                        dfd = N - n
                        rv = f(dfn, dfd)
                        alpha = enlarge[2]
                        coeff = ((N ** 2 - 1) * n) / ((N - n) * N)
                        percentile = f.ppf(alpha, dfn, dfd)
                        RHS = percentile * coeff

                        def ConstraintMD1(model):
                            return np.dot(np.dot(model.x - np.array(mu), cov.I), model.x - np.array(mu))[0, 0] <= RHS
                        model.ConstraintMD1 = Constraint(rule=ConstraintMD1)
                    else:
                        beta = enlarge[2]
                        print(f'The Mahalanobis distance is penalized in the objective with penalty coeff: {beta}.')
                        model.OBJ.set_value(expr=model.OBJ.expr + model.OBJ.sense * beta * np.dot(np.dot(model.x - np.array(mu), cov.I), model.x - np.array(mu))[0, 0])

                else:
                    print('The l1 norm is used for the enlarged CH trust region')
                    model.eHelp = Var(data.columns, domain=Reals, name=['eHelper_%s' % str(x) for x in data.columns])
                    model.e = Var(data.columns, domain=Reals, name=['epsilon_%s' % str(x) for x in data.columns])
                    def constraint_ETR1(model, i):
                        return model.x[i] + model.e[i] == sum(model.lam[k] * data.loc[k, i] for k in samples)
                    def constraint_ETR21(model, i):
                        return model.eHelp[i] >= model.e[i]
                    def constraint_ETR22(model, i):
                        return model.eHelp[i] >= - model.e[i]
                    model.add_component('constraint_ETR0', Constraint(rule=sum(model.lam[k] for k in samples) == 1))
                    model.add_component('constraint_ETR1', Constraint(data.columns, rule=constraint_ETR1))
                    model.add_component('constraint_ETR21', Constraint(data.columns, rule=constraint_ETR21))
                    model.add_component('constraint_ETR22', Constraint(data.columns, rule=constraint_ETR22))

                    if enlarge[1] == 0:  # Enlarge using a bounding constraint
                        ub = enlarge[2]
                        print(f'The trust region is being enlarged with a constraint upper bounded by: {ub}.')
                        def constraint_ETR3(model):
                            return sum(model.eHelper[i] for i in data.columns) <= ub
                        model.add_component('constraint_ETR3',Constraint(rule=constraint_ETR3))
                    else:  # Enlarge using a penalty in the objective function
                        beta = enlarge[2]
                        print(f'The trust region is being enlarged with penatly coeff: {beta}.')
                        model.OBJ.set_value(expr=model.OBJ.expr + model.OBJ.sense * beta * sum(model.eHelp[i] for i in data.columns))
            else:
                def constraint_TR1(model, i):
                    return model.x[i] == sum(model.lam[k] * data.loc[k, i] for k in samples)
                model.add_component('constraint_TR0', Constraint(rule=sum(model.lam[k] for k in samples) == 1))
                model.add_component('constraint_TR1', Constraint(data.columns, rule=constraint_TR1))

        print('... Trust region defined.')

    ## Identify decision variable indices
    N = data.columns
    ## Add trust region constraints (with optional pre-trained cluster model)
    if tr:
        constraints_tr(model, data, clustering_model, enlarge_tr)

    ## Initialize variables
    model.y = Var(Any, dense=False, domain=Reals)
    model.l = Var(Any, dense=False, domain=Binary)
    model.y_viol = Var(Any, dense=False, domain=Binary)
    model.v = Var(Any, dense=False, domain=NonNegativeReals)
    model.v_ind = Var(Any, dense=False, domain=Binary)

    ## Iterate over all learned models
    for i, row in model_master.iterrows():
        if row['objective']!=0:
            print(f"Embedding objective function for {row['outcome']}")
        else:
            print(f"Embedding constraints for {row['outcome']}")
        # For each outcome, call the related embedding function.  
        # Pass in the master model (to attach the constraints to) and the outcome-specific items from model-master
        # RF has one additional argument, the max_violation proportion
        mtype = row['model_type']
        mfile = pd.read_csv(row['save_path'])
        if mtype in ['cart', 'iai', 'iai-single']:
            constraints_tree(model, row['outcome'], mfile, 
                lb=row['lb'], ub=row['ub'],
                weight_objective=row['objective'], SCM=row['SCM_counterfactuals'], features=row['features'])
        elif mtype == 'rf':
            constraints_rf(model, row['outcome'], mfile, 
                lb=row['lb'], ub=row['ub'], 
                max_violation=max_violation, ## additional argument required for RF
                weight_objective=row['objective'], SCM=row['SCM_counterfactuals'], features=row['features'])
        elif mtype == 'gbm':
            constraints_gbm(model, row['outcome'], row['task'], mfile,
                lb=row['lb'], ub=row['ub'], weight_objective=row['objective'], SCM=row['SCM_counterfactuals'],
                            features=row['features'])
        elif mtype == 'mlp':
            constraints_mlp(model, data, row['outcome'], row['task'], mfile,
                lb=row['lb'], ub=row['ub'], 
                weight_objective=row['objective'], SCM=row['SCM_counterfactuals'], features=row['features'])
        elif mtype in ['linear', 'svm']:
            constraints_linear(model, row['outcome'], row['task'], mfile,
                lb=row['lb'], ub=row['ub'],
                weight_objective=row['objective'], SCM=row['SCM_counterfactuals'], features=row['features'])
    return model

def model_selection(performance, constraints_embed=[], objectives_embed={}, scores=False):
    '''
    Select models using aggregated 'performance' table. The models are selected based on the highest valid_score, assuming higher scores are better (true in sklearn).
    Objectives are passed through a dictionary with an assigned weight (value) per outcome name (key).
    Constraints are passed as a list. The upper and lower bounds must be set after generating the master model file.
    If 'scores' is true, return the validation scores in the master model file.
    '''
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
                           rename({'alg': 'model_type'}, axis=1).loc[:, ['outcome', 'model_type', 'save_path', 'task']]
    model_master['objective'] = model_master['outcome'].apply(lambda x: objectives_embed[x] \
        if x in objectives_embed.keys() \
        else 0)
    print(model_master)
    return model_master


def check_model_master(model_master, print_model = True):
    '''
    Check that the model_master file is valid: all outcomes must be unique, the model must be from a valid list, and a model cannot appear as both a constraint and objective.
    Optionally, print out the learned objectives and constraints in a readable format.
    '''
    if print_model == True:
        obj_cnt = sum(model_master['objective'] != 0)
        if obj_cnt == 0:
            print("No learned objective")
        else:
            obj = list(model_master.query('objective != 0')['outcome'])
            print(f"Learn objective for {obj}; will add to manually set objective.")

    ## Cannot have repeat outcome names
    assert len(model_master['outcome'].unique()) == len(model_master['outcome']), "Cannot use duplicate outcome names."
    for i, row in model_master.iterrows():
        ## ASSERT: must select from valid model list
        assert row['model_type'] in ['iai','iai-single','rf','cart','linear','svm','mlp', 'gbm'], "Invalid model type selected."
        if row['objective'] != 0:
            ## ASSERT: Model cannot be constrained and optimized
            assert (pd.isna(row['lb']) & pd.isna(row['ub'])), "Outcome cannot appear as both objective term and constraint."
            ## Print output for objective
            if print_model:
                print(f"\nEmbedding objective term for {row['outcome']} using {row['model_type']} model.")
                print(f"Outcome weight = {row['objective']}")
        else:
            ## Print output for constraint
            if print_model:
                constraint_lb = f"{round(row['lb'], 3)} <= " if not pd.isna(row['lb']) else ""
                constraint_ub = f" <= {round(row['ub'], 3)}" if not pd.isna(row['ub']) else ""
                if (constraint_lb + constraint_ub) != "":
                    print(f"\nEmbedding constraint for {row['outcome']} using {row['model_type']} model.")
                    print(constraint_lb + row['outcome'] + constraint_ub)
                    if row['task'] == 'binary':
                        print(f"The outcome '{row['outcome']}' is a probability")
                else: 
                    print(f"\Warning: {row['outcome']} does not appear as constraint or objective term.")