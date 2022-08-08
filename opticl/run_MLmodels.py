#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# module load python/3.6.3
# module load sloan/python/modules/3.6
# srun --pty --mem=16G -p sched_mit_sloan_interactive python3

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import matplotlib
# matplotlib.use('Agg')
# import shap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import pickle
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import opticl 

def r_squared(y_true, y_pred, y_mean):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_mean) ** 2).sum()
    return (1 - (ss_res / ss_tot))

def sens_spec(y_true, y_pred, threshold):
    y_pred_bin = 1 * (y_pred > threshold)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred_bin).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    print("Sensitivity: " + str(sens))
    print("Specificity: " + str(spec))
    return [sens, spec]

def create_and_save_pickle(gs, pickle_path):
    try:
        if type(gs).__name__ == 'ElasticNetCV':
            exp = {'model': gs,
                   'best_params': gs.best_params_,
                   'param_grid': gs.param_grid}
        else:
            exp = {'model': gs.best_estimator_,
                   'best_params': gs.best_params_,
                   'param_grid': gs.param_grid}
    except:
        exp = {'model': gs}
    with open(pickle_path, 'wb') as handle:
        pickle.dump(exp, handle, protocol=4)
    return


def shap_summary(model, X, col_names, save_path, filetype='.pdf'):
    plt.close()
    explainer = shap.TreeExplainer(model,
                                   # data=test_x, model_output="probability",
                                   );
    shap_values = explainer.shap_values(X);
    if len(shap_values) == 2:
        shap_values = shap_values[1]

    importance = pd.DataFrame(list(zip(col_names, np.mean(abs(shap_values), axis=0))),
                              columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=False)
    importance.to_csv(save_path + '_importance.csv', index=False)

    shap.summary_plot(shap_values, X, show=False,
                      max_display=10,
                      plot_size=(10, 5),
                      plot_type="violin",
                      feature_names=list(col_names))
    f = plt.gcf()
    plt.xlabel('SHAP value (impact on model output)')
    f.savefig(save_path + '_summary_plot' + filetype,
              bbox_inches='tight'
              )
    plt.clf()
    plt.close()


def initialize_model(model_choice, task, cv_folds, parameter_grid, gs_metric, seed, mlp_solver='adam'):
    ## select scoring metric
    if gs_metric == None:
        if task == 'binary':
            gs_metric = 'roc_auc'
        elif task == 'multiclass':
            gs_metric = 'roc_auc_ovr'
        elif task == 'continuous':
            gs_metric = 'neg_mean_squared_error'
            # gs_metric = 'r2'

    if model_choice == "linear":
        if task == 'binary':
            from sklearn.linear_model import LogisticRegression
            param_grid = {'C': np.arange(0.001, 1, 0.05), 'penalty': ['l2', 'l1']}
            est = LogisticRegression(random_state=seed, solver='saga', max_iter=1e4)
        elif task == 'multiclass':
            from sklearn.linear_model import LogisticRegression
            param_grid = parameter_grid if parameter_grid is not None else {'C': np.arange(0.001, 1, 0.05),
                                                                            'penalty': ['l2', 'l1'], 'max_iter': [1e4]}
            est = LogisticRegression(random_state=seed, multi_class='multinomial', solver='saga', max_iter=1e4)
        elif task == 'continuous':
            from sklearn.linear_model import ElasticNet
            param_grid = parameter_grid if parameter_grid is not None else {'alpha': [0.1, 1, 10, 100, 1000],
                                                                            'l1_ratio': np.arange(0.1, 1.0, 0.1)}
            est = ElasticNet(random_state=seed, max_iter=1e4)
        gs = GridSearchCV(estimator=est, param_grid=param_grid, scoring=gs_metric, cv=cv_folds)

    elif model_choice == "cart":
        from sklearn.tree import DecisionTreeClassifier
        param_grid = parameter_grid if parameter_grid is not None else {"max_depth": [3, 4, 5, 6, 7, 8, 9, 10],
                                                                        'min_samples_leaf': [0.02, 0.04, 0.06],
                                                                        "max_features": [0.4, 0.6, 0.8, 1.0]}
        if task in ['binary', 'multiclass']:
            from sklearn.tree import DecisionTreeClassifier
            est = DecisionTreeClassifier(random_state=seed, criterion='gini')
        elif task == 'continuous':
            from sklearn.tree import DecisionTreeRegressor
            est = DecisionTreeRegressor(random_state=seed)
        gs = GridSearchCV(estimator=est, param_grid=param_grid, scoring=gs_metric, cv=cv_folds)

    elif model_choice in ["rf", "rf_shallow"]:
        if model_choice == "rf":
            param_grid = parameter_grid if parameter_grid is not None else {
                'n_estimators': [250, 500],
                'max_features': ['auto'],
                'max_depth': [6, 7, 8]
            }
        else:
            param_grid = parameter_grid if parameter_grid is not None else {
                'n_estimators': [10, 25],
                'max_features': ['auto'],
                'max_depth': [2, 3, 4]
            }

        if task in ['binary', 'multiclass']:
            from sklearn.ensemble import RandomForestClassifier
            est = RandomForestClassifier(random_state=seed, criterion='gini')
        elif task == 'continuous':
            from sklearn.ensemble import RandomForestRegressor
            est = RandomForestRegressor(random_state=seed)
        gs = GridSearchCV(estimator=est, param_grid=param_grid, scoring=gs_metric, cv=cv_folds)

    elif model_choice == 'gbm':
        param_grid = parameter_grid if parameter_grid is not None else {
            "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
            "max_depth": [2, 3, 4, 5],
            "n_estimators": [20]
        }
        if task == 'binary':
            from sklearn.ensemble import GradientBoostingClassifier
            est = GradientBoostingClassifier(random_state=seed, init='zero')
        elif task == 'multiclass':
            from sklearn.ensemble import GradientBoostingClassifier
            est = GradientBoostingClassifier(random_state=seed, init='zero')
        elif task == 'continuous':
            from sklearn.ensemble import GradientBoostingRegressor
            est = GradientBoostingRegressor(random_state=seed)
        gs = GridSearchCV(estimator=est, param_grid=param_grid, scoring=gs_metric, cv=cv_folds)

    elif model_choice == "xgb":
        param_grid = parameter_grid if parameter_grid is not None else {
            'min_child_weight': [1, 5, 10],
            'gamma': [0.5, 1, 2, 5, 10],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'max_depth': [4, 5, 6],
            'n_estimators': [250]
        }
        if task == 'binary':
            from xgboost import XGBClassifier
            est = XGBClassifier(random_state=seed, objective='binary:logistic')
        elif task == 'multiclass':
            from xgboost import XGBClassifier
            est = XGBClassifier(random_state=seed, objective='multi:softmax')
        elif task == 'continuous':
            from xgboost import XGBRegressor
            est = XGBRegressor(random_state=seed)
        gs = GridSearchCV(estimator=est, param_grid=param_grid, scoring=gs_metric, cv=cv_folds)

    elif model_choice == "iai":
        from julia import Julia
        Julia(compiled_modules=False)
        from interpretableai import iai
        if task in ['binary', 'multiclass']:
            est = iai.OptimalTreeClassifier(
                random_seed=seed,
                ls_num_hyper_restarts=5,  # 5 is default
                fast_num_support_restarts=10,
                hyperplane_config={'sparsity': 'sqrt'}
            )
        elif task == 'continuous':
            est = iai.OptimalTreeRegressor(
                random_seed=seed,
                ls_num_hyper_restarts=5,  # 5 is default
                fast_num_support_restarts=10,
                hyperplane_config={'sparsity': 'sqrt'}
            )
        gs = iai.GridSearch(est,
                            max_depth=range(2, 6), minbucket=[5, 10]
                            )
    elif model_choice == "iai-single":
        from julia import Julia
        Julia(compiled_modules=False)
        from interpretableai import iai
        if task in ['binary', 'multiclass']:
            est = iai.OptimalTreeClassifier(
                random_seed=seed,
            )
        elif task == 'continuous':
            est = iai.OptimalTreeRegressor(
                random_seed=seed,
            )
        gs = iai.GridSearch(est,
                            max_depth=range(2, 6), minbucket=[.01, .02, .05]
                            )
    elif model_choice == "svm":
        param_grid = parameter_grid if parameter_grid is not None else {
            'C': [.1, 1, 10, 100]
        }
        if task in ['binary', 'multiclass']:
            from sklearn.svm import LinearSVC
            est = LinearSVC(max_iter=1e5, dual=False, penalty='l2')
        elif task == 'continuous':
            from sklearn.svm import LinearSVR
            est = LinearSVR(max_iter=1e5, dual=False, loss='squared_epsilon_insensitive')
        gs = GridSearchCV(estimator=est, param_grid=param_grid, scoring=gs_metric, cv=cv_folds)

    elif model_choice == "mlp":
        param_grid = parameter_grid if parameter_grid is not None else {
            'hidden_layer_sizes': [(10,), (20,), (50,), (100,)]
        }
        if task in ['binary', 'multiclass']:
            assert task == 'binary', 'sorry, the multiclass is under development'
            from sklearn.neural_network import MLPClassifier
            est = MLPClassifier(random_state=seed, solver=mlp_solver, max_iter=10000)
        elif task == 'continuous':
            from sklearn.neural_network import MLPRegressor
            est = MLPRegressor(random_state=seed, solver=mlp_solver, max_iter=10000)
        gs = GridSearchCV(estimator=est, param_grid=param_grid, scoring=gs_metric, cv=cv_folds)
    return gs


def run_model(train_x, y_train, test_x, y_test, model_choice, outcome, task, cv_folds=3,
              seed=1, save_path='../results/', save=False, shap=False,
              # weights = []
              parameter_grid=None,
              metric=None
              ):
    assert task in ['multiclass', 'binary', 'continuous']
    assert model_choice in ['linear', 'cart', 'rf', 'rf_shallow', 'xgb', 'gbm', 'iai', 'iai-single', 'svm', 'mlp']
    # Path(save_path).mkdir(parents=True, exist_ok=True)

    print("------------- Initialize grid  ----------------")
    mlp_solver = 'adam' if train_x.shape[0] >= 1000 else 'lbfgs'
    gs = initialize_model(model_choice, task, cv_folds,
                          parameter_grid, metric, seed, mlp_solver=mlp_solver)

    print("------------- Running model  ----------------")
    print(f"Algorithm = {model_choice}, metric = {metric}")
    np.random.seed(seed)
    if (model_choice == 'iai') | (model_choice == 'iai-single'):
        metric = 'mse' if task == 'continuous' else 'auc'
        gs.fit_cv(train_x, y_train, n_folds=cv_folds, validation_criterion=metric)
    else:
        gs.fit(train_x, y_train)

    filename = 'results/' + model_choice + '_' + outcome + '_trained.pkl'
    with open(filename, 'wb') as f:
        print(f'saving... {filename}')
        pickle.dump(gs.best_estimator_, f)
        # if len(weights) > 0:
        #     print("Applying sample weights")
        #     gs.fit(train_x, y_train, sample_weight = weights)
        # else:

    if (model_choice == 'iai') | (model_choice == 'iai-single'):
        grid_result = gs.get_grid_results()
        valid_score = grid_result.query('rank_valid_score == 1')['mean_valid_score'].values[0]
        best_params = gs.get_best_params()
        param_grid = {'minbucket': grid_result['minbucket'].unique(),
                      'max_depth': grid_result['max_depth'].unique()}
        model = gs.get_learner()
    else:
        valid_score = gs.best_score_
        best_params = gs.best_params_
        param_grid = gs.param_grid
        model = gs.best_estimator_

    print("------------- Model evaluation  ----------------")
    if task == 'binary':
        if model_choice != 'svm':
            print("-------------------training evaluation-----------------------")
            train_pred = np.array(gs.predict_proba(train_x))[::, 1]
            train_score = metrics.roc_auc_score(y_train, train_pred)
            print("Train Score: " + str(train_score))

            print("-------------------testing evaluation-----------------------")
            test_pred = np.array(gs.predict_proba(test_x))[::, 1]
            test_score = metrics.roc_auc_score(y_test, test_pred)
            print("Test Score: " + str(test_score))

            preds_train = pd.DataFrame({'true': y_train, 'pred': train_pred})
            preds_test = pd.DataFrame({'true': y_test, 'pred': test_pred})

            performance_dict = {'save_path': save_path, 'seed': seed,
                                'cv_folds': cv_folds, 'task': task, 'parameters': param_grid,
                                'best_params': best_params,
                                'valid_score': valid_score, 'train_score': train_score, 'test_score': test_score}
        else:
            print("-------------------training evaluation-----------------------")
            train_pred = gs.predict(train_x)
            train_score = gs.score(train_x, y_train)
            print("Train Score: " + str(train_score))

            print("-------------------testing evaluation-----------------------")
            test_pred = gs.predict(test_x)
            test_score = gs.score(test_x, y_test)
            print("Test Score: " + str(test_score))

            preds_train = pd.DataFrame({'true': y_train, 'pred': train_pred})
            preds_test = pd.DataFrame({'true': y_test, 'pred': test_pred})

            performance_dict = {'save_path': save_path, 'seed': seed,
                                'cv_folds': cv_folds, 'task': task, 'parameters': param_grid,
                                'best_params': best_params,
                                'valid_score': valid_score, 'train_score': train_score, 'test_score': test_score}

    elif task == 'multiclass':
        print("-------------------training evaluation-----------------------")
        train_pred = gs.predict_proba(train_x)
        train_score = metrics.roc_auc_score(y_train, train_pred, multi_class='ovr')
        print("Train Score: " + str(train_score))

        print("-------------------testing evaluation-----------------------")
        test_pred = gs.predict_proba(test_x)
        test_score = metrics.roc_auc_score(y_test, test_pred, multi_class='ovr')
        print("Test Score: " + str(test_score))

        preds_train = pd.DataFrame(train_pred, columns=gs.classes_);
        preds_train['true'] = y_train
        preds_test = pd.DataFrame(test_pred, columns=gs.classes_);
        preds_test['true'] = y_test

        performance_dict = {'save_path': save_path, 'seed': seed,
                            'cv_folds': cv_folds, 'task': task, 'parameters': param_grid, 'best_params': best_params,
                            'valid_score': valid_score, 'train_score': train_score, 'test_score': test_score}

    elif task == 'continuous':
        print("-------------------training evaluation-----------------------")
        train_pred = gs.predict(train_x)
        train_mse = metrics.mean_squared_error(y_train, train_pred)
        print("Train MSE: " + str(train_mse))
        train_r2 = r_squared(y_train, train_pred, y_train.mean())
        print("Train R2: " + str(train_r2))

        print("-------------------testing evaluation-----------------------")
        test_pred = gs.predict(test_x)
        test_mse = metrics.mean_squared_error(y_test, test_pred)
        print("Test MSE: " + str(test_mse))
        test_r2 = r_squared(y_test, test_pred, y_train.mean())
        print("Test R2: " + str(test_r2))

        preds_train = pd.DataFrame({'true': y_train, 'pred': train_pred})
        preds_test = pd.DataFrame({'true': y_test, 'pred': test_pred})

        performance_dict = {'save_path': save_path, 'seed': seed,
                            'cv_folds': cv_folds,
                            'task': task, 'parameters': param_grid, 'best_params': best_params,
                            'valid_score': valid_score,
                            'train_score': train_mse, 'train_r2': train_r2,
                            'test_score': test_mse, 'test_r2': test_r2}
    performance = pd.DataFrame([list(performance_dict.values())], columns=performance_dict.keys(), index=[0])
    if save:
        print("------------- Save results  ----------------")
        if not os.path.exists('results/%s/' % model_choice):
            os.makedirs('results/%s/' % model_choice)
        performance.to_csv('results/%s/%s_performance.csv' % (model_choice, outcome), index=False)
        # preds_train.to_csv(save_path+"_trainprobs.csv")
        # preds_test.to_csv(save_path+"_testprobs.csv")

        if model_choice == 'cart':
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 15), dpi=200)
            tree.plot_tree(model,
                           feature_names=train_x.columns,
                           filled=True)
            fig.savefig(save_path + '_tree.png')
            plt.clf()
            plt.close()
        elif model_choice == 'iai':
            df_A, node_info = extract_tree_info(model, train_x.columns, save_path)
            model.write_html(save_path + '_tree.html')
            model.write_png(save_path + '_tree.png')
        elif model_choice == 'linear':
            coef = pd.DataFrame(model.coef_.transpose(),
                                columns=gs.classes_, index=train_x.columns)
            coef.to_csv(save_path + "_coefficients.csv", index=True)
        elif shap:
            shap_summary(model, train_x, train_x.columns, save_path)

        # create_and_save_pickle(gs, save_path+".pkl")

    return model, performance


def train_ml_models(outcome_list, version, s = 1, bootstrap_proportion = 0.5, save_models = False, save_path = 'results/'):
    performance = pd.DataFrame()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for outcome_main in outcome_list.keys():
        print(f'Learning a model for {outcome_main}')
        outcome_specs = outcome_list[outcome_main]
        alg_list = outcome_specs['alg_list']
        task_type = outcome_specs['task_type']
        bootstrap_iterations = outcome_specs['bootstrap_iterations']
        bootstrap_yn = True if bootstrap_iterations > 0 else 0
        ## Iterate over bootstrap iterations (or single loop if none)
        X_train_all = outcome_specs['X_train']
        y_train_all = outcome_specs['y_train']
        X_test = outcome_specs['X_test']
        y_test = outcome_specs['y_test']
        for i in range(max(1,bootstrap_iterations)):
            if not bootstrap_yn: 
                print("No bootstrap - training on full training data")
                outcome = outcome_main
                X_train = X_train_all
                y_train = y_train_all
            else:
                print("Bootstrap iteration %d of %d" % (i+1, bootstrap_iterations))
                ## If bootstrapping, save outcome with subscript
                outcome = outcome_main + '_s%d' % i
                bs_sample = int(bootstrap_proportion*X_train_all.shape[0])
                X_train, y_train = resample(X_train_all, y_train_all,
                    replace = True, n_samples = bs_sample, random_state=i)
            for alg in alg_list:
                print(f'training {outcome} with {alg}')
                if not os.path.exists(save_path+alg+'/'):
                    os.makedirs(save_path+alg+'/')
                ## Run shallow/small version of RF
                alg_run = 'rf_shallow' if alg == 'rf' else alg
                m, perf = run_model(X_train, y_train, X_test, y_test, alg_run, outcome, task = task_type,
                                       seed = s, cv_folds = 5, 
                                       # metric = 'r2',
                                       save = False
                                      )
                ## Save model
                constraintL = opticl.ConstraintLearning(X_train, y_train, m, alg)
                constraint_add = constraintL.constraint_extrapolation(task_type)
                constraint_add.to_csv(save_path+'/%s/%s_%s_model.csv' % (alg, version, outcome), index = False)
                ## Extract performance metrics
                try:
                    threshold = outcome_specs['lb'] or outcome_specs['ub']
                    perf['auc_threshold'] = threshold
                    perf['auc_train'] = metrics.roc_auc_score(y_train >= threshold, m.predict(X_train))
                    perf['auc_test'] = metrics.roc_auc_score(y_test >= threshold, m.predict(X_test))
                except: 
                    perf['auc_threshold'] = np.nan
                    perf['auc_train'] = np.nan
                    perf['auc_test'] = np.nan
                perf['seed'] = s
                perf['outcome'] = outcome_main
                perf['outcome_label'] = outcome
                perf['alg'] = alg
                perf['bootstrap_iteration'] = i
                perf['save_path'] = save_path+'%s/%s_%s_model.csv' % (alg, version, outcome) 
                perf.to_csv(save_path+'%s/%s_%s_performance.csv' % (alg, version, outcome), index = False)
                performance = performance.append(perf)
                print()
    print('Saving the performance...')
    # performance.to_csv(save_path+'%s_performance.csv' % version, index = False)
    print('Done!')
    return performance
