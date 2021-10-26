from sklearn import tree
import pandas as pd
import numpy as np
import pickle
import itertools

from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn import metrics

## Linear model export utilities: optionally "unscale" coefficients
## Usage: scaler_info = pd.read_csv('../processed-data/gastric_train2008_scaler_info.csv'); 
##        coef_rescale = rescale_coefficients(m, X_train.columns, scaler_info)

def process_coefficients(m, feature_names, scaler_info = None):

    coef = pd.DataFrame({'feature': feature_names, 'coefficient_raw': m.coef_})

    if scaler_info == None:
        print("Warning: no scaler information provided - returning original coefficients")
        coef.loc[:,'coefficient'] = coef.loc[:,'coefficient_raw']
        coef = coef.append(pd.DataFrame({'feature': ['intercept'],
                                'coefficient_raw': [m.intercept_],
                                'coefficient': [m.intercept_]}))
    else:
        coef = coef.merge(scaler_info, left_on = 'feature', right_on = 'feature', how = 'left')
        coef.loc[:,'coefficient'] = (coef['coefficient_raw']/np.sqrt(coef['var'])).fillna(0)

        coef.loc[:,'adj'] = (coef['coefficient_raw']*coef['mean']/np.sqrt(coef['var'])).fillna(0)

        intercept_unscale = m.intercept_ - sum(coef['adj'])

        ## Append intercept
        coef = coef.append(pd.DataFrame({'feature': ['intercept'],
                                        'coefficient_raw': [m.intercept_],
                                        'coefficient': [m.intercept_ - sum(coef['adj'])]}))
    
    return coef

## Tree export utilities: save node and split information, as well as plots of trees 
## Usage: either export_tree(m) for CART or export_tree(m.estimators_[idx]) for RF

def shift_index(idx):
    if idx >= 0:
        return idx + 1
    else:
        return -2

def find_split(clf, node):
    a = np.zeros(clf.n_features_)
    if node == -2:
        return a, 0
    elif clf.tree_.children_left[node] == clf.tree_.children_right[node]:# for recursion with descendents case
        return a, 0
    else: 
        ft = clf.tree_.feature[node]
        b = clf.tree_.threshold[node]
        a[ft] = 1
        return a, b
    
def find_parent(node, df):
    # Return row (= idx)
    row = np.where(df.loc[:,["child_left","child_right"]] == node)[0]
    if row.size == 0: 
        return -2
    else:
        return df.loc[row.item(),'id']


def find_descendants(clf, parent_node, direction = 'all', return_type = 'array'):
    if direction == 'left':
        stack = [clf.tree_.children_left[parent_node]]
    elif direction == 'right':
        stack = [clf.tree_.children_right[parent_node]]
    else: 
        stack = [clf.tree_.children_left[parent_node],clf.tree_.children_right[parent_node]]
    children = []
    idx=0
    while idx <= len(stack)-1:
        node = stack[idx]
        if node != -1:
            stack.append(clf.tree_.children_left[node]) 
            stack.append(clf.tree_.children_right[node])
        idx += 1
    desc_list = [y for y in stack if y != -1]
    if return_type == 'array':
        desc_array = np.zeros(clf.tree_.node_count)
        desc_array[desc_list] = 1
        return desc_array
    else:
        return desc_list

def tree_to_table(clf, features):
    node_table = pd.DataFrame(columns = ['id','node_type','child_left','child_right','pred','split_a','split_b'])
    #split_table = # 'split_a','split_b',
    for i in range(clf.tree_.node_count):
#         print(i)
        pred_val = clf.tree_.value[i].item()
        a, b = find_split(clf, i)
        child_left = shift_index(clf.tree_.children_left[i])
        child_right = shift_index(clf.tree_.children_right[i])
        if child_left!=child_right:
            split_type = 'split'
        else:
            split_type = 'leaf'
        
        node_table.loc[i,:] = np.array([shift_index(i),split_type,child_left,child_right,pred_val,a,b], dtype = 'object')
    
    ## parent index does not need shift - already pulling from shifted valaues
    node_table['parent'] = node_table['id'].apply(lambda idx: find_parent(idx, node_table))
    
    split_info = pd.DataFrame(np.stack([i for i in node_table['split_a']], axis=0), columns = features)
    split_info['b'] = node_table['split_b']

    node_info = node_table.loc[:,['id','node_type','child_left','child_right','parent','pred']]

    return node_info, split_info

def extract_tree_info(lnr, features, save_path):

    node_count = lnr.get_num_nodes()
    A = pd.DataFrame(np.zeros((node_count, len(features))))
    A.columns = features

    b = np.zeros(node_count)

    node_info = pd.DataFrame(columns = ['id','node_type','child_left','child_right','parent','pred'])

    for i in range(0, node_count):
        node_id = i+1 # shift from julia
        ## get general (not )
        if lnr.is_hyperplane_split(node_id):
            splits = lnr.get_split_weights(node_id)
            assert (len(splits) == 1) | (splits[1] == {})
            for key in splits[0].keys():
                A.loc[i, key] = splits[0][key]
            b_add = lnr.get_split_threshold(node_id)
            ## get node-specific info
            node_type = 'split'
            parent = lnr.get_parent(node_id) if lnr.get_depth(node_id) > 0 else -2
            child_left = lnr.get_lower_child(node_id)
            child_right = lnr.get_upper_child(node_id)
            pred = np.nan
        elif lnr.is_parallel_split(node_id):
            key = lnr.get_split_feature
            A.loc[i,lnr.get_split_feature(node_id)] = 1
            b_add = lnr.get_split_threshold(node_id)
            ## get node-specific info
            node_type = 'split'
            parent = lnr.get_parent(node_id) if lnr.get_depth(node_id) > 0 else -2
            child_left = lnr.get_lower_child(node_id)
            child_right = lnr.get_upper_child(node_id)
            pred = np.nan
        else:
            # If not parallel or hyperplane, check that node is leaf. Otherwise A will be incorrectly constructed.
            assert lnr.is_leaf(node_id)
            node_type = 'leaf'
            parent = lnr.get_parent(node_id) if lnr.get_depth(node_id) > 0 else -2
            child_left = -2
            child_right = -2
            pred = lnr.get_regression_constant(node_id)
        node_info.loc[i,:] = [node_id, node_type, child_left, child_right, parent, pred]

    A['b'] = b
    node_info.to_csv(save_path+'_node_info.csv', index = False)
    A.to_csv(save_path+'_splits.csv', index = False)

    return A, node_info

    
def export_tree(model, features, save_path = None):
    node_info, split_info = tree_to_table(model, features)
   
    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (15,15), dpi=200)
    tree.plot_tree(model,
                    feature_names = features, 
                    filled = True);
    if save_path != None:
        # Save figure
        fig.savefig(save_path+'_tree.png')
        # Save tree attributes for MIP
        node_info.to_csv(save_path+'_node_info.csv', index = False)
        split_info.to_csv(save_path+'_splits.csv', index = False)
    return node_info, split_info
        
