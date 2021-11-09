import pandas as pd
import numpy as np


'''
ContraintLearning class contains a set of functions for the
extrapolation of constraints from a trained Optimal Classification Tree
(with and without hyperplanes). 
'''
class ConstraintLearning(object):
    '''
    The initialization requires:
    X: dataset without the target column
    y: target column
    grid: learned model.
    '''
    def __init__(self, X, y, learner, algorithm):
        if not algorithm in ['rf','cart','linear','svm','mlp', 'gbm', 'iai', 'iai-single']: #error!
            raise ValueError("invalid algorithm!") 

        self.__data = X
        self.__label = y
        self.__learner = learner
        self.__algorithm = algorithm

    def get_features_list(self):
        '''
        Returns the list of features that describe the dataset.
        '''
        return list(self.__data.columns)

    def opt_data_reduction(self, datapoint, data):
        '''
        This function is called by data_reduction(). It checks if
        the datapoint is within the convex hull solving an optimziation model.
        '''
        model = ConcreteModel()
        model.I = RangeSet(0, data.shape[0] - 1)
        model.J = RangeSet(0, data.shape[1] - 1)

        def x_init(model, i, j):
            return data[i][j]

        def d_init(model, i):
            return datapoint[i]

        model.x = Param(model.I, model.J, initialize=x_init)
        model.d = Param(model.J, initialize=d_init)
        model.l = Var(model.I, domain=NonNegativeReals)

        def obj_function(model):
            return 1

        model.OBJ = Objective(rule=obj_function, sense=minimize)

        def constraint_rule1(model):
            return sum(model.l[i] for i in model.I) == 1

        def constraint_rule2(model, j):
            return sum(model.l[i] * model.x[i, j] for i in model.I) == model.d[j]

        model.Constraint1 = Constraint(rule=constraint_rule1)
        model.Constraint2 = Constraint(model.J, rule=constraint_rule2)

        opt = SolverFactory('cplex')
        results = opt.solve(model)

        # print('Termination condition:', results['Solver']._list[0]['Termination condition'].key)
        if results['Solver']._list[0]['Termination condition'].key != 'optimal':
            return True
        else:
            return False

    def data_reduction(self):
        '''
        This function delete samples that are not verteces of the convex hull.
        '''
        data_reduced = pd.DataFrame(columns=self.__data.columns)
        c = 0
        while c < self.__data.shape[0]:
            datapoint = self.__data[c:c + 1]
            data = np.delete(self.__data, datapoint, axis=0)
            result = self.opt_data_reduction(datapoint[0], data)
            if result:
                data_reduced.append(datapoint)
            c += 1
        return data_reduced

    def __leaf_nodes_iai(self, class_c):
        '''
        It returns a list of node corresponding to the leaves of the OCT(-H)
        '''
        leaf_nodes = []
        num_nodes = self.__learner.get_num_nodes()
        '''
        find the node_indices of the leaves
        '''
        if class_c in ['continuous','binary']:
            for node in range(1, num_nodes + 1):
                if self.__learner.is_leaf(node_index=node):
                    leaf_nodes.append(node)
        # else:
        #     for node in range(1, num_nodes + 1):
        #         if self.__learner.is_leaf(node_index=node):
        #             if self.__learner.get_classification_label(node) == class_c:
        #                 leaf_nodes.append(node)
        return leaf_nodes

    def constraint_extrapolation_iai(self, class_c):
        '''
        It returns a matrix with all the constraints that describe the tree.
        This function returns constraints for each leaf. the column ID represent the reference leaf.
        IMPORTANT: (constraint structure) the sign is always <=.
        '''

        columns = ['ID'] + self.get_features_list() + ['threshold', 'prediction']
        leaf_nodes = self.__leaf_nodes_iai(class_c)

        '''
        Initialize dataframe where:
        ID: represent the same set of constraint that describes a leaf_nodes
        features: coefficient for each feature of the data. If coeff = 0 the feature
        is not in the constraints.
        threshold: right-hand side of the constraints.
        '''
        constraints = pd.DataFrame(columns=columns)
        ID = 1
        for leaf in leaf_nodes:
            node = leaf
            stop = False
            parent_node = self.__learner.get_parent(node_index=node)
            while not stop:
                constraint = pd.DataFrame(data=np.zeros(len(columns)).reshape(1,-1), columns=columns)
                # sign true means that the inequality constraint is Ax<=b otherwise A>b
                sign = True if self.__learner.get_lower_child(node_index=parent_node) == node else False
                threshold = self.__learner.get_split_threshold(node_index=parent_node)
                constraint['threshold'] = threshold-0.000001 if sign else -threshold

                # .is_hyperplane_split is TRUE if the split is a non-orthogonal hyperplane
                if self.__learner.is_hyperplane_split(node_index=parent_node):
                    dict_weights = self.__learner.get_split_weights(node_index=parent_node)
                    for key in dict_weights[0]:
                        if sign:
                            constraint[key] = self.__learner.get_split_weights(node_index=parent_node)[0][key]
                        else:
                            constraint[key] = -self.__learner.get_split_weights(node_index=parent_node)[0][key]
                else:
                    if sign:
                        constraint[self.__learner.get_split_feature(node_index=parent_node)] = 1
                    else:
                        constraint[self.__learner.get_split_feature(node_index=parent_node)] = -1
                constraint['ID'] = ID
                if class_c == 'continuous':
                    constraint['prediction'] = self.__learner.get_regression_constant(leaf)
                elif class_c == 'binary':
                    constraint['prediction'] = self.__learner.get_classification_proba(leaf)[1] # assume label '1' is class of interest
                elif class_c == 'multiclass':
                    print('Under Development')

                constraints = constraints.append(constraint)

                node = parent_node
                # check if the parent_node is a root node
                if node == 1:
                    stop = True
                else:
                    parent_node = self.__learner.get_parent(node_index=node)
            ID += 1

        return constraints

    def __find_path_skTree(self, node_numb, path, leaf, children_left, children_right):
        '''
        This function is used to find the path of nodes that are visited before reaching a leaf
        '''
        path.append(node_numb)
        if node_numb == leaf:
            return True
        left = False
        right = False
        if (children_left[node_numb] != -1):
            left = self.__find_path_skTree(children_left[node_numb], path, leaf, children_left, children_right)
        if (children_right[node_numb] != -1):
            right = self.__find_path_skTree(children_right[node_numb], path, leaf, children_left, children_right)
        if left or right:
            return True
        path.remove(node_numb)
        return False

    def __get_rule_skTree(self, leaf, path, column_names, columns, ID, type_tree, children_left, feature, threshold):
        '''
        This functions transform the list of nodes composing a path into a set of constraints
        '''
        constraints_leaf = pd.DataFrame(columns=columns)
        for index, node in enumerate(path):
            constraint = pd.DataFrame(data=np.zeros(len(columns)).reshape(1, -1), columns=columns)
            # We check if we are not in the leaf
            if node != leaf:
                # Do we go under or over the threshold ?
                if (children_left[node] == path[index + 1]):
                    constraint[column_names[feature[node]]] = 1
                    constraint['threshold'] = threshold[node]
                else:
                    constraint[column_names[feature[node]]] = -1
                    constraint['threshold'] = -(threshold[node] + 0.000001)
                constraint['ID'] = ID
                constraints_leaf = constraints_leaf.append(constraint)
        return constraints_leaf

    def constraint_extrapolation_skTree(self, class_c):
        '''
        :param class_c: either r: regression or c: classification
        :return: set of constraints that describe each leaf of the tree.
        Constraints with the same ID describe the same leaf and must be satisfied together.
        '''
        children_left = self.__learner.tree_.children_left
        children_right = self.__learner.tree_.children_right
        feature = self.__learner.tree_.feature
        threshold = self.__learner.tree_.threshold

        # Leaves
        leave_id = self.__learner.apply(self.__data)
        if class_c == 'multiclass':
            columns_classes = [f'prediction_class_{i}' for i in range(len(self.__learner.tree_.value[leave_id[0]][0]))]
            columns = ['ID'] + [feature for feature in self.get_features_list()] + ['threshold'] + columns_classes
        else:
            columns = ['ID'] + [feature for feature in self.get_features_list()] + ['threshold', 'prediction']
        constraints = pd.DataFrame(columns=columns)
        for i, leaf in enumerate(np.unique(leave_id)):
            path_leaf = []
            self.__find_path_skTree(0, path_leaf, leaf, children_left, children_right)
            constraints_leaf = self.__get_rule_skTree(leaf, path_leaf, self.get_features_list(), columns, i + 1, class_c, children_left, feature, threshold)
            if class_c == 'continuous':
                constraints_leaf['prediction'] = self.__learner.tree_.value[leaf].item()
            elif class_c == 'binary':
                constraints_leaf['prediction'] = self.__learner.tree_.value[leaf][0, 1]/sum(self.__learner.tree_.value[leaf][0])
                # constraints_leaf['prediction'] = np.round(self.__learner.tree_.value[leaf].item())
            elif class_c == 'multiclass':
                # for i, class_name in enumerate(columns_classes):
                #     constraints_leaf[class_name] = self.__learner.tree_.value[leaf][0, i]/sum(self.__learner.tree_.value[leaf][0])
                print('Under Development')
            constraints = constraints.append(constraints_leaf)

        return constraints

    def constraint_extrapolation_SVM(self, class_c):
        '''
        :return: constraint: it has the following structure: Coeff*x+intercept >= 0
        '''
        if class_c == "continuous":
            columns = [feature for feature in self.get_features_list()]
            constraint = pd.DataFrame(data=[self.__learner.coef_], columns=columns)
            constraint['intercept'] = self.__learner.intercept_
        elif class_c == "binary":
            columns = [feature for feature in self.get_features_list()]
            constraint = pd.DataFrame(data=[self.__learner.coef_[0]], columns=columns) ## only one element of coefficient array of arrays
            constraint['intercept'] = self.__learner.intercept_[0]
        return constraint

    def constraint_extrapolation_skRF(self, class_c):
        columns = ['Tree_id', 'ID'] + [feature for feature in self.get_features_list()] + ['threshold', 'prediction']
        constraints = pd.DataFrame(columns=columns)
        for tree_id, tree in enumerate(self.__learner):
            children_left = tree.tree_.children_left
            children_right = tree.tree_.children_right
            feature = tree.tree_.feature
            threshold = tree.tree_.threshold

            # Leaves
            leave_id = tree.apply(self.__data)

            for i, leaf in enumerate(np.unique(leave_id)):
                path_leaf = []
                self.__find_path_skTree(0, path_leaf, leaf, children_left, children_right)
                constraints_leaf = self.__get_rule_skTree(leaf, path_leaf, self.get_features_list(), columns[:-1], i + 1, class_c, children_left, feature, threshold)
                constraints_leaf['Tree_id'] = tree_id
                if class_c == 'continuous':
                    constraints_leaf['prediction'] = tree.tree_.value[leaf].item()
                elif class_c == 'binary':
                    constraints_leaf['prediction'] = float(tree.tree_.value[leaf][0, 1]/sum(tree.tree_.value[leaf][0]))
                    # constraints_leaf['prediction'] = np.round(self.__learner.tree_.value[leaf].item())
                elif class_c == 'multiclass':
                    # for i, class_name in enumerate(columns_classes):
                    #     constraints_leaf[class_name] = self.__learner.tree_.value[leaf][0, i]/sum(self.__learner.tree_.value[leaf][0])
                    print('Under Development')
                constraints = constraints.append(constraints_leaf)

        return constraints

    def constraint_extrapolation_skGBM(self, class_c):
        columns = ['Tree_id', 'ID'] + [feature for feature in self.get_features_list()] + ['threshold', 'prediction', 'initial_prediction', 'learning_rate']
        constraints = pd.DataFrame(columns=columns)
        for tree_id, tree_array in enumerate(self.__learner.estimators_):
            tree = tree_array.item()
            children_left = tree.tree_.children_left
            children_right = tree.tree_.children_right
            feature = tree.tree_.feature
            threshold = tree.tree_.threshold

            # Leaves
            leave_id = tree.apply(self.__data)

            for i, leaf in enumerate(np.unique(leave_id)):
                path_leaf = []
                self.__find_path_skTree(0, path_leaf, leaf, children_left, children_right)
                constraints_leaf = self.__get_rule_skTree(leaf, path_leaf, self.get_features_list(), columns[:-1], i + 1, class_c, children_left, feature, threshold)
                constraints_leaf['Tree_id'] = tree_id
                if class_c == 'continuous':
                    # print(tree.tree_.value[leaf])
                    constraints_leaf['prediction'] = tree.tree_.value[leaf].item()
                    constraints_leaf['initial_prediction'] = self.__learner.init_.constant_.item()
                    constraints_leaf['learning_rate'] = self.__learner.learning_rate
                else:
                    constraints_leaf['prediction'] = tree.tree_.value[leaf].item()
                    constraints_leaf['initial_prediction'] = 0
                    constraints_leaf['learning_rate'] = self.__learner.learning_rate
                constraints = constraints.append(constraints_leaf)
        return constraints

    def constraint_extrapolation_skEN(self, class_c):
        '''
        :return: constraint: prediction follows the structure Coeff*x+intercept
        '''
        ## Assume a regression model
        assert class_c != 'multiclass', 'sorry, the multiclass is under development'
        columns = [feature for feature in self.get_features_list()]
        if class_c == 'continuous':
            constraint = pd.DataFrame(data=[self.__learner.coef_], columns=columns)
        else:
            constraint = pd.DataFrame(data=[self.__learner.coef_[0]], columns=columns)
        constraint['intercept'] = self.__learner.intercept_

        return constraint

    def __extract_layer(self, l):
        df_sub = pd.DataFrame(self.__learner.coefs_[l].transpose()).add_prefix('node_')
        df_sub['intercept'] = self.__learner.intercepts_[l]
        df_sub['layer'] = l
        df_sub['node'] = range(len(df_sub))
        return df_sub

    def constraint_extrapolation_MLP(self, class_c):
        assert class_c != 'multiclass', 'sorry, the multiclass is under development'
        n_layers = len(self.__learner.coefs_)
        constraints = pd.concat([self.__extract_layer(l) for l in range(n_layers)],axis=0)
        cols_to_move = ['intercept', 'layer', 'node']
        constraints = constraints[cols_to_move + [col for col in constraints.columns if col not in cols_to_move]]
        return constraints


    def constraint_extrapolation(self, class_c):
        if self.__algorithm in ["iai","iai-single"]:
            constraints = self.constraint_extrapolation_iai(class_c)
        elif self.__algorithm == "cart":
            constraints = self.constraint_extrapolation_skTree(class_c)
        elif self.__algorithm == "rf":
            constraints = self.constraint_extrapolation_skRF(class_c)
        elif self.__algorithm == "gbm":
            constraints = self.constraint_extrapolation_skGBM(class_c)
        elif self.__algorithm == "linear":
            constraints = self.constraint_extrapolation_skEN(class_c)
        elif self.__algorithm == "svm":
            constraints = self.constraint_extrapolation_SVM(class_c)
        elif self.__algorithm == "mlp":
            constraints = self.constraint_extrapolation_MLP(class_c)
        return constraints