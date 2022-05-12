import pandas as pd
import numpy as np

class LinearRegression:

    def __init__(self):
        self._B_0 = None
        self._B_1 = None

    def fit(self, X, y):
        X_bar, y_bar = np.mean(X), np.mean(y)
        cov_matrix = np.cov(X, y)
        self._B_1 = cov_matrix[0][1] / cov_matrix[0][0]
        self._B_0 = y_bar - self._B_1 * X_bar

    def predict(self, test):
        predictions = []
        for i in test:
            predictions.append(self._B_0 + self._B_1 * i)
        return np.array(predictions)

    def score(self, X_test, y_test):
        sse = np.sum((y_test - self.predict(X_test)) ** 2)
        sst = np.sum((y_test - np.mean(y_test)) ** 2)
        return 1 - sse / sst

class Node:
    def __init__(self, value=None, threshold=None, left=None, right=None):
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeRegressor:
    def __init__(self, min_samples_split=2, max_depth=float('inf')):
        self.root = None
        self._min_samples_split = min_samples_split
        self._max_depth = max_depth
        
    def _build(self, X, y, depth):
        num_samples = np.shape(X)[0]
        if num_samples >= self._min_samples_split and depth <= self._max_depth:
            best_split = self._get_best_split(X, y, num_samples)
            if len(best_split) != 0 and best_split["var_red"] > 0:
                left_subtree = self._build(best_split["left_values"]['X'], 
                    best_split["left_values"]['y'], depth + 1)
                right_subtree = self._build(best_split["right_values"]['X'], 
                    best_split['right_values']['y'], depth + 1)
                return Node(None, best_split["threshold"],
                            left_subtree, right_subtree)
        return Node(np.mean(y))
    
    def _get_best_split(self, X, y, num_samples):
        best_split = {}
        max_var_red = -float("inf")
        for threshold in np.unique(X):
            left_values = {'X': [], 'y': []}
            right_values = {'X': [], 'y': []}
            for i in range(len(X)):
                if X[i] <= threshold:
                    left_values['X'].append(X[i])
                    left_values['y'].append(y[i])
                else:
                    right_values['X'].append(X[i])
                    right_values['y'].append(y[i])
            if len(left_values['y']) > 0 and len(right_values['y']) > 0:
                y, left_y, right_y = y, left_values['y'], right_values['y']
                curr_var_red = self.get_var_red(y, left_y, right_y)
                if curr_var_red > max_var_red:
                    best_split["threshold"] = threshold
                    best_split["left_values"] = left_values
                    best_split["right_values"] = right_values
                    best_split["var_red"] = curr_var_red
                    max_var_red = curr_var_red

        return best_split
      
    def get_var_red(self, node, left_child, right_child):        
        weight_l = len(left_child) / len(node)
        weight_r = len(right_child) / len(node)
        return np.var(node) - (weight_l * np.var(left_child) + \
            weight_r * np.var(right_child))

    def fit(self, X, y):       
        self.root = self._build(X, y, 0)
        
    def _predict(self, x, tree):       
        if tree.value != None: 
            return tree.value
        if x <= tree.threshold:
            return self._predict(x, tree.left)
        else:
            return self._predict(x, tree.right)
    
    def predict(self, X):        
        return [self._predict(x, self.root) for x in X]

class RandomForestRegressor:

    def __init__(self, num_trees=100, min_samples_split=2, 
            max_depth=float('inf')):
        self._num_trees = num_trees
        self._min_samples_split = min_samples_split
        self._max_depth = max_depth
        self._decision_trees = []

    def _sample(self, X, y):
        samples = np.random.choice(a=X.shape[0], size=X.shape[0], replace=True)
        return X[samples], y[samples]

    def fit(self, X, y):
        if len(self._decision_trees) > 0:
            self._decision_trees = []

        trees_built = 0
        while trees_built < self._num_trees:
            decision_tree = DecisionTreeRegressor(
                min_samples_split=self._min_samples_split,
                max_depth=self._max_depth
            )

            decision_tree.fit(*self._sample(X, y))
            self._decision_trees.append(decision_tree)
            trees_built += 1

    def predict(self, X):
        y = []
        for i in self._decision_trees:
            y.append(i.predict(X))
        return np.mean(y, axis=0)

class KNeighborsRegressor:

    def __init__(self, k):
        self._k = k
        
    def fit(self, X_train, y_train):
        self._X_train = X_train
        self._y_train = y_train
    
    def predict(self, X_test):
        y_pred = [0] * len(X_test)
        for i in range(len(X_test)):
            dist = abs(self._X_train - X_test[i])
            y_pred[i] = self._y_train[np.argsort(dist, 
                axis=0)[:self._k]].mean()
        return y_pred
