from DecisionTree import DecisionTreeClassifier, DecisionTreeRegressor
import numpy as np
from collections import Counter

class RandomForestClassifier:
    def __init__(self,n_trees=2, max_depth=100, min_samples_split=2, n_features=None):
        '''
        Initialization function to set the parameters and attributes for the Random Forest Classifier
        :param n_trees: number of trees to create
        :param max_depth: maximum depth of each tree
        :param min_samples_split: minimum number of samples in a node
        :param n_features: number of features to consider
        '''
        self.n_trees = n_trees
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.n_features=n_features
        self.trees = []

    def fit(self, X, y):
        '''
        fit function to create and train the Random Forest Classifier
        :param X: input data
        :param y: output labels
        :return: none
        '''
        self.trees = []
        for _ in range(self.n_trees):
            #creating decision trees randomly
            tree = DecisionTreeClassifier(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                n_features=self.n_features)
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions

class RandomForestRegressor:
    def __init__(self, n_trees=10, max_depth=5, min_samples_split=2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.forest = []

    def fit(self, X, y):
        for i in range(self.n_trees):
            # Subsample the training data
            idx = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
            X_train_sub = X[idx,:]
            y_train_sub = y[idx]
            # Create a decision tree regressor and fit it to the subsampled data
            dt = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            dt.fit(X_train_sub, y_train_sub)
            # Add the decision tree to the forest
            self.forest.append(dt)

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for dt in self.forest:
            y_pred += dt.predict(X)
        y_pred /= self.n_trees
        return y_pred
