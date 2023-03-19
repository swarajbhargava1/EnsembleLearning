import numpy as np
from collections import Counter

class Node:
    '''
    This class is used to define Nodes for the samples as and when they come
    '''
    def __init__(self,feature=None,threshold=None,left_node=None,right_node=None,*,value=None):
        '''
        Storing information about each node
        :param feature: which feature has been used to define the node
        :param threshold: what was the threshold used to define the node
        :param left_node: what node is to the left of this node
        :param right_node: what node is to the right of this node
        :param value: value of the node if it is a leaf node
        '''
        self.feature = feature
        self.threshold = threshold
        self.left_node = left_node
        self.right_node = right_node
        self.value = value

    def is_leaf_node(self):
        '''
        function to identify if the root is a leaf or not
        :return: Boolean True or False
        '''
        return self.value is not None


class DecisionTreeClassifier:
    '''
    Decision Tree classifier class based on the CART algorithm using Information Gain to calculate the splits
    '''
    def __init__(self,min_samples_split=2,max_depth=100,n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):

        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)
        self.root = self._grow_tree(X,y)

    def _grow_tree(self,X,y,depth=0):
        n_samples,n_feats = X.shape
        n_labels = len(np.unique(y))

        #define stopping criteria
        if(depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats,self.n_features,replace=False) #False so we don't get the same index again

        #finding the best split
        best_feature, best_threshold = self._best_split(X,y,feat_idxs)

        #create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature,best_threshold,left,right)

    def _best_split(self,X,y,feat_idxs):
        '''
        The function identifies the best split based on information gain for the given tree
        :param X:
        :param y:
        :param feat_idxs:
        :return: best split index and threshold
        '''
        best_gain=-1
        split_idx,split_threshold = None,None

        for idx in feat_idxs:
            X_sub = X[:,idx]
            thresholds = np.unique(X_sub)

            for thr in thresholds:
                #calculating information gain
                gain = self._information_gain(y, X_sub,thr)

                if gain > best_gain:
                    best_gain=gain
                    split_idx = idx
                    split_threshold = thr

        return split_idx,split_threshold

    def _information_gain(self,y,X,threshold):
        '''
        calculates the information gain based on the following formula:
        IG = Entropy(parent) - weighted average of the entropy of the children
        :param y: class values
        :param X: feature
        :param threshold: division threshold
        :return: gain
        '''

        parent_entropy = self._entropy(y)

        #create children for the parent
        left_idx,right_idx = self._split(X,threshold)

        if(len(left_idx)==0 or len(right_idx)==0):
            return 0

        #calculating the weighted entropy of the children
        n=len(y)
        n_l,n_r = len(left_idx),len(right_idx)
        e_l,e_r = self._entropy(y[left_idx]), self._entropy(y[right_idx])

        children_entropy = (n_l/n)*e_l + (n_r/n)*e_r

        information_gain = parent_entropy - children_entropy

        return information_gain

    def _entropy(self,y):
        '''
        calculates the entropy within each node from the following formula:
        p_i = (no. of instances of i)/(total samples)
        and entropy = -Σ(p_i * log(p_i))
        :param y:
        :return: entropy
        '''
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def _split(self,X,threshold):
        '''
        returns the indexes based on a threshold to split one parent node
        :param X:
        :param threshold:
        :return:
        '''
        left_idxs = np.argwhere(X <= threshold).flatten()
        right_idxs = np.argwhere(X > threshold).flatten()
        return left_idxs, right_idxs

    def _most_common_label(self,y):
        '''
        returns the most common label for a node
        :param y:
        :return:
        '''
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        '''
        prediction function to find the value of X
        :param X:
        :return: the predicted classes for the dataset
        '''
        return [self._traverse_tree(x,self.root) for x in X]

    def _traverse_tree(self, x, node):
        '''
        recursive function to go through the decision tree created during fitting and finding the right class
        :param x: value
        :param node: which node
        :return: final return would be the most appropriate class
        '''
        if node.is_leaf_node():
            return node.value

        if x[node.feature]<=node.threshold:
            return self._traverse_tree(x, node.left_node)

        return self._traverse_tree(x, node.right_node)
