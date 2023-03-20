# EnsembleLearning
### Repository for Ensemble Learning Group Project for Abhishek Tiwari, Deepesh Dwivedi, Namrata Tadanki and Swaraj Bhargava 

In this repository, there are two major components to the project - 

1. Creating a Decision Tree Algorithm from scratch for Classification and Regression Tasks 
2. Implementation of ensemble learning techniques on the Airbnb Properties in NYC dataset for Predicting Prices 

## 1. Decision Tree algorithm - 
The Decision Tree algorithm is based on the CART(Classification and Regression Trees) algorithm using information gain to create the nodes and the leaves. There are two components to this algorithm -
### a. DecisionTreeClassifier - using information gain to calculate the splits. The attributes that can be accessed are - 

root - the root node of the class Node(custom) to access all information about the node - 'feature' used for splitting, 'threshold' using which the feature was split, 'left_node' to the node, and 'right_node' to the node. If the node is a leaf node, which can be checked using the 'is_leaf_node()' function, the node also has a 'value' attached to it. 

Then the object stores information about the 'min_samples_split' that informs how many minimum samples need to be present in a node for a split to happen, 'max_depth' to inform about the maximum depth a tree can be, 'n_features' to consider the number of features to be selected for the algorithm. 

To fit a tree to this algorithm, the 'fit()' function can be accessed and the features and the corresponding labels can be given to create the Decision Tree. To predict labels for a given dataset, the 'predict()' function can be accessed and the function will return the predicted labels for a given set of features. 

There is an additional functionality which allows a user to visualize the tree using the 'visualize_tree' function which takes in the name of the features for the dataset and creates a .png render of the tree. To access this functionality, please download the graphviz package on your desktop using this [link](https://graphviz.org/download/). 

### b. DecisionTreeRegressor - Using mse to determine which split works the best. The attributes that can be accessed are - 

'tree' to access the entire tree which is created after fitting, 'max_depth' to determine the maximum depth a tree can take and 'min_samples_split' to determine the number of samples required for a node to split. 

As the classifier, to fit a tree to this algorithm, the 'fit()' function can be accessed and the features and the corresponding values can be given to create the Decision Tree. To predict the values for a given dataset, the 'predict()' function can be accessed and the function will return the predicted values for a given set of features. 

You can visualize the tree using the 'plot_tree()' functionality which takes the name of the features for the dataset and creates a .png render of the tree. To access this functionality, please download the graphviz package on your desktop using this [link](https://graphviz.org/download/). 


## 2. Aribnb Price Prediction - 

This part of the project entails practical implementation of various ensemble methodologies on the Airbnb properties dataset to predict the price of certain properties. In particular, the focus is on bagging and boosting techniques to improve the performance of the models. 
