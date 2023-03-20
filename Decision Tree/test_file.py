from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from DecisionTree import DecisionTreeClassifier
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from DecisionTree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn import tree,ensemble
from RandomForest import RandomForestClassifier,RandomForestRegressor
import time
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

data = datasets.load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Measuring performance of custom decision tree classifier

start = time.time()
clf = DecisionTreeClassifier(max_depth=10)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
end = time.time()

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

acc = accuracy(y_test, predictions)
print(f"The accuracy for the custom DT model is {acc} and the time taken is {end-start}")


clf.visualize_tree(feature_names=data.feature_names)

# dot.render('tree',format='png',view=True)

#Measuring performance of sklearn decision tree classifier

start = time.time()
clf = tree.DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
end = time.time()

acc = accuracy(y_test, predictions)
print(f"The accuracy for the sklearn DT model is {acc} and the time taken is {end-start}")





#Measuring performance of custom random forest classifier

start = time.time()
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
end = time.time()


acc = accuracy(y_test, predictions)
print(f"The accuracy for the custom RF model is {acc} and the time taken is {end-start}")

#Measuring performance of sklearn random forest classifier

start = time.time()
clf = ensemble.RandomForestClassifier(max_depth=10,random_state=42)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
end = time.time()

acc = accuracy(y_test, predictions)
print(f"The accuracy for the sklearn RF model is {acc} and the time taken is {end-start}")


#for regression task

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a decision tree regressor on the training set & Evaluate the model on the testing set
start = time.time()
dt = DecisionTreeRegressor(max_depth=3, min_samples_split=12)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
end = time.time()


mse = mean_squared_error(y_test, y_pred)
print(f"The MSE for the custom DT Regressor model is {mse} and the time taken is {end-start}")

dt.plot_tree(feature_names=data.feature_names)
#checking with the sklearn decision tree


r2 = r2_score(y_test, y_pred)
print(f"The R2_Score for the custom DT Regressor model is {r2} and the time taken is {end-start}")

#Check this against SKLearn DT Regressor
start = time.time()
clf = tree.DecisionTreeRegressor(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
end = time.time()

mse = mean_squared_error(y_test, y_pred)
print(f"The MSE for the SKlearn DT model is {mse} and the time taken is {end-start}")

r2 = r2_score(y_test, y_pred)
print(f"The R2_Score for the Sklearn DT model is {r2} and the time taken is {end-start}")

#Checking with the custom Random Forrest Regressor
start = time.time()
clf = RandomForestRegressor()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
end = time.time()

mse = mean_squared_error(y_test, y_pred)
print(f"The MSE for the Custom RF DT Regressor model is {mse} and the time taken is {end-start}")

r2 = r2_score(y_test, y_pred)
print(f"The R2_Score for the Custom RF DT Regressor model is {r2} and the time taken is {end-start}")


#Checking against SKlearn RF Regressor
start = time.time()
clf = ensemble.RandomForestRegressor(max_depth=10,random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
end = time.time()

mse = mean_squared_error(y_test, y_pred)
print(f"The MSE for the SKLearn RF DT Regressor model is {mse} and the time taken is {end-start}")

r2 = r2_score(y_test, y_pred)
print(f"The R2_Score for the SKLearn RF DT Regressor model is {r2} and the time taken is {end-start}")









