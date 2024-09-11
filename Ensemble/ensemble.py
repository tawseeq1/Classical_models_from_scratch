# -*- coding: utf-8 -*-
"""MLFA_7"""

#Name: Syed Mohammad Tawseeq
#Roll: 22CH10090
#MLFA Assignment Ensemble Learning

# from sklearn.datasets import load_boston-----------------`load_boston` has been removed from scikit-learn since version 1.2.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)  #some preprocessing
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])  #some preprocessing
target = raw_df.values[1::2, 2]  #some preprocessing

from sklearn.utils import shuffle
data, target = shuffle(data, target, random_state=42)   #shuffling

X_train, X_temp, y_train, y_temp = train_test_split(data, target, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 5, 10, 15],
    'max_features': [int(np.log2(data.shape[1]))]  # log base 2 of the number of features
}

rf_regressor = RandomForestRegressor(criterion='squared_error', random_state=42) # initializing the Random Forest Regressor

grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error') # performing Grid Search cros Validation
grid_search.fit(X_train, y_train)

#            best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

best_rf_regressor = RandomForestRegressor(criterion='squared_error', random_state=42, **best_params)  # Initializing Random Forest Regressor with best hyperparameters


best_rf_regressor.fit(X_train, y_train)  # training the model on whole training set

y_pred = best_rf_regressor.predict(X_test)  #performing inference on the testing set

mse = mean_squared_error(y_test, y_pred) # calculating MSE
print("Mean Squared Error:", mse)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Target Values')
plt.ylabel('Predicted Target Values')
plt.title('Actual vs Predicted Target Values')
plt.show()



from sklearn.datasets import load_breast_cancer  #importing breast cancer dataset

breast_cancer = load_breast_cancer()  #loading breast cancer dataset
X, y = breast_cancer.data, breast_cancer.target

from sklearn.utils import shuffle
data, target = shuffle(data, target, random_state=42)  #shuffling

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

class AdaBoostClassifier:   #defining class Adaboost
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []

    def fit(self, X, y):
        n_samples, _ = X.shape
        weights = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            model = DecisionTreeClassifier(max_depth=1)  #creating a decision tree classifier
            model.fit(X, y, sample_weight=weights)
            y_pred = model.predict(X)
            err = np.sum(weights * (y_pred != y))

            alpha = 0.5 * np.log((1 - err) / (err + 1e-10))
            weights *= np.exp(-alpha * y * y_pred)
            weights /= np.sum(weights)

            self.models.append(model)
            self.alphas.append(alpha)

    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        return np.sign(np.dot(self.alphas, predictions))

best_accuracy = 0
best_n_estimators = None
for n_estimators in [50, 100, 150]:
    ada_boost = AdaBoostClassifier(n_estimators=n_estimators)
    ada_boost.fit(X_train, y_train)
    y_pred_val = ada_boost.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred_val)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_n_estimators = n_estimators

print("Best number of weak learners:", best_n_estimators)

from sklearn.metrics import classification_report, confusion_matrix

best_ada_boost = AdaBoostClassifier(n_estimators=best_n_estimators)  # training with the best number of weak learners
best_ada_boost.fit(X_train, y_train)

y_pred_test = best_ada_boost.predict(X_test)  #prediction

#  classification report and confusion matrix
print("Custom AdaBoost Classifier:")
print(classification_report(y_test, y_pred_test))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_test))

from sklearn.ensemble import AdaBoostClassifier as SklearnAdaBoostClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [50, 100, 150]}  # hyperparameter tuning for sklearn implementation
grid_search = GridSearchCV(estimator=SklearnAdaBoostClassifier(), param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_n_estimators_sklearn = grid_search.best_params_['n_estimators']  # best number of weak learners

best_ada_boost_sklearn = SklearnAdaBoostClassifier(n_estimators=best_n_estimators_sklearn) #training sklearn AdaBoostClassifier with the best number of weak learners
best_ada_boost_sklearn.fit(X_train, y_train)

y_pred_test_sklearn = best_ada_boost_sklearn.predict(X_test) #prediction

#  classification report and confusion matrix for sklearn's AdaBoostClassifier
print("Sklearn's AdaBoost Classifier:")
print(classification_report(y_test, y_pred_test_sklearn))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_test_sklearn))

