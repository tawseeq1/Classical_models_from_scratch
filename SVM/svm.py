#Name: Syed Mohamad Tawseeq
#Roll: 22CH10090

import numpy as np #importing numpy
import pandas as pd #importing pandas
import matplotlib.pyplot as plt #importing matplotlib
from keras.datasets import mnist #importing mnist dataset
import tensorflow as tf #importing tensorflow

from sklearn.model_selection import train_test_split #importing train test split
from sklearn.svm import SVC #importing SVC
from sklearn.metrics import classification_report #importing classification score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV #importing grid and random search
from scipy.stats import uniform

#-------------------------------------------------------------------------------------------------
#Support Vector Classification
#DATA PREPERATION
(X_train, y_train), (X_test, y_test) = mnist.load_data()   #loading mnist data

X_train = X_train.reshape(X_train.shape[0], -1) #reshaping to flatten the images
X_test = X_test.reshape(X_test.shape[0], -1) #reshaping to flatten the images

X_train = X_train.astype('float32') / 255   #changing to float values and dividing by 255 to normalize
X_test = X_test.astype('float32') / 255   #changing to float values and dividing by 255 to normalize

X_train = X_train[:10000]  #taking first 10000 values
y_train = y_train[:10000]   #taking first 10000 values
X_test = X_test[:2000]   #taking first 2000 values
y_test = y_test[:2000]   #taking first 2000 values

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

#Model Training with Different Kernels
svc_linear = SVC(kernel = 'linear')  #creating a linear kernel
svc_polynomial = SVC(kernel = 'poly') #creating a polynomial kernel
svc_rbf = SVC(kernel = 'rbf') #creating a rbf kernel

svc_linear.fit(X_train, y_train) #  training SVC model for linear kernel
svc_polynomial.fit(X_train, y_train) #  training SVC model for polynomial kernel
svc_rbf.fit(X_train, y_train) #  training SVC model for rbf kernel

y_pred_linear = svc_linear.predict(X_test) #making predictions from testing data for linear kernel
y_pred_polynomial = svc_polynomial.predict(X_test) #making predictions from testing data for polynomial kernel
y_pred_rbf = svc_rbf.predict(X_test) #making predictions from testing data for rbf kernel

report_linear = classification_report(y_test, y_pred_linear) #getting the reports for linear kernel
report_poly = classification_report(y_test, y_pred_polynomial) #getting the reports for polynomial kernel
report_rbf = classification_report(y_test, y_pred_rbf)  #getting the reports for rbf kernel

print("Classification Report - Linear Kernel:")
print(report_linear)
print("--------------------")
print("Classification Report - Polynomial Kernel:")
print(report_poly)
print("-------------------------------------")
print("Classification Report - RBF Kernel:")
print(report_rbf)

#Hyperparameter Tuning
param_grid_grid_search = {'C': [0.1, 1, 10, 100],
                          'gamma': [1, 0.1, 0.01, 0.001]} #parameters for grid search
param_dist_random_search = {'C': uniform(0.1, 100),
                            'gamma': ['scale', 'auto', 1, 0.1, 0.01, 0.001]} #parameters for randomized search

svc = SVC(kernel='rbf')

# Implementing Grid search first with cross validation 5
grid_search = GridSearchCV(estimator=svc, param_grid=param_grid_grid_search, cv=5)
grid_search.fit(X_train, y_train) #fitting the grid search
best_params_grid_search = grid_search.best_params_

# Implementing Randomized search first with cross validation 5
random_search = RandomizedSearchCV(estimator=svc, param_distributions=param_dist_random_search, n_iter=10, cv=5)
random_search.fit(X_train, y_train) #fitting the random search
best_params_random_search = random_search.best_params_
#Note: This takes too much time to run
print("Best Hyperparameters from GridSearchCV:", best_params_grid_search)
print("Best Hyperparameters from RandomizedSearchCV:", best_params_random_search)

#Training the model with Best Hyperparameters that we found above using grid and randomized search
best_model1 = SVC(kernel='rbf', C=100, gamma=0.01) #hyperparameters using grid search
best_model1.fit(X_train, y_train)
best_model2 = SVC(kernel='rbf', C=83.00246311417051, gamma=0.01) #hyperparameters using randomized search
best_model2.fit(X_train, y_train)
y_pred_best_model1 = best_model1.predict(X_test) #predicting using model 1
y_pred_best_model2 = best_model2.predict(X_test) #predicting using model 2

report_best_model1 = classification_report(y_test, y_pred_best_model1) #getting the reports for y_pred_best_model1
report_best_model2 = classification_report(y_test, y_pred_best_model2) #getting the reports for y_pred_best_model2
print("Classification Report - C=100, gamma = 0.01")
print(report_best_model1)
print("--------------------")
print("Classification Report - C=83.00246311417051, gamma = 0.01")
print(report_best_model2)

#both are giving same accuracy hence both can be said as best hyperparameters lets take the results by grd search , C=100, gamme = 0.01

#Visualization
from sklearn.metrics import confusion_matrix
# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_best_model1)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues) #displaying the confusion matrix as image using a nearest-neighbor interpolation method, with a blue colormap
plt.title("Confusion Matrix for SVC with RBF Kernel") #Setting the title of the plot
plt.colorbar()
tick_marks = np.arange(len(np.unique(y_test))) #creates tick marks along the x-axis
plt.xticks(tick_marks, np.unique(y_test))
plt.yticks(tick_marks, np.unique(y_test))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Now filling the confusion matrix with values
for i in range(conf_matrix.shape[0]): #iterating through each cell of the confusion matrix
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black") #adding text to plot at position (j, i) with the value from the confusion matrix formatted as integer

plt.tight_layout() #this prevents overlapping of elements
plt.show()

#-------------------------------------------------------------------------------------------------
#Support Vector Regression
from sklearn.datasets import fetch_california_housing #importing housing dataset
from sklearn.svm import SVR #importing SVR
from sklearn.metrics import mean_squared_error #mse

#Data Preparation
california_housing = fetch_california_housing()
X = california_housing.data
y = california_housing.target
# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

#Model Training with Default Parameters
svr_default = SVR(epsilon=0.5)
svr_default.fit(X_train, y_train)

y_pred_default = svr_default.predict(X_test) # Making predictions on the testing set

mse_default = mean_squared_error(y_test, y_pred_default) # calculating mean squared error (MSE)
print("Mean Squared Error (Default SVR):", mse_default)

# Scatter plot visualization of predictions versus ground truth
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_default)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("SVR with Default Parameters: Predictions vs Ground Truth")
plt.show()

#Hyperparameter Tuning using GridSearchCV
epsilon_values = np.arange(0, 2.6, 0.1) #create epsilon values from 0 to 2.6 with 0.1 difference
param_grid = {'epsilon': epsilon_values}

grid_search = GridSearchCV(SVR(), param_grid, cv=10) #using gris search with 10 fold cross valiation
grid_search.fit(X_train, y_train)
#Note: This takes too much time to run
print("\nBest epsilon parameter obtained from GridSearchCV:", grid_search.best_params_['epsilon'])

# Step 4: Model Training with Best Hyperparameter
best_epsilon = grid_search.best_params_['epsilon']
svr_best = SVR(epsilon=best_epsilon) #using the best epsilon in our regression
svr_best.fit(X_train, y_train)

y_pred_best = svr_best.predict(X_test) # predicting target for the testing set
mse_best = mean_squared_error(y_test, y_pred_best) # calculating MSE
print("Mean Squared Error (Best SVR):", mse_best)

# Scatter plot visualization of predictions versus ground truth
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_best)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("SVR with Best Hyperparameter: Predictions vs Ground Truth")
plt.show()
