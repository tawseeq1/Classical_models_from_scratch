# Name: Syed Mohamad Tawseeq 
# Roll: 22CH10090

#First we will import necessary libraries for our progam
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

#EXPERIMENT 1
data = pd.read_csv('BostonHousingDataset.csv') #reading the data into variable named data
print(data.describe()) #some analysis about the data
column_to_drop = ['LSTAT', 'B'] #the columns we have to drop as per the question
data = data.drop(column_to_drop, axis = 1) #dropping the columns
data = data.dropna() #dropping the columns with NaN values
dataset_altered = data.astype(float)  #setting each datapoint as floating point value
print(dataset_altered.head(9))

#EXPERIMENT 2
plt.figure(figsize=(12, 4))  #creating a figure with three subplots
plt.subplot(1, 3, 1)
sns.histplot(dataset_altered['NOX'], kde=True)
plt.subplot(1, 3, 2)
sns.histplot(dataset_altered['RM'], kde=True)
plt.subplot(1, 3, 3)
sns.histplot(dataset_altered['AGE'], kde=True)
plt.show()
correlation_table = data.corr() 
print(correlation_table) #Printing the correlation table
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_table, annot=True, cmap='coolwarm') #plotting the heatmap using seaborn
plt.title("Correlation Matrix Heatmap")
plt.show()

#EXPERIMENT 3
features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO'] # these are our features
target = ['MEDV'] #this is our target 
dataset_altered_features = dataset_altered[features] #new dataset with only features
dataset_altered_target = dataset_altered[target] #new dataset with only target
from sklearn.model_selection import train_test_split #test train split
X_train, X_test, y_train, y_test = train_test_split(
dataset_altered_features,
    dataset_altered_target,
    test_size=0.10,  # 10% for testing as per given condition
    shuffle=False,  # no shufflingas per given condition
    random_state=100  # random state for reproducibility as per given condition
)
print("Shape of X_train is: ", X_train.shape)
print("Shape of y_train is: ", y_train.shape)
print("Shape of X_test is: ", X_test.shape)
print("Shape of y_test is: ", y_test.shape)

#EXPERIMENT 4
def RMSE(predictions, targets): #function to calculate RMSE
    mse = np.mean((predictions - targets)**2)
    rmse = np.sqrt(mse)
    return rmse
def LR_ClosedForm(X_train, y_train):  #implementing closed form function
    X = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1) #adding column of 1's to include bias
    theta = np.linalg.inv(X.T @ X) @ X.T @ y_train  #formula for closed form linear regression 
    predictions = np.dot(X, theta)
    return theta, predictions
thetaa_df, predictions_closed_form = LR_ClosedForm(X_train, y_train)  #theta_df contains all the optimized values parameters and bias
theta_df = np.array(thetaa_df) # converting it into an array for easy calculations
Intercept = theta_df[0] #first element would be the bias , so storing it seperately.
Coefficients = theta_df[1:]#rest of the elements would be coefficients (w)
print("Coefficients: ", Coefficients) #printing coefficients
print("Intercept is: ", Intercept) #printing intercept
#testing for out test data
test_pred_1 = np.dot(X_test, Coefficients) + Intercept #using same values of intercept and coefficients on test set
rmse = RMSE(test_pred_1, y_test) #RMSE 
print(rmse)

#EXPERIMENT 5
def derivative(function, X_train, y_train): #function to calculate the derivativee (dJ/dw and dJ/db)
  m = X_train.shape[0] #no. of rows
  cost = (1/2*m)*np.sum((function - y_train)**2, axis=0) #calculating the cost while setting axis = 0
  dw = (1/m) * np.dot(X_train.T, (function - y_train)) #using simple mathematics , we get this formula (note dw = dJ/dw)
  db = (1/m) * np.sum(function - y_train) #using simple mathematics , we get this formula (note db = dJ/db)
  return cost, dw, db
def LR_Gradient(X_train, y_train, num_iterations, learning_rate): #function to implement Linear regression with gradient descent.
  no_of_columns = X_train.shape[1] #no. of columns
  w = np.zeros(no_of_columns) #initializing w with zeros
  b = 0.0 #initializing b with 0
  for i in range(num_iterations):
    function = np.dot(X_train, w) + b #function would be (w.x  + b)
    cost , dw, db = derivative(function, X_train, y_train) # getting cost and derivatives from the earlier function
    w -= learning_rate * dw.T[0]   #updating coefficients
    b -= learning_rate * db     # updating intercept
  return w, b
def predict(X_test, w, b): #function to predict for new data.
  y_pred = np.dot(X_test, w) + b
  return y_pred
def normalization(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_scaled = (X - mean) / std
    return X_scaled, mean, std
X_train_scaled, mean_xtrain, std_xtrain = normalization(X_train) # we want to do normalization for X_train and use sam emean and std for test
X_test_scaled = (X_test - mean_xtrain)/std_xtrain # doing as mentioned above for test
X_train_numpy = X_train_scaled.to_numpy() #converting to nupy array for better calculations
X_test_numpy = X_test_scaled.to_numpy() #converting to nupy array for better calculations
y_train_numpy = y_train.to_numpy() #converting to nupy array for better calculations
y_test_numpy = y_test.to_numpy() #converting to nupy array for better calculations
learning_rates = [0.1, 0.01, 0.001]
for i in learning_rates:
    w, b = LR_Gradient(X_train_numpy, y_train_numpy, 500, i)
    y_pred_gradient_descent = predict(X_test_scaled, w, b)
    rmse = RMSE(y_pred_gradient_descent, y_test_numpy)
    print("RMSE for learning rate ", i, " = ")
    print(rmse)

