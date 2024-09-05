# -*- coding: utf-8 -*-
"""MLFA_6"""

#Syed Mohamad Tawseeq
#Roll : 22CH10090
#MLFA Lab Assignment 6

import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)   #this checks whether we are using GPU or not

# First we will define transformations that will be used for data augmentation
transform_train = transforms.Compose([
    transforms.RandomRotation(degrees=10),  # rotation
    transforms.RandomHorizontalFlip(),      # horizontal flipping
    transforms.RandomCrop(size=28, padding=4),  # cropping
    transforms.ToTensor(),                  # Converting our image to tensor
])

import requests
from io import BytesIO

file_url = 'https://drive.google.com/uc?export=download&id=1YDgR-i0eSo_LGwTiwxbriLH_90MYyKNB'
response = requests.get(file_url)
file_bytes = BytesIO(response.content)

data = np.load(file_bytes)

data

X_train = pd.DataFrame(data['X_train'].reshape(-1, 28*28))  # Reshaping the images to 2d
y_train = pd.DataFrame(data['y_train'])
X_test = pd.DataFrame(data['X_test'].reshape(-1, 28*28))  # Reshaping the images to 2d
y_test = pd.DataFrame(data['y_test'])

X_train_values = X_train.values.reshape(-1, 28*28)
X_test_values = X_test.values.reshape(-1, 28*28)

X_train_tensor = torch.tensor(X_train_values, dtype=torch.float32) # Converting the images to PyTorch tensors
X_test_tensor = torch.tensor(X_test_values, dtype=torch.float32) # Converting the images to PyTorch tensors

y_train_values = y_train.values.flatten() #flattening the y values
y_test_values = y_test.values.flatten() #flattening the y values

y_train_tensor = torch.tensor(y_train_values, dtype=torch.long) # Converting the labels to PyTorch tensors
y_test_tensor = torch.tensor(y_test_values, dtype=torch.long) # Converting the labels to PyTorch tensors

train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor) # Creating the pytorch datasets
test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor) # Creating the pytorch datasets

from torch.utils.data import TensorDataset, DataLoader

test_loader = DataLoader(test_dataset, batch_size=32) # creating DataLoader for the test set with batch size set to 32, this can be changed to later

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42) # splitting the training set into train and val  (80% train AnD 20% val)

X_train_values = X_train.values.reshape(-1, 28*28) # reshaping
X_val_values = X_val.values.reshape(-1, 28*28) # reshaping
y_train_values = y_train.values.flatten()   # reshaping
y_val_values = y_val.values.flatten()# reshaping

X_train_tensor = torch.tensor(X_train_values, dtype=torch.float32) # converting training and val data to Pytorch tensors
X_val_tensor = torch.tensor(X_val_values, dtype=torch.float32) # converting training and val data to Pytorch tensors
y_train_tensor = torch.tensor(y_train_values, dtype=torch.long) # converting training and val data to Pytorch tensors
y_val_tensor = torch.tensor(y_val_values, dtype=torch.long) # converting training and val data to Pytorch tensors


train_dataset_augmented = TensorDataset(X_train_tensor, y_train_tensor) # creating tensor dataset
train_dataset_augmented.transform = transform_train  #  transformations

train_loader = DataLoader(train_dataset_augmented, batch_size=32, shuffle=True) # creatign dataloader

val_dataset = TensorDataset(X_val_tensor, y_val_tensor)  # creatign tensor dataset

val_loader = DataLoader(val_dataset, batch_size=32)  # creatign dataloader

class test_nn(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(test_nn, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define hyperparameters
input_size = 28*28  # Input size (28x28 image)
output_size = 10    # Number of classes (0-9)
batch_size = 32     # Batch size for training
epochs = 10         # Number of epochs
num_hidden_layers = 2  # Number of hidden layers
activation = nn.ReLU()  # Activation function

# Defining the grid of hyperparameters to be tuned
param_grid = {
    'hidden_size1': [128, 256, 512],  # number of units in the first hidden layer
    'hidden_size2': [64, 128, 256],   # number of units in the second hidden layer
    'learning_rate': [0.001, 0.01, 0.1]  # nearning rate
}

best_accuracy = 0
best_params = None

#Loopingg over each combination
for params in ParameterGrid(param_grid):
    model = test_nn(input_size, params['hidden_size1'], params['hidden_size2'], output_size) # Creating the model with the current set of hyperparameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    for epoch in range(epochs): # Training our model
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    # Evaluating the model on the validation set
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total

    if accuracy > best_accuracy:     # Updating best accuracy and best parameters
        best_accuracy = accuracy
        best_params = params

print("Best hyperparameters:", best_params)
print("Best validation accuracy:", best_accuracy)

# Defining our new model with best hyperparameters
class best1_model(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(best1_model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initializing model, loss function, and optimizer
model = best1_model(input_size=28*28, hidden_size1=512, hidden_size2=64, output_size=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# below are the Lists to store training and validation loss
train_losses = []
val_losses = []

# Training loop
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        inputs = inputs.view(inputs.shape[0], -1)  # Flattening the input
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    # Validation loop
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.view(inputs.shape[0], -1)  # Flattening the input
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    print(f'Epoch [{epoch+1}/10], '
          f'Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

# Plottingn
plt.plot(range(1, 11), train_losses, label='Training Loss')
plt.plot(range(1, 11), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Evaluation loop on test set
model.eval()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.view(inputs.shape[0], -1)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_loss /= len(test_loader)
test_accuracy = 100 * correct / total
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}%')

from sklearn.metrics import precision_score, recall_score, confusion_matrix

# eval loop on val set
model.eval()
val_loss = 0.0
correct = 0
total = 0
val_predicted = []
val_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.view(inputs.shape[0], -1)  # Flattening the input
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        val_predicted.extend(predicted.tolist())
        val_labels.extend(labels.tolist())

val_loss /= len(val_loader)
val_accuracy = 100 * correct / total
print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}%')

#  confusion matrix
conf_matrix = confusion_matrix(val_labels, val_predicted)

#  precision And recall scores
precision = precision_score(val_labels, val_predicted, average=None)
recall = recall_score(val_labels, val_predicted, average=None)

# Visualization of confusion matrix
print('Confusion Matrix:')
print(conf_matrix)

# precision and recall scores
print('\nPrecision Scores:')
print(precision)
print('\nRecall Scores:')
print(recall)

#From here is the start of question 2

import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, num_filters_conv1, num_filters_conv2):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=num_filters_conv1, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=num_filters_conv1, out_channels=num_filters_conv2, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(num_filters_conv2 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, num_filters_conv2 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy

import torch.optim as optim

# Defining hyperparameter grids for tuning purpose
num_filters_grid_conv1 = [16, 32, 64]
num_filters_grid_conv2 = [32, 64, 128]
learning_rate_grid = [0.001, 0.01, 0.1]

best_accuracy = 0
best_num_filters_conv1 = None
best_num_filters_conv2 = None
best_learning_rate = None

# Hyperparameter loop
for num_filters_conv1 in num_filters_grid_conv1:
    for num_filters_conv2 in num_filters_grid_conv2:
        for learning_rate in learning_rate_grid:
            # Creating CNN model for each case
            cnn_model = CNNModel(num_filters_conv1, num_filters_conv2)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(cnn_model.parameters(), lr=learning_rate)

            # Training loop
            for epoch in range(10):
                cnn_model.train()
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    inputs = inputs.view(-1, 1, 28, 28)  # Reshapingg input
                    outputs = cnn_model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                # val loop
                cnn_model.eval()
                val_accuracy = 0.0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs = inputs.view(-1, 1, 28, 28)  # Reshaping input
                        outputs = cnn_model(inputs)
                        val_accuracy += calculate_accuracy(outputs, labels)
                    val_accuracy /= len(val_loader)

                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    best_num_filters_conv1 = num_filters_conv1
                    best_num_filters_conv2 = num_filters_conv2
                    best_learning_rate = learning_rate

print(f'Best hyperparameters: Num Filters Conv1 = {best_num_filters_conv1}, Num Filters Conv2 = {best_num_filters_conv2}, Learning Rate = {best_learning_rate}')
print(f'Best Accuracy = {best_accuracy}')

num_filters_grid_conv1 = 16
num_filters_grid_conv2 = 32
learning_rate_grid = 0.001
# Training CNN model using the best hyperparameters
cnn_model = CNNModel(num_filters_conv1, num_filters_conv2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=best_learning_rate)

train_losses = []
val_losses = []

for epoch in range(10):
    cnn_model.train()
    epoch_train_losses = []
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        inputs = inputs.view(-1, 1, 28, 28)
        outputs = cnn_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_train_losses.append(loss.item())
    train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
    train_losses.append(train_loss)
    print(f"Epoch {epoch+1}/{10}, Training Loss: {train_loss:.4f}")

    cnn_model.eval()
    epoch_val_losses = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.view(-1, 1, 28, 28)
            outputs = cnn_model(inputs)
            loss = criterion(outputs, labels)
            epoch_val_losses.append(loss.item())
    val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
    val_losses.append(val_loss)

# Evaluate funtion to used for both test and val eval purpose
def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.view(-1, 1, 28, 28)
            outputs = model(inputs)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    avg_loss = loss / len(data_loader)
    return accuracy, avg_loss

val_accuracy, val_loss = evaluate_model(cnn_model, val_loader)
test_accuracy, test_loss = evaluate_model(cnn_model, test_loader)

# Visualization of training and val loss graphs over all epochs
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# our evaluation results
print(f'Validation Accuracy: {val_accuracy}, Test Accuracy: {test_accuracy}')
print(f'Validation Loss: {val_loss}, Test Loss: {test_loss}')

# Evaluate model performance on validation and test sets
from sklearn.metrics import precision_score, recall_score

def evaluate_model_performance(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    loss = 0
    all_predicted = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.view(-1, 1, 28, 28)
            outputs = model(inputs)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            all_predicted.extend(predicted.tolist())
            all_labels.extend(labels.tolist())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    precision = precision_score(all_labels, all_predicted, average='weighted')
    recall = recall_score(all_labels, all_predicted, average='weighted')
    return precision, recall

val_precision, val_recall = evaluate_model_performance(cnn_model, val_loader)
test_precision, test_recall = evaluate_model_performance(cnn_model, test_loader)

print("val Metrics:")
print(f"Precision: {val_precision}")
print(f"Recall: {val_recall}")

print("\nTest Metrics:")
print(f"Precision: {test_precision}")
print(f"Recall: {test_recall}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_params_cnn = count_parameters(cnn_model)
print("trainable Parameters in our cnn Model:", num_params_cnn)
num_params_cnn = count_parameters(model)
print("trainable Parameters in our neural net Model:", num_params_cnn)

from thop import profile

# Input to the model
input_data = torch.randn(1, 1, 28, 28)

# Profile the model
flops1, params1 = profile(cnn_model, inputs=(input_data,))
print("Estimated FLOPs for CNN Model:", flops1)

!pip install flopth

from flopth import flopth
flopss, paramss = flopth(model, in_size=(20,),show_detail=True)
print(flopss, paramss)
