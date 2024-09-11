# -*- coding: utf-8 -*-
"""MLFA_LT_2"""

#Name: Syed Mohamad Tawseeq
#Roll: 22CH10090
#MLFA Lab Test 2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

!unzip /content/drive/MyDrive/CLIP_Tasks/lab_test_2_dataset.zip #unzipping the file that was uploaded on google drive

import os
from PIL import Image
import numpy as np
dataset_path = "/content/lab_test_2_dataset" #here we get the file from google colab menu
data = []
labels = []

for folder_name in os.listdir(dataset_path):   #loops in for every folder of the dataset (age)
    age = int(folder_name)    #as the folder name is the age
    folder_path = os.path.join(dataset_path, folder_name)
    for filename in os.listdir(folder_path): #loops for each image in a folder
        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path) #opens the image

        image_np = np.array(image) #converts it into numpy array
        data.append(image_np) #appends it into out data
        labels.append(age) #appends label to out data

import torch
import numpy as np
import random

seed_value = 2022

#experiment 1
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)

device = torch.device('cpu')

print(device)

from sklearn.model_selection import train_test_split
#experiment 2
random.shuffle(data)

train_size = 0.70
val_size = 0.15
test_size = 0.15

X_train, X_val_test, y_train, y_val_test = train_test_split(data, labels, test_size=(val_size + test_size), random_state=seed_value)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=test_size/(val_size + test_size), random_state=seed_value)

print("Overall :", len(data))
print("Training size:", len(X_train))
print("Validation size:", len(X_val))
print("Testing size:", len(X_test))



X_train_tensor = torch.tensor(X_train) #converting to pytorch tensors
X_test_tensor = torch.tensor(X_test)#converting to pytorch tensors
X_val_tensor = torch.tensor(X_val_test)#converting to pytorch tensors
y_train_tensor = torch.tensor(y_train)#converting to pytorch tensors
y_test_tensor = torch.tensor(y_test)#converting to pytorch tensors
y_val_tensor = torch.tensor(y_val_test)#converting to pytorch tensors

# X_train_tensor

import torch
from torch.utils.data import DataLoader, TensorDataset
train_dataset = TensorDataset(X_train_tensor, y_train_tensor) #converting to Tensor Data set (grouping train and test data)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor) #converting to Tensor Data set (grouping train and test data)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)  #converting to Tensor Data set (grouping train and test data)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #converting to dataloader with specific batch size
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) #converting to dataloader with specific batch size
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) #converting to dataloader with specific batch size

train_loader

import torch.nn as nn
from torch.nn import MaxPool2d

class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.dropout = nn.Dropout(0.2)
        #self.maxpool = nn.MaxPool2D()

    def forward(self, x):
      x = torch.relu(self.conv1(x))
      x = torch.nn.functional.max_pool2d(x, kernel_size=2)
      x = torch.relu(self.conv2(x))
      x = torch.nn.functional.max_pool2d(x, kernel_size=2)

      x = x.reshape(-1, 32*8*8)  # here we use .reshape() instead of .view()
      x = torch.relu(self.fc1(x))
      x = self.dropout(x)  # Specify training mode
      x = torch.relu(self.fc2(x))
      x = self.fc3(x)

      return x


model = MyCNN()   #defining my model

train_loss = []
test_loss = []
val_loss = []
epochs11 = 25
loss_criterion = nn.MSELoss()

optimizer = optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
for epochs in range(epochs11):
  model.train()
  running_train_loss = 0.0  #for each epoch
  for input, labels in train_loader:
    input = input.to(torch.float32)
    input = input.permute(0, 3, 1, 2)
    optimizer.zero_grad() #setting grads to zero
    output = model(input).float() #inputing into the model
    labels = labels.to(torch.float32)
    loss = loss_criterion(output, labels) #calculating loss
    loss = loss.to(torch.float32)
    loss.backward() #back prop
    optimizer.step() #updating parameters
    running_train_loss += loss.item() * input.size(0)
  train_loss_each = running_train_loss / len(train_loader.dataset) # averaging out by batches
  train_loss.append(train_loss)
  model.eval()
  running_val_loss = 0.0
  with torch.no_grad():
      for inputs, labels in val_loader:
          inputs = inputs.to(torch.float32)
          inputs = inputs.permute(0, 3, 1, 2)
          inputs, labels = inputs.to(device), labels.to(device)
          outputs = model(inputs).float()
          labels = labels.to(torch.float32)
          loss = loss_criterion(outputs, labels.float().unsqueeze(1))
          running_val_loss += loss.item() * inputs.size(0)
  val_loss_each = running_val_loss / len(val_loader.dataset)
  val_loss.append(val_loss)

  if (epochs + 1) % 5 == 0:   # saving the model at regular intervals,  every 5 epochs
      torch.save(model.state_dict(), f"model_epoch_{epochs+1}.pt")

import matplotlib.pyplot as plt

plt.plot(range(1, 26), train_loss, label="Train Loss")
plt.plot(range(1, 26), val_loss, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Losses")
plt.legend()
plt.show()

import torch.nn.functional as F

model.eval()
test_loss = 0.0
predictions = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = F.mse_loss(outputs, labels.float().unsqueeze(1))  # Calculating MSE loss
        test_loss += loss.item() * inputs.size(0)
        predictions.extend(outputs.cpu().numpy().flatten())  # Storing predictions

test_loss /= len(test_loader.dataset)
print(f"MSE on Test: {test_loss:.4f}")

predictions = np.array(predictions)
plt.scatter(y_test, predictions, alpha=0.5)
plt.xlabel("Ground Age")
plt.ylabel("Predicted value of Age")
plt.title("Scatter Plot, Predicted vs. Ground Truth Ages")
plt.grid(True)
plt.show()
