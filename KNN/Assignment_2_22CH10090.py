#Name: Syed Mohamad Tawseeq
#Roll: 22CH10090
#MLFA Assignment 2

#importing the libraries that we need
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

#importing the dataset and extracting features and output values 
df = pd.read_csv('C:/Users/smta0/IITKGP/2nd Year/Sem4/MLFA/Assignments/2/Iris.csv')
features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
predictions = ['Species']
target_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
#################################################################
#############   Basic Processing of Data        #################
#################################################################
#Label strings as numbers and replacethem in our new dataset
label_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
df1 = df.replace(label_mapping)

#Analysis of the data
df1[features].describe()
df1.groupby(['Species']).describe()

# Function to calculate euclidean distance between two points (vectors)
def euclidean_distance(point1, point2):
    sum = np.sum((point1 - point2)**2)
    return np.sqrt(sum)

# Function to calculate Z normalization
def z_score_norm(data):
    mean = np.mean(data) #mean
    sd = np.std(data) #standard deviation
    z_score_normalized = (data - mean) / sd #z score
    return z_score_normalized

#function to calculate accuracy
def calculate_accuracy(actual_labels, predicted_labels):
    correct_predictions = sum(actual_labels == predicted_labels)
    total_predictions = len(actual_labels)
    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy

#function to calculate confusion matrix
def calculate_confusion_matrix(actual, predicted, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int) # Initialize a matrix with zeros
    for i in range(len(actual)):  
        # Extracting the actual and predicted classes
        actual_class = int(actual[i])
        predicted_class = int(predicted[i])
        confusion_matrix[actual_class][predicted_class] += 1 #here we updatethe confusion matrix based actual and predicted classes
    return confusion_matrix

#function to plot confusion matrix
def plot_confusion_matrix(confusion_matrix, class_names):
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix, cmap='Blues') # This shows the confusion matrix as an image with a blue colormap
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    for i in range(len(class_names)):  # This loop adds text for each cell in the matrix
        for j in range(len(class_names)):
            text = ax.text(j, i, str(confusion_matrix[i, j]), ha='center', va='center', color='green')
    plt.show()
    
#Normalize the data
df1['SepalLengthCm'] = z_score_norm(df1['SepalLengthCm'])
df1['SepalWidthCm'] = z_score_norm(df1['SepalWidthCm'])
df1['PetalLengthCm'] = z_score_norm(df1['PetalLengthCm'])
df1['PetalWidthCm'] = z_score_norm(df1['PetalWidthCm'])

#Shuffle and split into test train sets
shuffled_df = df1.sample(frac=1, random_state=42) #Random state makes sure to get the same original value everytime
df_train = shuffled_df[:int(0.7*shuffled_df.shape[0])] #first 70% elements of the shuffled data
df_test = shuffled_df[int(0.7*shuffled_df.shape[0]):] #last 30% elements of the shuffled data

#########################################################################
#             Function to calculate KNN_Normal
##########################################################################
def KNN_Normal(df_train, df_test, K, features1):
    answers  = np.zeros((df_test.shape[0])) #array to save final answers
    nearestKpoints = np.zeros((df_test.shape[0], K), dtype=object)  # array to store distance and original indicies of nearest K points of all points in test data
    for i in range(df_test[features1].shape[0]):
        distances = [] # create a distance list so that we can append it later
        for j in range(df_train[features1].shape[0]):
            dist = euclidean_distance(df_train.iloc[j][features1], df_test.iloc[i][features1]) #calculating euclidean distance usign the function declared earlier
            distances.append((dist, j)) #appending each train data distance to the list
        distances = sorted(distances)[:K] # sorting and choosing first K , which would be nearest K neighbours
        nearestKpoints[i] = distances # put them in the array created earlier
        indices = [t[1] for t in nearestKpoints[i]] #storing near points indices seperately
        classarr = np.zeros((df_test.shape[0]))   # no use
        arr = np.zeros((K))
        for m in range(K):
            arr[m] = df_train['Species'].iloc[indices[m]]  #output values of train data
        unique_elements, counts = np.unique(arr, return_counts=True)
        max_count_index = np.argmax(counts) #the no. whis efrequesncy would be max, is the class
        number_with_max_frequency = unique_elements[max_count_index]    
        answers[i] = number_with_max_frequency

    return answers
#######################################################
############ Experiment 1 ##########################
###################################################
K_values = [1, 3, 5, 10, 20] # given
K_list = []
accuracy_list = []
for K in K_values: #loop to check for all given K
    result = KNN_Normal(df_train, df_test, K, features)
    acc = calculate_accuracy(df_test['Species'], result)
    K_list.append(K) #appending K list 
    accuracy_list.append(acc) #appending accuracy list
#below is the plotting
plt.plot(K_list, accuracy_list)
plt.title('Percentage Accuracy vs K for Normal KNN')
plt.xlabel('Value of K')
plt.ylabel('Accuracy in percentage')
plt.legend({'Accuracy'})
plt.xticks((K_list))
plt.show()

####plottig confusion matrix###
confff = KNN_Normal(df_train, df_test, 10, features)
actual_values = df_test['Species'].to_numpy() #converting actual output to numpy array
confusion_matrix11 = calculate_confusion_matrix(actual_values, confff, 3)
plot_confusion_matrix(confusion_matrix11, class_names=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])

#########################################################################
#########################################################################
#             Function to calculate KNN_Weighted
##########################################################################
def KNN_Weighted(df_train, df_test, K, features2):
    answers2  = np.zeros((df_test.shape[0])) #same as previous
    nearestKpoints2 = np.zeros((df_test.shape[0], K), dtype=object) #same as previous
    for i in range(df_test[features2].shape[0]):
        distances2 = [] #same as above
        sum0 = 0 #initializing sum of 1/distances of each class to 0
        sum1 = 0 #initializing sum of 1/distances of each class to 0
        sum2 = 0 #initializing sum of 1/distances of each class to 0
        for j in range(df_train[features2].shape[0]):
            dist2 = euclidean_distance(df_train.iloc[j][features2], df_test.iloc[i][features2]) #same as previous
            distances2.append((dist2, j)) #same as previous
        distances2 = sorted(distances2)[:K] #same as previous
        nearestKpoints2[i] = distances2 #same as previous
        indices2 = [t[1] for t in nearestKpoints2[i]] #same as previous
        dists2 = [t[0] for t in nearestKpoints2[i]]     #same as previous but stores distances   
        arr2 = np.zeros((K))
        for m in range(K):
            arr2[m] = df_train['Species'].iloc[indices2[m]] #output of train data.
            
        for l in range(K): #add distance of respective class to respective sum
            if arr2[l] == 0:
                sum0 += (1 / dists2[l])
            elif arr2[l] == 1:
                sum1 += (1 / dists2[l])
            elif arr2[l] == 2:
                sum2 += (1 / dists2[l])
        #class depends on maximum sum value of (1/d)
        if max(sum0, sum1, sum2) == sum0:
            answers2[i] = 0
        elif max(sum0, sum1, sum2) == sum1:
            answers2[i] = 1
        elif max(sum0, sum1, sum2) == sum2:
            answers2[i] = 2

    return answers2


#######################################################
############ Experiment 2 ##########################
#################################################
K_values2 = [1, 3, 5, 10, 20] #same as previous
K_list2 = [] #same as previous
accuracy_list2 = []
for K in K_values2:
    result2 = KNN_Weighted(df_train, df_test, K, features)
    acc2 = calculate_accuracy(df_test['Species'], result2)
    K_list2.append(K) #same as previous
    accuracy_list2.append(acc2) #same as previous
plt.plot(K_list2, accuracy_list2)
plt.title('Percentage Accuracy vs K for Weighted KNN')
plt.xlabel('Value of K')
plt.ylabel('Accuracy in percentage')
plt.legend({'Accuracy'})
plt.xticks(K_list2)
plt.show()

####plottig confusion matrix###
confff2 = KNN_Weighted(df_train, df_test, 10, features) #same as previous
actual_values1 = df_test['Species'].to_numpy() #same as previous
confusion_matrix111 = calculate_confusion_matrix(actual_values1, confff2, 3) #same as previous
plot_confusion_matrix(confusion_matrix111, class_names=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']) #same as previous

############################################################################

######################################################
############ Experiment 3 ##########################
############################################################
#     Adding Noise and testing the accuracy of results

noise = np.random.normal(loc=0, scale=1, size=(df_train.shape[0], df_train.shape[1])) #created noise as give in assignment
fraction = 0.1 #to get 10% noise
num_samples_with_noise = int(fraction * df_train.shape[0])#no. of samples is 10%
indices_with_noise = np.random.choice(df_train.shape[0], num_samples_with_noise, replace=False) #getting the indicies
df_train_with_noise = df_train.copy()
df_train_with_noise.iloc[indices_with_noise, :] += noise[indices_with_noise, :] #adding the noise

#testing the normal knn with the noised data
ans_normal = KNN_Normal(df_train_with_noise, df_test, 10, features)
acc_normal = calculate_accuracy(df_test['Species'], ans_normal)
acc_normal

#testing the weighted knn with the noised data
ans_weighted = KNN_Weighted(df_train_with_noise, df_test, 10, features)
acc_weighted = calculate_accuracy(df_test['Species'], ans_weighted)
acc_weighted

####################################################
############ Experiment 4 ##########################
################################################
#Curse of Dimentionality
#First we extract the features as per our given question
petal_features = ['PetalLengthCm', 'PetalWidthCm']
sepal_features = ['SepalLengthCm', 'SepalWidthCm']
length_features = ['SepalLengthCm', 'PetalLengthCm']
width_features = ['SepalWidthCm', 'PetalWidthCm']

#storing accuracy of five different cases in  accuracy_curse list
accuracy_curse = []
dimensions = [features, petal_features, sepal_features, length_features, width_features]
for i in dimensions:
    result3 = KNN_Normal(df_train, df_test, 10, i)
    accuracy3 = calculate_accuracy(df_test['Species'], result3)
    accuracy_curse.append(accuracy3)
    
#Plotting accuracy curse against the given 5 cases of dimensions
dimensions_lists = ['All Features','petal_features', 'sepal_features', 'length_features', 'width_features']
plt.plot(dimensions_lists, accuracy_curse)
plt.title('Curse of Dimentionality')
plt.xlabel('Dimensions')
plt.legend({'Accuracy'})
plt.ylabel('Accuracy')
plt.show()


