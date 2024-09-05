# Syed Mohamad Tawseeq
# 22CH10090
# MLFA ASSIGNMENT Naive Bayes

#first we will import 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#reading the data and doing some sata preprocssing and labeling/replacing species with numbers
data = pd.read_csv('Iris.csv')
columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
unique_categories = data['Species'].unique()
category_mapping = {category: i+1 for i, category in enumerate(unique_categories)}
data['Species'] = data['Species'].replace(category_mapping)

#some import functions like accuracy score, likelihood,naive bayes classifier and prior. 
def accuracy_score(y_true, y_pred):
    correct = 0
    total = len(y_true)
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct += 1
    
    return correct / total

def calculate_likelihood_categorical(df, feat_name, feat_val, Y, label):
    feat = list(df.columns)
    df = df[df[Y]==label]
    p_x_given_y = len(df[df[feat_name]==feat_val]) / len(df)
    return p_x_given_y
#Exercise 1
def naive_bayes_categorical(df, X, Y):
    # getingfeature names
    features = list(df.columns)[:-1]
    # calculating the prior from the function defined below
    prior = calculate_prior(df, Y)
    Y_pred = []
    # looping over every data sample
    for index, x in X.iterrows():
        # calculating likelihood
        labels = sorted(list(df[Y].unique()))
        likelihood = [1] * len(labels)
        for j in range(len(labels)):
            for feat_name in features:
                feat_val = x[feat_name]  # Accessing values using column name
                likelihood[j] *= calculate_likelihood_categorical(df, feat_name, feat_val, Y, labels[j])
        # calculating posterior probability ( but numerator only as we only need to compare)
        post_prob = [1] * len(labels)
        for j in range(len(labels)):
            post_prob[j] = likelihood[j] * prior[j]

        Y_pred.append(labels[np.argmax(post_prob)])

    return Y_pred


def calculate_prior(df, Y):
    classes = sorted(list(df[Y].unique()))
    prior = []
    for i in classes:
        prior.append(len(df[df[Y]==i])/len(df))
    return prior
#setting k = 2, 3, 5 and respective possible labels in lists an iterating over them
K = [2, 3, 5]
labels = [[0, 1], [0, 1, 2], [0, 1, 2, 3, 4]]  # Adjust labels to match the number of bins
accuracy = {} #empty list to store accuracy
d1 = data.copy()  #copying the data, so that after each loop , we get the same data back
for i in range(len(K)):
    data = d1.copy()
    #discretizing each feature values as they are continous using bins and labels
    data["cat_SepalLengthCm"] = pd.cut(data["SepalLengthCm"].values, bins=K[i], labels=labels[i])
    data["cat_SepalWidthCm"] = pd.cut(data["SepalWidthCm"].values, bins=K[i], labels=labels[i])
    data["cat_PetalLengthCm"] = pd.cut(data["PetalLengthCm"].values, bins=K[i], labels=labels[i])
    data["cat_PetalWidthCm"] = pd.cut(data["PetalWidthCm"].values, bins=K[i], labels=labels[i])

    newcolumns = ["cat_SepalLengthCm", "cat_SepalWidthCm", "cat_PetalLengthCm", "cat_PetalWidthCm"]
    #creating the new dataset using new columns and adding species to it
    data_binned = data[newcolumns + ['Species']]
    shuffled_df = data_binned.sample(frac=1, random_state=42)  #random state makes sure to that we get same value every time we run the code
    df_train = shuffled_df[:int(0.8 * shuffled_df.shape[0])]  #first 80% elements of the shuffled data
    df_test = shuffled_df[int(0.8 * shuffled_df.shape[0]):]  # last 20% elements of the shuffled data

    X_test = df_test[newcolumns]
    Y_test = df_test['Species']

    Y_pred = naive_bayes_categorical(df_train, X=X_test, Y="Species") #using the classifies we defined as a function above

    accuracy[K[i]] = accuracy_score(Y_test, Y_pred)
    print(f"Accuracy score for K={K[i]}: {accuracy[K[i]]}")
    print('-------------------------------------------------')

# Converting dictionary keys and values to lists so that we can plot them easily
K_values = list(accuracy.keys())
accuracy_values = list(accuracy.values())
#plotting
plt.plot(K_values, accuracy_values)
plt.xlabel('Number of Bins (K)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Bins (K)')
plt.show()

def add_noise(data, fraction, mean=0, std_dev=2.0, random_state=42):
    #Adding noise to a fraction of the data using a normal distribution.
    noisy_data = data.copy() #copying the data
    # Converting category columns to numerical data typee
    for col in noisy_data.select_dtypes(include='category').columns:
        noisy_data[col] = noisy_data[col].cat.codes.astype('float64')
    #the number of samples to add noise to
    num_samples = int(fraction * len(noisy_data))
    # random seed
    np.random.seed(random_state)
    #randomly selecting samples to add noise
    idx = np.random.choice(len(noisy_data), size=num_samples, replace=False)
    # adding noise to the selected samples
    noise = np.random.normal(loc=mean, scale=std_dev, size=(num_samples, len(noisy_data.columns)-1))
    noisy_data.iloc[idx, :-1] += noise

    return noisy_data


# Experiment 2: Adding noise to a fraction of the training data and evaluate Naive Bayes performance
noise_fractions = [0.1, 0.4, 0.8, 0.9]  # Defining noise fractions as given in the problem
accuracy_with_noise = {}  # Dictionary to store accuracy scores for each noise fraction
# Looping over each noise fraction
for fraction in noise_fractions:
    # Adding noise to the training data
    noisy_train_data = add_noise(df_train, fraction=fraction)
    #Using Naive Bayes classifier using the optimal K found in Experiment 1
    Y_pred = naive_bayes_categorical(noisy_train_data, X=X_test, Y="Species")
    #Calculating accuracy score
    accuracy = accuracy_score(Y_test, Y_pred)
    #storing the accuracy score for this noise fraction
    accuracy_with_noise[fraction] = accuracy

    #Printing the accuracy scores for each noise fraction
for fraction, accuracy in accuracy_with_noise.items():
    print(f"Noise Fraction: {fraction}, Accuracy: {accuracy}")

