import csv
import os.path
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import numpy as np
import matplotlib.pyplot as plt
import itertools
# Get keys from names file
def getKeysfromNames(filename):
    try:
        if filename[-6:]!=".names": raise NameError('filename must end with ".names"')
        keys = []
        d={}
        with open(filename, 'r') as file_in:
                    reader = csv.reader(file_in, delimiter=':')
                    for line in reader:
                        if len(line)>1 and not ' ignore.' in line:
                            key = line[0]
                            keys.append(key)
                            d[key] = line[1].split(',')
        return keys, d
    except NameError: print ("filename was incorrect"); raise

def writeNames(xShape, filename):
    with open(filename, 'w') as file_out:
        file_out.write('paymentStatus\n')
        for i in range(xShape[1]):
            file_out.write('feature{}: continuous.\n'.format(i))
        file_out.close()
        print ('Successfully wrote .names file')

def load_data(folderpath, filename=None, split = 0.33):
    # Load train and test data and split target from predictors.
    # If filename is not specified it tries to load all files and concatenate them
    if filename:
        X_train = pd.read_csv(folderpath + filename + '.csv', header=None)
        print ("Train Data Loaded")
        if os.path.isfile(folderpath + 'test_' + filename + '.csv'):
            X_test = pd.read_csv(folderpath + 'test_' + filename + '.csv', header=0)
            y_train = X_train.iloc[:, 0].astype(int)
            X_train.drop(X_train.columns[0], axis=1, inplace = True)
            y_test = X_test.iloc[:, 0].astype(int)
            X_test.drop(X_test.columns[0], axis=1, inplace = True)
            print("Test Data Loaded")
        else:
            print(".test file doesn't exist. Performing Train Test Split")
            Y = X_train.iloc[:, 0].astype(int)
            X_train.drop(0, axis=1, inplace = True)
            X_train, X_test, y_train, y_test = train_test_split(X_train, Y, test_size = split, random_state = 42)
            print("Data Splitted")
        if os.path.isfile(folderpath + filename + '.names'):
            keys, d = getKeysfromNames(folderpath + filename + '.names')
            X_train.columns = X_test.columns = keys[:-1]
            print ("keys loaded !")
        else: print("keys not loaded, because .names file doesn't exist")
    else:
        print ("Load all files from folder and concatenate")
        filenames = os.listdir(folderpath)
        print ("Found {} files".format(len(filenames)))
        dfs = []
        for filename in filenames:
            dfs.append(pd.read_csv(folderpath+filename,header=None))
        # Concatenate all data into one DataFrame
        X = pd.concat(dfs, ignore_index=True)
        Y = X.iloc[:,0]
        X.drop(X.columns[0], axis=1, inplace=True)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=split, random_state=42)
        print("Data Splitted in Train and Test")

    return X_train, X_test, y_train, y_test

def resample(X,Y, bias):
    # bias: Bias - Factor for (  # non-frauds/#frauds)
    # Resampling Biased
    idx_frauds = list(Y.loc[(Y == 1)].index)
    idx_random = list(Y.iloc[random.sample(range(len(Y)), bias * len(idx_frauds))].index)
    idxs = np.concatenate([idx_frauds, idx_random])

    Y_sample = Y.loc[idxs]
    X_sample = X.loc[idxs]

    print("Class 0: {} samples".format((Y_sample == 0).sum()))
    print("Class 1: {} samples".format((Y_sample == 1).sum()))

    return X_sample, Y_sample

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')