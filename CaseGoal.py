"""
Created on Wed Aug 25 00:09:45 2020

@author: Kemal Sami
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
#from sklearn.modelselection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense,Dropout

# Including csv file as a dataframe -------------------------------------------------------------------
df = pd.read_csv("term-deposit-marketing-2020.csv")

# Data label encoding for ML --------------------------------------------------------------------------
le= LabelEncoder()
df_without_labeling = df.iloc[:,[0,5,9,11,12]]
df_for_labeling = df.iloc[:,[1,2,3,4,6,7,8,10,13]]
df_encoded = df_for_labeling.astype(str).apply(le.fit_transform)

# Data convertioning in dataframe to array for some preprocessing -------------------------------------
frames = [df_without_labeling,df_encoded]
df_for_ML = pd.concat(frames,axis=1)
data = df_for_ML.drop(["default"],axis=1).to_numpy()

# Putting data into a for loop because of calculating 5 fold cross validation--------------------------  
score = [0.0]*5
for i in range(5): # range = 5 because "k-fold" 's k is 5
   
    # Data splitting for train and test of model ------------------------------------------------------
    data = np.random.permutation(data)# shuffle data for 
                                      # equalizing to chance of positive samples in the test and train
                                      # and changing data within train and test dataset
    features = data[:,0:12].astype(np.float32)
    targets = data[:,12].reshape(40000,1)
    num_val_samples = int(len(features) * 0.2)
    train_features = features[:-num_val_samples]
    train_targets = targets[:-num_val_samples]
    val_features = features[-num_val_samples:]
    val_targets = targets[-num_val_samples:]
    
    # Weight determining for train samples ------------------------------------------------------------
    counts = np.bincount(train_targets[:, 0])
    print(
        "Number of positive samples in training data for {}. loop: {} ({:.2f}% of total)".format(i+1,
            counts[1], 100 * float(counts[1]) / len(train_targets)
        )
    )
    weight_for_0 = 1.0 / counts[0]
    weight_for_1 = 1.0 / counts[1]
    
    # Normalizing to data for increasing metrics using training data ----------------------------------
    mean = np.mean(train_features, axis=0)
    train_features -= mean
    val_features -= mean
    std = np.std(train_features, axis=0)
    train_features /= std
    val_features /= std
    
    # Building a sequential neural network for binary classification ----------------------------------
    model = Sequential()
    model.add(Dense(256,activation='relu',input_dim=12))
    model.add(Dropout(0.5))
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1,activation='sigmoid'))
    
    # Compiling and fitting model ---------------------------------------------------------------------
    model.compile(optimizer='adam',loss="binary_crossentropy",metrics=['accuracy'])
    
    class_weight = {0: weight_for_0, 1: weight_for_1}
    
    model_fit=model.fit(
        train_features,
        train_targets,
        batch_size=2048,
        epochs=200,
        verbose=0, # prints score for each epoch. If you want to see them, you can change value as 1
        validation_data=(val_features, val_targets),
        class_weight=class_weight,
    )
    score[i]=model_fit.history['val_acc'][len(model_fit.history['val_acc'])-1]
    
# Calculating average score ------------------------------------------------------------------------
score_avg=np.mean(score)
score_std=np.std(score)
print("The average score according to the 5-fold cross-validation method is: {:.2f}%".format(score_avg*100))
print("and the standard deviation is: ",score_std)