# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 22:02:05 2020

@author: Kemal Sami
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')

# Including csv file as a dataframe ---------------------------------------------------------------
df = pd.read_csv("term-deposit-marketing-2020.csv")

# Data label encoding for ML ----------------------------------------------------------------------
df_without_labeling = df.iloc[:,[0,5,9,11,12]]
df_for_labeling = df.iloc[:,[1,2,3,4,6,7,8,10,13]]
df_encoded = df_for_labeling.astype(str).apply(le.fit_transform)

# Data splitting for feature analysis -------------------------------------------------------------
frames = [df_without_labeling,df_encoded]
df_for_ML = pd.concat(frames,axis=1)
features = df_for_ML.iloc[:,0:13].astype(np.float32)
targets = df_for_ML.iloc[:,13]

# Analyzing of the features based on entropy estimation from k-nearest neighbors distances --------
information_gain = pd.DataFrame(index=df_for_ML.drop(["y"],axis=1).keys(),
                                data=mutual_info_classif(features, targets, discrete_features=True,
                                                         n_neighbors=3)
                                )

# Analyzing features based on f-test, a kind of variance analysis method --------------------------
f_val,p_val= f_classif(features, targets)
f_val = pd.DataFrame(index=df_for_ML.drop(["y"],axis=1).keys(),data=f_val)
p_val = pd.DataFrame(index=df_for_ML.drop(["y"],axis=1).keys(),data=p_val)

# Analyzing features based on correlation analysis method -----------------------------------------
corrmat = df_for_ML.corr()
# Visualising of the correlation matrix -----------------------------------------------------------
f, ax = plt.subplots(figsize=(13,13))
sns.heatmap(corrmat, 
            cmap='twilight', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True)