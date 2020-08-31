"""
Created on Wed Aug 26 02:13:53 2020

@author: Kemal Sami
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()

import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

# Including csv file as a dataframe ---------------------------------------------------------------
df = pd.read_csv("term-deposit-marketing-2020.csv")

# Data visualization for seeing class weight for discrete feature -------------------------------------
cat_col = ["job","marital","education","contact","month"]
for col in cat_col:
    sns.set()
    plt.figure()
    sns.pairplot(df,size=3.0, hue=col)
    plt.show()