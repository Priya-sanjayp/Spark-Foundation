#!/usr/bin/env python
# coding: utf-8

# The Sparks Foundation

# Task 5 : Task-6 Prediction using Decision Tree Algorithm

# Create the Decision Tree classifier and visualize it graphically.

# The purpose is if we feed any new data to this classifier, it would be able to predict the right class accordingly.

# Github link: https://github.com/Priya-sanjayp/Spark-Foundation

# # importing libraries

# In[2]:


import sklearn.datasets as datasets
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split    
from sklearn import tree
from matplotlib import pyplot as plt
import numpy as np


# In[3]:


iris=datasets.load_iris() #importing dataset
iris


# In[4]:


iris.feature_names


# In[5]:


iris.target_names


# In[6]:


iris.target


# # Splitting and training model

# In[7]:


x=iris.data
y=iris.target           #spliting the datset into 40% test data and 60% train data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)


# In[8]:


Dtree = DecisionTreeClassifier(max_leaf_nodes=3,)    
Dtree.fit(x_train,y_train)


# In[9]:


Dtree.score(x_train,y_train)


# # Visualising the Graph

# In[10]:


plt.figure(figsize=(15,10))
tree.plot_tree(Dtree, filled=True, rounded=True, feature_names=iris.feature_names,)
plt.show()


# In[11]:


y_pred = Dtree.predict(x_test)
y_pred


# In[ ]:




