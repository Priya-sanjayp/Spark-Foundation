#!/usr/bin/env python
# coding: utf-8

# # TASK02: K- Means Clustering

# # The Spark Foundation

# Name: Priyanka kadam

# importing library

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets


# Importing the dataset

# In[2]:


df=pd.read_csv(r"G:\TSF\task2\Iris.csv")


# In[3]:


df.head(10)


# In[4]:


df.tail(10)


# In[5]:


df.describe()


# In[14]:


df.shape


# In[6]:


df.isnull().sum()


# In[7]:


df.info()


# How do you find the optimum number of clusters for K Means? How does one determine the value of K?

# Fitting the Elbow Method to find the actual number of clusters

# Train Test Split

# In[8]:


x = df.iloc[:, [0, 1, 2, 3]].values


# In[9]:


from sklearn.cluster import KMeans
wcss = []
for i in range(1, 20):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


# In[10]:


plt.plot(range(1, 20), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In the Elbow method the number of Cluster prediction is true. So this Technique indiactes a number of Clusters=3

# # Model Development with K-Means Clustering

# In[11]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++',max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)


# # Visualising the Clusters

# In[12]:


# Visualising the clusters - On the first two columns
plt.scatter(x[y_kmeans ==0,0], x[y_kmeans == 0,1], s=100, c='red', label='setosa')
plt.scatter(x[y_kmeans ==1,0], x[y_kmeans ==1,1], s=100, c='orange', label='versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],s = 100, c = 'yellow', label = 'virginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'blue', label = 'Centroids')

plt.legend()


# From the above visualization we see that one species can be easily clusterd which is not the case with the other two. Which may be the reason why the Dendograms predicted 2 Clusters.

# Further K-Means with the Elbow method did an amazing job at predicting the number of true Clusters which is 3. The above visualization is the clustered graph.

# In[ ]:




