#!/usr/bin/env python
# coding: utf-8

# Task 3: Perform 'Exploratory Data Analysis' on 'SampleSuperstore' dataset

# name: Priyanka Kadam

# DataSet:https://drive.google.com/file/d/1lV7is1B566UQPYzzY8R2ZmOritTW299S/view

# # Exploratory Data Analysis

# IMPOTING LIBRARIES

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # LODING THE DATA

# In[2]:


df=pd.read_csv(r"G:\TSF\task3\SampleSuperstore.csv")


# In[3]:


df.head(10)


# In[4]:


df.tail(9)


# In[5]:


df.shape


# In[6]:


df.columns.values


# In[7]:


df.info()


# In[8]:


df.isnull().sum()


# In[9]:


df.describe()


# In[10]:


df.duplicated().sum()


# In[11]:


df[df.duplicated(keep=False)]


# In[12]:


df.drop_duplicates(subset=None, keep='first', inplace=True)


# In[13]:


df.duplicated().sum()


# In[14]:


df.head()


# # Visualization

# In[15]:


plt.bar(df['Quantity'], df['Profit'])
plt.title(' Profit Vs Quantity', fontsize=14)
plt.xlabel('Quantity', fontsize=14)
plt.ylabel('Profit', fontsize=14)
plt.grid(True)
plt.show()


# In[16]:


plt.bar(df['Quantity'], df['Sales'])
plt.title(' Quantity Vs Sales', fontsize=14)
plt.xlabel('Quantity', fontsize=14)
plt.ylabel('Sales', fontsize=14)
plt.grid(True)
plt.show()


# In[17]:


plt.figure(figsize=(16,6))
sns.relplot(x='Quantity', y='Profit',color='purple',hue="Segment", data=df)


# In[18]:


plt.title("Region")
df['Region'].value_counts().plot.pie(autopct='%1.1f%%', shadow = True);


# In[21]:


plt.title("Segment")
df['Segment'].value_counts().plot.pie(autopct='%1.1f%%', shadow = True);


# In[22]:


df['freq'] = df.groupby('City')['City'].transform('count')


# In[23]:


df1 = df.sort_values('freq',ascending = False).groupby('City').head(10)


# In[24]:


df1 = df1[['City', 'freq']]


# In[25]:


df1.duplicated().sum()


# In[26]:


plt.figure(figsize = [14.70, 8.27])
plt.subplots_adjust(wspace = 0.85)
sns.barplot(x = df1['City'], y = df['freq'], color = sns.color_palette()[9]);
plt.xlabel('City')
plt.ylabel('Count');


# In[27]:


n_data=['Sales','Quantity','Discount','Profit']
plt.figure(figsize=(10,5))
sns.heatmap(df[n_data].corr(),annot=True, fmt='.4f',cmap='RdYlGn',center=0 );


# In[28]:


plt.figure(figsize=(20,16))
sns.pairplot(df, hue="Sub-Category") 
plt.show()


# # Final Report

# 1)The business is successful in New York City and Los Angeles

# 2)It needs to focus on Southern and Central part of the Country

# 3)It also needs to work on selling Home Office supplies

# 4)Most of the Postal Code Areas buy products less than 10k

# 5)States like Wyoming and West Virginia needs the attention of the business the most

# In[ ]:




