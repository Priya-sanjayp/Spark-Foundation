#!/usr/bin/env python
# coding: utf-8

# # The Sparks Foundation

# Github link:https://github.com/Priya-sanjayp/Spark-Foundation

# Name: Priyanka Kadam

# # Importing the Libraries

# In[1]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# # Loading the dataset

# In[2]:


df=pd.read_csv(r'C:\Users\user\Downloads\Global Terrorism - START data.zip', encoding='latin-1')


# In[3]:


df.head(5)


# In[4]:


df.tail(5)


# In[5]:


df.shape


# In[6]:


df.columns.values


# In[7]:


df.count()


# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


df.isnull().sum()


# In[11]:


missing_percentage = df.isnull().sum()*100/len(df)
missing_percentage


# In[12]:


df.dropna(axis=1,inplace=True)


# In[13]:


df.isnull().sum()


# In[14]:


df.corr()


# # Data Visualization

# In[15]:


sns.countplot(x=df['iyear'])
plt.ylabel('Number of attacks')
plt.xticks(rotation=90)
plt.xlabel('Attack Year')
plt.title('Increase in Terror',size=25, fontweight="bold")


# In[16]:


sns.countplot(x=df['iyear'], hue='success', data=df)
plt.ylabel('Number of attacks')
plt.xticks(rotation=90)
plt.xlabel('Attack Year')
plt.title('Increase in Terror',size=25, fontweight="bold")


# In[17]:


sns.countplot(x=df['region_txt'], hue='success', data=df)
plt.ylabel('Number of attacks')
plt.xticks(rotation=90)
plt.xlabel('Attack Region')


# In[18]:


sns.barplot(x=df['attacktype1_txt'].value_counts()[:20].index,y=df['attacktype1_txt'].value_counts()[:20].values)
plt.ylabel('Total Attack', fontsize=20)
plt.xticks(rotation=90)
plt.xlabel('Attack Type', fontsize=20)
plt.title('Attack types',size=25, fontweight="bold")


# # Final Report

# Iraq ranked first on the global terrorism for their terrorist activity followed by Pakistan, Afganistan, India, and so on

# Most targeted areas are private citizens and property, military, police, and so on.

# Global terror attack deaths rose sharply starting year 2011

# In conclusion with the ranking, Iraq suffered from most terrorist attacks in 2014, with the most deaths in that year

# In[ ]:




