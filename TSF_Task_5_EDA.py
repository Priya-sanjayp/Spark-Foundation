#!/usr/bin/env python
# coding: utf-8

# The Sparks Foundation

# Name:Priyanka Sanjayrao Kadam

# Task 5 : Perform 'Exploratory Data Analysis' on dataset 'Indian Premier League'

# datasetlink:

# # Importing the libraries

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Loading the data

# In[2]:


deli_df = pd.read_csv(r'G:\TSF\task5\deliveries.csv', encoding='latin-1')
match_df = pd.read_csv(r'G:\TSF\task5\matches.csv', encoding='latin-1')
deli_df.head()


# In[3]:


match_df.head()


# In[4]:


# Let us get some basic stats #
print("Number of matches played so far : ", match_df.shape[0])
print("Number of seasons : ", len(match_df.season.unique()))


# Number of matches each season:

# Let us first look at the number of matches played per season.

# In[5]:


sns.countplot(x='season', data=match_df)
plt.show()


# There is a spike in the middle for three years where the number of matches are more than 70.

# Number of matches in each venue:

# In[6]:


plt.figure(figsize=(14,6))
sns.countplot(x='venue', data=match_df)
plt.xticks(rotation='vertical')
plt.show()


# There are quite a few venues present in the data with "M Chinnaswamy Stadium" being the one with most number of matches followed by "Eden Gardens".

# Number of matches played by each team:

# In[7]:


temp_df = pd.melt(match_df, id_vars=['id','season'], value_vars=['team1', 'team2'])

plt.figure(figsize=(12,6))
sns.countplot(x='value', data=temp_df)
plt.xticks(rotation='vertical')
plt.show()


# "Mumbai Indians" lead the pack with most number of matches played followed by "Royal Challengers Bangalore".

# Number of wins per team:

# In[8]:


plt.figure(figsize=(12,6))
sns.countplot(x='winner', data=match_df)
plt.xticks(rotation='vertical')
plt.show()


# MI again leads the pack followed by CSK.

# Champions each season:

# Now let us see the champions in each season.

# In[9]:


temp_df = match_df.drop_duplicates(subset=['season'], keep='last')[['season', 'winner']].reset_index(drop=True)
temp_df


# Toss decision:

# Let us see the toss decisions taken so far.

# In[10]:


temp_series = match_df.toss_decision.value_counts()
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))
colors = ['gold', 'lightskyblue']
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.title("Toss decision percentage")
plt.show()


# Almost 55% of the toss decisions are made to field first. Now let us see how this decision varied over time.

# In[11]:


plt.figure(figsize=(12,6))
sns.countplot(x='season', hue='toss_decision', data=match_df)
plt.xticks(rotation='vertical')
plt.show()


# Since there is a very strong trend towards batting second let us see the win percentage of teams batting second.

# In[12]:


num_of_wins = (match_df.win_by_wickets>0).sum()
num_of_loss = (match_df.win_by_wickets==0).sum()
labels = ["Wins", "Loss"]
total = float(num_of_wins + num_of_loss)
sizes = [(num_of_wins/total)*100, (num_of_loss/total)*100]
colors = ['gold', 'lightskyblue']
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.title("Win percentage batting second")
plt.show()


# So percentage of times teams batting second has won is 53.2. Now let us split this by year and see the distribution.

# In[13]:


# create a function for labeling #
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.02*height,
                '%d' % int(height),
                ha='center', va='bottom')


# Top players of the match:

# In[14]:


temp_series = match_df.player_of_match.value_counts()[:10]
labels = np.array(temp_series.index)
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots()
rects = ax.bar(ind, np.array(temp_series), width=width, color='y')
ax.set_xticks(ind+((width)/2.))
ax.set_xticklabels(labels, rotation='vertical')
ax.set_ylabel("Count")
ax.set_title("Top player of the match awardees")
autolabel(rects)
plt.show()


# Top Umpires:

# In[15]:


temp_df = pd.melt(match_df, id_vars=['id'], value_vars=['umpire1', 'umpire2'])

temp_series = temp_df.value.value_counts()[:10]
labels = np.array(temp_series.index)
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots()
rects = ax.bar(ind, np.array(temp_series), width=width, color='r')
ax.set_xticks(ind+((width)/2.))
ax.set_xticklabels(labels, rotation='vertical')
ax.set_ylabel("Count")
ax.set_title("Top Umpires")
autolabel(rects)
plt.show()


# Dharmasena seems to be the most sought after umpire for IPL matches followed by Ravi. Others are fairly close to each other.

# How lucky are the toss winning teams?:

# Now let us see how lucky is the toss winner. Do they often end up on the winning side?! Let us see.

# It seems for Chennai Super Kings (CSK) winning the toss is an indication of winning the match with high probability.

# On the other hand, Pune Warriors end up losing the matches more often when they won the toss.

# So far we have looked at the match data to get insights. Now let us look at the delivery dataset which is more granular to gain some more insights. To start with, let us look at the top few rows.

# In[18]:


match_df.head()


# Batsman analysis:

# Let us start our analysis with batsman. Let us first see the ones with most number of IPL runs under their belt.

# In[24]:


temp_df = deli_df.groupby('batsman')['batsman_runs'].agg('sum').reset_index().sort_values(by='batsman_runs', ascending=False).reset_index(drop=True)
temp_df = temp_df.iloc[:10,:]

labels = np.array(temp_df['batsman'])
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots()
rects = ax.bar(ind, np.array(temp_df['batsman_runs']), width=width, color='blue')
ax.set_xticks(ind+((width)/2.))
ax.set_xticklabels(labels, rotation='vertical')
ax.set_ylabel("Count")
ax.set_title("Top run scorers in IPL")
autolabel(rects)
plt.show()


# Gambhir is way ahead of others

# Now let us check the number of 6's

# In[27]:


temp_df = deli_df.groupby('batsman')['batsman_runs'].agg(lambda x: (x==6).sum()).reset_index().sort_values(by='batsman_runs', ascending=False).reset_index(drop=True)
temp_df = temp_df.iloc[:10,:]

labels = np.array(temp_df['batsman'])
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots()
rects = ax.bar(ind, np.array(temp_df['batsman_runs']), width=width, color='g')
ax.set_xticks(ind+((width)/2.))
ax.set_xticklabels(labels, rotation='vertical')
ax.set_ylabel("Count")
ax.set_title("Batsman with most number of sixes.!")
autolabel(rects)
plt.show()


# There you see the big man. Gayle, the unassailable leader in the number of sixes.

# Raina is third in both number of 4's and 6's

# Now let us see the batsman who has played the most number of dot balls.

# In[28]:


temp_df = deli_df.groupby('batsman')['batsman_runs'].agg(lambda x: (x==0).sum()).reset_index().sort_values(by='batsman_runs', ascending=False).reset_index(drop=True)
temp_df = temp_df.iloc[:10,:]

labels = np.array(temp_df['batsman'])
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots()
rects = ax.bar(ind, np.array(temp_df['batsman_runs']), width=width, color='c')
ax.set_xticks(ind+((width)/2.))
ax.set_xticklabels(labels, rotation='vertical')
ax.set_ylabel("Count")
ax.set_title("Batsman with most number of dot balls.!")
autolabel(rects)
plt.show()


# It is interesting to see that the same names repeat again here as well. I think since these guys have played more number of balls, they have more dot balls as well.

# In[30]:


plt.figure(figsize=(12,6))
sns.countplot(x='dismissal_kind', data=deli_df)
plt.xticks(rotation='vertical')
plt.show()


# Caught is the most common dismissal type in IPL followed by Bowled. There are very few instances of hit wicket as well. 'Obstructing the field' is one of the dismissal type as well in IPL.!

# In[ ]:




