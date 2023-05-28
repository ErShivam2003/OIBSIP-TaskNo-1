#!/usr/bin/env python
# coding: utf-8

# # By: Shivam Kumar Patel

# # Oasis Intern : Iris Flower Classification

# #### Iris flower has three species; setosa, versicolor, and virginica, which differs according to their measurements.

# #### Importing Libraries

# In[4]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# #### Importing Dataset

# In[5]:


iris_df = pd.read_csv(r"C:\Users\sp924\Downloads\Iris.csv")
iris_df.head()


# In[6]:


iris_df.tail()


# ### Exploratory Data Analysis

# In[7]:


iris_df.shape


# In[8]:


iris_df.columns


# In[9]:


iris_df.info()


# In[10]:


#dropping unwanted id from data
iris_df.drop('Id',axis=1,inplace = True)


# In[11]:


iris_df.sample(5)


# In[12]:


#checking for missing values
iris_df.isnull().sum()


# In[13]:


iris_df.duplicated().sum()


# In[14]:


iris_df.drop_duplicates(keep='first',inplace=True)


# In[15]:


iris_df.nunique()


# In[16]:


iris_df.describe()


# In[17]:


iris_df['SepalLengthCm'].unique()


# In[18]:


iris_df['PetalLengthCm'].unique()


# In[19]:


iris_df['PetalWidthCm'].unique()


# In[20]:


iris_df['Species'].value_counts()


# In[21]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.boxplot(y="SepalLengthCm",x="Species",data=iris_df)

plt.subplot(2,2,2)
sns.boxplot(y="SepalLengthCm",x="Species",data=iris_df)

plt.subplot(2,2,3)
sns.stripplot(x="Species",y="PetalLengthCm",data=iris_df,hue='Species')

plt.subplot(2,2,4)
sns.stripplot(x="Species",y="PetalLengthCm",data=iris_df,hue='Species')

plt.show()


# In[22]:


iris1=iris_df[iris_df['Species']=='Iris-satosa']
iris2=iris_df[iris_df['Species']=='Iris-versicolor']
iris3=iris_df[iris_df['Species']=='Iris-virginica']


# In[23]:


plt.scatter(iris1['SepalLengthCm'],iris1['SepalWidthCm'],color='yellow',label='Iris-satosa')
plt.scatter(iris2['SepalLengthCm'],iris2['SepalWidthCm'],color='red',label='Iris-versicolor')
plt.scatter(iris3['SepalLengthCm'],iris3['SepalWidthCm'],color='black',label='Iris-virginica')

plt.legend()


# In[24]:


plt.scatter(iris1['PetalLengthCm'],iris1['PetalWidthCm'],color='yellow',label='Iris-satosa')
plt.scatter(iris2['PetalLengthCm'],iris2['PetalWidthCm'],color='red',label='Iris-versicolor')
plt.scatter(iris3['PetalLengthCm'],iris3['PetalWidthCm'],color='black',label='Iris-virginica')

plt.legend()


# In[25]:


sns.pairplot(data=iris_df,hue='Species')


# #### Model Training

# In[26]:


le = LabelEncoder()


# In[27]:


iris_df['Species'] = le.fit_transform(iris_df['Species'])


# In[28]:


iris_df.sample(5)


# In[29]:


X = iris_df.iloc[:,0:4]
Y = iris_df.iloc[:,4]


# In[30]:


print(X.sample(5))
print(X.shape)


# In[31]:


print(Y.sample(5))
print(Y.shape)


# In[32]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y , test_size=0.2,random_state=2)


# In[33]:


svc=SVC()


# In[34]:


svc.fit(X_train,Y_train)


# In[35]:


Y_pred = svc.predict(X_test)


# In[37]:


print(f'Accuracy of the Decision Tree Model : {accuracy_score(Y_pred,Y_test)*100}')


# ###  Tree Model 

# In[38]:


dtc = DecisionTreeClassifier()
dtc.fit(X_train, Y_train)


# In[39]:


Y_pred_2 = dtc.predict(X_test)


# In[41]:


print(f'Accuracy of the Decision Tree Model : {accuracy_score(Y_pred_2,Y_test)*100}')


# ### Thank you 
