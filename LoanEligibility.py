#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv('/Users/mohamadkatranji/Downloads/training_data.csv')
data.head()


# In[3]:


data['Dependents'] = data['Dependents'].replace(['3+'],3)
data['Dependents'] = data['Dependents'].replace(['0'],0)
data['Dependents'] = data['Dependents'].replace(['1'],1)
data['Dependents'] = data['Dependents'].replace(['2'],2)
data=data.drop('Loan_ID',axis=1)
data.head()


# In[4]:


data.info()


# In[5]:


data.isnull().sum()


# In[6]:


data.describe()


# In[7]:


data['LoanAmount'].fillna(data['LoanAmount'].mean(), inplace=True)
data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mean(), inplace=True)
data['Credit_History'].fillna(data['Credit_History'].mean(), inplace=True)


# In[8]:


data.isnull().sum()


# In[9]:


data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)
data['Married'].fillna(data['Married'].mode()[0], inplace=True)
data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)
data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0], inplace=True)
data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)
data['Self_Employed'].fillna(data['Self_Employed'].mode()[0], inplace=True)


# In[10]:


data.isnull().sum()


# In[11]:


data['TotalIncome']=data['ApplicantIncome']+data['CoapplicantIncome']
data.head()


# In[12]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
data.iloc[:,0] = labelencoder.fit_transform(data.iloc[:,0])
data.iloc[:,1] = labelencoder.fit_transform(data.iloc[:,1])
data.iloc[:,3] = labelencoder.fit_transform(data.iloc[:,3])
data.iloc[:,4] = labelencoder.fit_transform(data.iloc[:,4])
data.iloc[:,10] = labelencoder.fit_transform(data.iloc[:,10])
data.iloc[:,11] = labelencoder.fit_transform(data.iloc[:,11])
data.head()


# In[13]:


#sns.pairplot(data)


# In[14]:


X = data.iloc[:,np.r_[0:5,7:11,12]].values
y = data.iloc[:,11].values
print(X)


# In[15]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state = 0)


# In[16]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[17]:


from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression(max_iter=1000, random_state = 0)
logisticRegr.fit(X_train, y_train)


# In[18]:


from sklearn.model_selection import cross_val_score
scores_1 = cross_val_score(logisticRegr, X, y, cv=5)
print(" Accuracy for logistic regression after using cross validation: " ,scores_1.mean())


# In[19]:


from sklearn.tree import DecisionTreeClassifier
clf_entropy =  DecisionTreeClassifier(criterion = 'entropy', max_depth = None, random_state = 0)
clf_entropy.fit(X_train, y_train)


# In[20]:


scores_2 = cross_val_score(clf_entropy, X, y, cv=5)
print(" Accuracy for Decision Tree after using cross validation: " ,scores_2.mean())


# In[21]:


from sklearn.neighbors import KNeighborsClassifier
KNN_classifier = KNeighborsClassifier(n_neighbors = 11, metric = 'euclidean')
KNN_classifier.fit(X_train, y_train)


# In[22]:


scores_3 = cross_val_score(KNN_classifier, X, y, cv=5)
print(" Accuracy for KNN after using cross validation: " ,scores_3.mean())


# In[23]:


from sklearn.ensemble import RandomForestClassifier
Random_classifier = RandomForestClassifier(n_estimators = 11, criterion = 'entropy', random_state = 0)
Random_classifier.fit(X_train, y_train)


# In[24]:


y_pred = Random_classifier.predict(X_test)
from sklearn import metrics
print('Accuracy for Random Forest Classification is: ', metrics.accuracy_score(y_pred, y_test))


# In[25]:


test = pd.read_csv('/Users/mohamadkatranji/Downloads/testing_data.csv')


# In[26]:


test.head()


# In[27]:


test.isnull().sum()


# In[28]:


test['LoanAmount'].fillna(test['LoanAmount'].mean(), inplace=True)
test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].mean(), inplace=True)
test['Credit_History'].fillna(test['Credit_History'].mean(), inplace=True)
test.isnull().sum()


# In[29]:


test['Gender'] = test['Gender'].fillna(test['Gender'].mode()[0])
test['Married'] = test['Married'].fillna(test['Married'].mode()[0])
test['Dependents'] = test['Dependents'].fillna(test['Dependents'].mode()[0])
test['Self_Employed'] = test['Self_Employed'].fillna(test['Self_Employed'].mode()[0])
test.isnull().sum()


# In[30]:


test['Total_Income'] = test['ApplicantIncome']+ test['CoapplicantIncome']
test=test.drop('Loan_ID',axis=1)
test['Dependents'] = test['Dependents'].replace(['3+'],3)
test['Dependents'] = test['Dependents'].replace(['0'],0)
test['Dependents'] = test['Dependents'].replace(['1'],1)
test['Dependents'] = test['Dependents'].replace(['2'],2)
test.head()


# In[31]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
test.iloc[:,0] = labelencoder.fit_transform(test.iloc[:,0])
test.iloc[:,1] = labelencoder.fit_transform(test.iloc[:,1])
test.iloc[:,3] = labelencoder.fit_transform(test.iloc[:,3])
test.iloc[:,4] = labelencoder.fit_transform(test.iloc[:,4])
test.iloc[:,10] = labelencoder.fit_transform(test.iloc[:,10])
test.iloc[:,11] = labelencoder.fit_transform(test.iloc[:,11])
test.head()


# In[32]:


m = test.iloc[:,np.r_[0:5,7:12]].values
m = sc.fit_transform(m)
print(m)


# In[33]:


pred_logisticregression = logisticRegr.predict(m)
print("results for loan status if logistic regresssion is used: ")
print(pred_logisticregression)


# In[34]:


pred_DecisionTreeClassifier = clf_entropy.predict(m)
print("results for loan status if decision trees is used: ")
print(pred_DecisionTreeClassifier)


# In[35]:


pred_KNeighborsClassifier = KNN_classifier.predict(m)
print("results for loan status if K neighbors classifier is used: ")
print(pred_KNeighborsClassifier)


# In[36]:


pred_RandomForestClassifier = Random_classifier.predict(m)
print("results for loan status if random Forest classifier is used: ")
print(pred_RandomForestClassifier)


# In[ ]:


#as we can see the random forest classifier performed the best out of all the models
#and such should be used in the future to predict loan status

