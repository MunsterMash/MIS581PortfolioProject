#!/usr/bin/env python
# coding: utf-8

# In[88]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


# medical = pd.read_csv('../documents/KaggleV2-May-2016.csv',parse_dates=['ScheduledDay','AppointmentDay'])
# File needs to be in the same root folder
medical = pd.read_csv('healthcare-dataset-stroke-data.csv')
medical.head()


# In[89]:


# medical['ever_married'].value_counts()
medical['ever_married'] = medical['ever_married'].map({'Yes':1,'No':0})


# In[90]:


le = LabelEncoder()
medical['gender'] = le.fit_transform(medical['gender'])
medical['work_type'] = le.fit_transform(medical['work_type'])
medical['Residence_type'] = le.fit_transform(medical['Residence_type'])
medical['smoking_status'] = le.fit_transform(medical['smoking_status'])
medical.fillna(0)


# In[91]:


medical.head()
medical.to_csv('medical.csv', index=False)


# In[92]:


df_cm = medical.corr(method='pearson', min_periods=1)
df_cm = df_cm.drop(['id'])
sns.set(font_scale=1.4) # for label size
sns.heatmap(df_cm, annot=True, annot_kws={"size": 12},fmt='g') # font size
# plt.figure(figsize=(80, 80))
plt.title("Correlation Matrix")
plt.rcParams["figure.figsize"] = (20,10)

plt.show()


# In[94]:


from scipy.stats.stats import pearsonr

target_col_name = 'stroke'
feature_target_corr = {}
feature_target_corrp = {}

for col in df_cm:
    if target_col_name != col:
        feature_target_corr[col] =             pearsonr(df_cm[col], df_cm[target_col_name])[0]
        feature_target_corrp[col + '_' + target_col_name] =             pearsonr(df_cm[col], df_cm[target_col_name])
print("Feature-Target Correlations")
# correlation coefficient
print(feature_target_corr)
# pvalue
print(feature_target_corrp)
print(type(feature_target_corr))


# In[96]:


myList = feature_target_corr.items()
myList = sorted(myList) 
x, y = zip(*myList) 
plt.xticks(rotation='vertical')
plt.title("Correlation Coefficient to Stroke Event")
plt.ylabel("Correlation Coefficient")
plt.xlabel("Independent Field")

plt.plot(x, y)
plt.show()


# In[97]:


sc = StandardScaler()
medical =medical[~medical.isin([np.nan, np.inf, -np.inf]).any(1)]

scaledData = pd.DataFrame(sc.fit_transform(medical.drop(['id', 'stroke'],axis=1)),columns=medical.drop(['id', 'stroke'],axis=1).columns)

x = scaledData
y = medical['stroke']


# In[98]:


df =pd.DataFrame(medical.drop(['id', 'gender', 'ever_married', 'heart_disease', 'Residence_type' ,'hypertension'],axis=1))
# medical.columns
sns.pairplot(df, hue="stroke")


# In[99]:


df =pd.DataFrame(medical.drop(['id', 'avg_glucose_level', 'bmi', 'age'],axis=1))
# medical.columns
sns.pairplot(df, hue="stroke")


# In[100]:


medical.columns
medical['hypertension'].hist(by=medical['stroke'])
medical['heart_disease'].hist(by=medical['stroke'])

# medical['stroke'].hist(by=medical['Residence_type'])


# In[101]:


xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = .20 ,random_state=32)


# In[102]:


# print (ytrain)
# print(type(xtrain))
# ytrain.isna()
# xtrain.fillna(0)
# ytrain.fillna(0)


# In[103]:



#Packages used for random and decision trees
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#Calling the random forest function from SciKit and fitting our datasets our datasets into the model
model = RandomForestClassifier()
rForest= model.fit(xtrain, ytrain)

#Our Predictions from the model
ypred = rForest.predict(xtest)
df_cm =pd.DataFrame(confusion_matrix(ytest,ypred), ['True', 'False'], ['True', 'False'])
# plt.figure(figsize=(10,7))
plt.title("Random Forest Confusion Matrix")
plt.rcParams["figure.figsize"] = (20,10)

sns.set(font_scale=1.4) # for label size
sns.heatmap(df_cm, annot=True, annot_kws={"size": 16},fmt='g') # font size
plt.show()

#Model Performance
print('\n***','Random Forest','*** \n')
print('accuracy_score \n',accuracy_score(ytest,ypred))
print('confusion_matrix \n',confusion_matrix(ytest,ypred))
print('classification_report \n',classification_report(ytest,ypred))


# In[104]:



#Calling the decision tree function from SciKit and fitting our datasets into the model
model = DecisionTreeClassifier()
dTree= model.fit(xtrain, ytrain)

#Our Predictions from the model
ypred = dTree.predict(xtest)
df_cm =pd.DataFrame(confusion_matrix(ytest,ypred), ['True', 'False'], ['True', 'False'])
# plt.figure(figsize=(10,7))
plt.title("Decision Tree Confusion Matrix")
plt.rcParams["figure.figsize"] = (20,10)

sns.set(font_scale=1.4) # for label size
sns.heatmap(df_cm, annot=True, annot_kws={"size": 16},fmt='g') # font size

plt.show()

#Model Performance
print('\n***','Decision Tree','*** \n')
print('accuracy_score \n',accuracy_score(ytest,ypred))
print('confusion_matrix \n',confusion_matrix(ytest,ypred))
print('classification_report \n',classification_report(ytest,ypred))



# In[105]:


from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(xtrain,ytrain)


ypred=logreg.predict(xtest)
df_cm =pd.DataFrame(confusion_matrix(ytest,ypred), ['True', 'False'], ['True', 'False'])
# plt.figure(figsize=(10,7))
plt.title("Logistic Regression Confusion Matrix")
plt.rcParams["figure.figsize"] = (20,10)

sns.set(font_scale=1.4) # for label size
sns.heatmap(df_cm, annot=True, annot_kws={"size": 16},fmt='g') # font size
plt.show()

#Model Performance
print('\n***','Logistic Regression','*** \n')
print('accuracy_score \n',accuracy_score(ytest,ypred))
print('confusion_matrix \n',confusion_matrix(ytest,ypred))
print('classification_report \n',classification_report(ytest,ypred))


# In[106]:


#Packages used for Support Vectors
from sklearn import svm

#Process time is slow for Support Vector models
model = svm.SVC(kernel='linear')

#Fitting our data
sVector= model.fit(xtrain, ytrain)

#Our Predictions from the model
ypred = sVector.predict(xtest)
df_cm =pd.DataFrame(confusion_matrix(ytest,ypred), ['True', 'False'], ['True', 'False'])
# plt.figure(figsize=(10,7))
plt.title("Support Vector Machine Learning Confusion Matrix")
plt.rcParams["figure.figsize"] = (20,10)

sns.set(font_scale=1.4) # for label size
sns.heatmap(df_cm, annot=True, annot_kws={"size": 16},fmt='g') # font size
plt.show()

#Model Performance
print('\n***','Support Vector Machine Learning','*** \n')
print('accuracy_score \n',accuracy_score(ytest,ypred))
print('confusion_matrix \n',confusion_matrix(ytest,ypred))
print('classification_report \n',classification_report(ytest,ypred))


# In[107]:


#Package used for Kmeans learning
from sklearn.cluster import KMeans

df = medical.drop(['id'], axis=1)
df.head()

#Sorting into two clusters across ENTIRE dataset
kmeans = KMeans(n_clusters=2, random_state=0).fit(df)
labels = kmeans.labels_

#kMean labels are added to our dataframe
df['clusters'] = labels


#Scatter plot of only clusters Age Vs No-Show using seaborn packages
sns.lmplot('age', 'stroke', 
           data=df, 
           fit_reg=False, 
           hue="clusters",  
           scatter_kws={"marker": "D", 
                        "s": 100})
plt.title('Age vs stroke')
plt.xlabel('Age')
plt.ylabel('stroke')

