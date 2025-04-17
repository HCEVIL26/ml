
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data=pd.read_csv(r"C:\Users\CAED14\Documents\rith1.csv")
print(data)


# In[6]:


def find_s_algorithm(data):
    attributes=data.iloc[:, :-1].values
    target=data.iloc[:, -1].values
    for i in range(len(target)):
        if target[i]=="Yes":
            hypothesis=attributes[i].copy()
            break
    for i in range(len(target)):
        if target[i]=="Yes":
            for j in range(len(hypothesis)):
                if hypothesis[j]!=attributes[i][j]:
                    hypothesis[j]='?'
                    
                  
    return hypothesis
final_hypothesis=find_s_algorithm(data)
print("Most Specific Hypothesis:",final_hypothesis)


# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
values=np.random.rand(100)
labels=[]



# In[14]:


for i in values[:50]:
    if i<=0.5:
        labels.append('Class1')
    else:
        labels.append('Class2')
labels+=[None]*50



# In[15]:


print(labels)


# In[16]:


data={
    "Point":[f"x{i+1}" for i in range(100)],
    "Value":values,
    "Label":labels
}


# In[17]:


df=pd.DataFrame(data)
df.head()


# In[18]:


variable_meaning={
    "Point":"The point number",
    "Value":"The value of the point",
    "Label":"The class of the point"
    
    
}


# In[19]:


variable_df=pd.DataFrame(list(variable_meaning.items()),
                        columns=["Feature","Description"])
print("\n Variable Meaning table:")
print(variable_df)


# In[20]:


df.nunique()


# In[22]:


df.shape


# In[23]:


print("\n Basic Information about the dataset")
df.info()


# In[24]:


print("\nSummary Statistics")
df.describe().T


# In[25]:


Summary_statistics="""
- The 'Value' column has a mean of approximately 0.47, indicating that the values ar e uniformly distributed. 
-The standard deviation of the 'Value' column is approximately 0.29, showing a mode rate spread around the mean. 
-The minimum value in the 'Value' column is approximately 0.0055, and the maximum value is approximately 0.9869. 
-The first quartile (25th percentile) is approximately 0.19, the median (50th percentile) is approximately 0.47, and the third quartile (75th percentile) is approximately 0.73."""
print(Summary_statistics)


# In[26]:


print("\n Missing Values in each column")
df.isnull().sum()


# In[29]:


num_col=df.select_dtypes(include=['int','float']).columns
df[num_col].hist(figsize=(12,8),bins=30,edgecolor='black')
plt.suptitle("Feature Distributions",fontsize=16)
plt.show()


# In[34]:


get_ipython().system('pip install Series')


# In[42]:


get_ipython().system('pip install --upgrade pip')


# In[ ]:


get_ipython().system('pip install pandas')


# In[ ]:


labeled_df=df[df["Label"].notna()]
X_train=labeled_df[["Value"]]
y_train=labeled_df["Label"]
unlabeled_df=df[df["Label"].isna()]
X_test=unlabeled_df[["Value"]]
true_labels=["Class1" if x<=0.5 else "CLass2" for x in values[50:]]
k_values=[1,2,3,4,5,20,30]
results={}
accuracies={}
for k in k_values:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    predictions=knn.predict(X_test)
    results[k]=predictions
    accuracy=accuracy_score(true_labels,predictions)*100
    accuracies[k]=accuracy
    print(f"Accuracy for k={k}:{accuracy:.2f}%")
    unlabeled_df[f"Label_k{k}"]=predictions
    
    


# In[36]:


print(predictions)


# In[ ]:


df1=unlabeled_df.drop(columns=['Label',axis=1])
    df1


# In[ ]:


print("\nAccuracies for diff k values")
for k,acc,in accuracies.items():
    print(f"k={k}:{acc:.2f}%")

