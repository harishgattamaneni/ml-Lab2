# In[4]:


import pandas as pd
import numpy as np

file_name = r"19CSE305_LabData_Set3.1 (1).xlsx"
worksheet_name = 'thyroid0387_UCI'
df = pd.read_excel(file_name, sheet_name=worksheet_name)
df.replace("?", np.nan, inplace=True)
nominal=['sex', 'on thyroxine', 'query on thyroxine',
       'on antithyroid medication', 'sick', 'pregnant', 'thyroid surgery',
       'I131 treatment', 'query hypothyroid', 'query hyperthyroid', 'lithium',
       'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH measured', 
       'T3 measured', 'TT4 measured', 'T4U measured',
       'FTI measured', 'TBG measured', 'referral source',
       'Condition']
interval=['TSH','T3','TT4',  'T4U','FTI',  'TBG']
ratio=['age']
nominal_encoded = pd.get_dummies(df,columns=nominal)
df=nominal_encoded
df


df.info()


import numpy as np
df.describe()

df.isnull().sum()

Q3 = df['TSH'].quantile(0.75)
Q1 = df['TSH'].quantile(0.25)
IQR = Q3 - Q1

upperB = Q3 + 1.5 * IQR
lowerB = Q1 - 1.5 * IQR

outliers_TT4 =df[(df['TSH'] < lowerB) | (df['TSH'] > upperB)]
outliers_TT4


# In[15]:


Q3 = df['T3'].quantile(0.75)
Q1 = df['T3'].quantile(0.25)
IQR = Q3 - Q1
upperB = Q3 + 1.5 * IQR
lowerB = Q1 - 1.5 * IQR

outliers_TT4 =df[(df['T3'] < lowerB) | (df['T3'] > upperB)]
outliers_TT4


# In[14]:


Q3 = df['T4U'].quantile(0.75)
Q1 = df['T4U'].quantile(0.25)
IQR = Q3 - Q1
upperB = Q3 + 1.5 * IQR
lowerB = Q1 - 1.5 * IQR

outliers_TT4 =df[(df['T4U'] < lowerB) | (df['T4U'] > upperB)]
outliers_TT4


# In[13]:


Q3 = df['FTI'].quantile(0.75)
Q1 = df['FTI'].quantile(0.25)
IQR = Q3 - Q1
upperB = Q3 + 1.5 * IQR
lowerB = Q1 - 1.5 * IQR

outliers_TT4 =df[(df['FTI'] < lowerB) | (df['FTI'] > upperB)]
outliers_TT4


# In[16]:


Q3 = df['TT4'].quantile(0.75)
Q1 = df['TT4'].quantile(0.25)
IQR = Q3 - Q1
upperB = Q3 + 1.5 * IQR
lowerB = Q1 - 1.5 * IQR

outliers_TT4 =df[(df['TT4'] < lowerB) | (df['TT4'] > upperB)]
outliers_TT4


# In[17]:


Q3 = df['TBG'].quantile(0.75)
Q1 = df['TBG'].quantile(0.25)
IQR = Q3 - Q1
upperB = Q3 + 1.5 * IQR
lowerB = Q1 - 1.5 * IQR

outliers_TT4 =df[(df['TBG'] < lowerB) | (df['TBG'] > upperB)]
outliers_TT4


# In[18]:


import pandas as pd
mean = df['age'].mean()
variance = df['age'].var()
print("Mean of  feature age:", mean)
print("Variance of  feature age:", variance)


# In[19]:


mean = df['TBG'].mean()
variance = df['TBG'].var()
print("Mean of feature TBG:", mean)
print("Variance of  feature TBG:", variance)


# In[20]:


mean = df['T4U'].mean()
variance = df['T4U'].var()
print("Mean of feature T4U:", mean)
print("Variance of feature T4U:", variance)


# In[21]:


mean = df['FTI'].mean()
variance = df['FTI'].var()
print("Mean of feature FTI:", mean)
print("Variance of feature FTI:", variance)


# In[22]:


mean = df['T3'].mean()
variance = df['T3'].var()
print("Mean of feature T3:", mean)
print("Variance of feature T3:", variance)


# In[23]:


mean = df['TT4'].mean()
variance = df['TT4'].var()
print("Mean of feature TT4:", mean)
print("Variance of  feature TT4:", variance)


# In[24]:


mean = df['TSH'].mean()
variance = df['TSH'].var()
print("Mean of feature TSH:", mean)
print("Variance of  feature TSH:", variance)


# In[25]:


df['T3'].fillna(df['T3'].median(), inplace=True)
df['T4U'].fillna(df['T4U'].median(), inplace=True)
df['TSH'].fillna(df['TSH'].median(), inplace=True)
df['TBG'].fillna(df['TBG'].median(), inplace=True)
df['FTI'].fillna(df['FTI'].median(), inplace=True)
df['TT4'].fillna(df['TT4'].median(), inplace=True)
df


# In[26]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scale = ['age','T3','TSH','T4U','TT4','TBG','FTI']
df[scale] = scaler.fit_transform(df[scale])
df


# In[27]:


v1 = df['sex_M']
v2 = df['Condition_M']
f01 = sum([1 for a, b in zip(v1, v2) if a == 0 and b == 1])
f11 = sum([1 for a, b in zip(v1, v2) if a == b == 1])
f00 = sum([1 for a, b in zip(v1, v2) if a == b == 0])
f10 = sum([1 for a, b in zip(v1, v2) if a == 1 and b == 0])
print(f00,f01,f10,f11)


# In[31]:


jc = f11 / (f01 + f10 + f11)
jc


# In[29]:


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
v2 = np.array(v2).reshape(1, -1)
v1 = np.array(v1).reshape(1, -1)
cosine_sim = cosine_similarity(v1, v2)

print("Cosine Similarity:", cosine_sim[0][0])


# In[32]:


import numpy as np
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
vectors = [
    [1, 0, 1, 1, 0],
    [0, 1, 1, 0, 0],
    [1, 1, 0, 0, 1],
    
]
n = len(vectors)
jc_matrix = np.zeros((n, n))
smc_matrix = np.zeros((n, n))
cos_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        jc_matrix[i][j] = jaccard_score(vectors[i], vectors[j])
        smc_matrix[i][j] = np.sum(np.logical_and(vectors[i], vectors[j])) / np.sum(np.logical_or(vectors[i], vectors[j]))
        cos_matrix[i][j] = cosine_similarity([vectors[i]], [vectors[j]])[0][0]


plt.figure(figsize=(12, 4))

plt.subplot(131)
sns.heatmap(jc_matrix, annot=True, cmap="YlGnBu", xticklabels=False, yticklabels=False)
plt.title('Jaccard Coefficient')

plt.subplot(132)
sns.heatmap(smc_matrix, annot=True, cmap="YlGnBu", xticklabels=False, yticklabels=False)
plt.title('Simple Matching Coefficient')

plt.subplot(133)
sns.heatmap(cos_matrix, annot=True, cmap="YlGnBu", xticklabels=False, yticklabels=False)
plt.title('Cosine Similarity')

plt.tight_layout()
plt.show()


# In[ ]: