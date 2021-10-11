#

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


df  = pd.read_csv('churn.csv')
df.columns = df.columns.str.lower().str.replace(' ','_')
categorical_columns = list(df.dtypes[df.dtypes=='object'].index)

for c in categorical_columns:
    df[c] =df[c].str.lower().str.replace(' ','_')
    
df.totalcharges = pd.to_numeric(df.totalcharges,errors ='coerce')
df.totalcharges =df.totalcharges.fillna(0)

df.churn = (df.churn=='yes').astype(int)


df_full_train,df_test =train_test_split(df,test_size=0.2,random_state=1)

features = ['tenure', 'monthlycharges', 'contract']
def predict(df,dv,model):
    dicts =df[features].to_dict(orient='records')
    
    X=dv.transform(dicts)
    y_pred=model.predict_proba(X)[:,1]
    
    return y_pred

def train(df_train,y_train,C=1.0):
    dicts = df_train[features].to_dict(orient = 'records')
    
    dv = DictVectorizer(sparse =False)
    x_train=dv.fit_transform(dicts)
    
    model = LogisticRegression(C=C , max_iter =1000)
    model.fit(x_train,y_train)
    
    return dv,model


# In[8]:


C =1.0
n_splits=5


# In[9]:
print(f'doing validation with C={C}')

kfold = KFold(n_splits,shuffle=True,random_state=1)
scores =[]
fold=0
for train_idx,val_idx in kfold.split(df_full_train):
    df_train=df_full_train.iloc[train_idx]
    df_val =df_full_train.iloc[val_idx]
    
    y_train = df_train.churn.values
    y_val =df_val.churn.values
    
    
    dv,model=train(df_train,y_train,C=C)
    y_pred = predict(df_val,dv,model)
    
    auc = roc_auc_score(y_val,y_pred)
    scores.append(auc)
    print(f'auc on fold{fold} is {auc}')
    fold =fold+1

print('validation results:')
print('C=%s %.3f =- %.3f' %(C,np.mean(scores),np.std(scores)))


# In[10]:


scores


# In[24]:
print('train the final model')

dv,model = train(df_full_train,df_full_train.churn.values,C=1.0)
y_pred = predict(df_test,dv,model)
y_test =df_test.churn.values
auc= roc_auc_score(y_test,y_pred)
auc

print(f'auc={auc}')

# In[11]:


import pickle


# In[13]:


output_file = f'model_C={C}.bin'
output_file


# In[14]:


f_out = open(output_file,'wb')
pickle.dump((dv,model),f_out)
f_out.close()


# In[15]:


with open(output_file,'wb') as f_out:
    pickle.dump((dv,model),f_out)

print(f'the model is saved to {output_file}')
# In[4]:


import pickle


# In[5]:


model_file= 'model_C=1.0.bin'


# In[6]:


with open(model_file,'rb') as f_in:
    dv,model=pickle.load(f_in)


# In[7]:


dv,model


# In[8]:


customer={
    
    "contract": "two_year",
     "tenure": 12,
     "monthlycharges": 19.7
}


# In[9]:


X=dv.transform([customer])


# In[10]:


model.predict_proba(X)[0,1]


# In[ ]:




