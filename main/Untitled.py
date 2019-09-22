
import pandas as pd


# In[9]:

# from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV


# In[89]:

from sklearn.ensemble import RandomForestRegressor


# In[28]:

import matplotlib.pyplot as plt


# In[20]:

import numpy as np


# In[2]:

df = pd.read_excel('./data/train.xlsx')


# In[103]:

df.columns


# In[ ]:




# In[116]:

params_for_maximize = ['Обводненность (вес), %', 'Время работы, ч', 'Забойное давление', 'Давление на приеме', 
                       'Пластовое давление', 'Нефть, т']

X_train_max = df[params_for_maximize].fillna(df[params_for_maximize].median())


# In[117]:

X_train_max.dtypes


# In[118]:

def clean(x):
    x = str(x)
    if x.find('0:00:00') >= 0:
        return np.nan
    else:
        return float(x)


# In[119]:

X_train_max['Забойное давление'] = X_train_max['Забойное давление'].apply(clean)
X_train_max['Давление на приеме'] = X_train_max['Давление на приеме'].apply(clean)


# In[120]:

X_train_max = X_train_max.fillna(X_train_max.median())


# In[122]:

# X_train_max['Давление на приеме'] = X_train_max['Давление на приеме'].astype(float)
# X_train_max['Забойное давление'] = X_train_max['Забойное давление'].astype(float)

y_train_max = X_train_max.pop('Нефть, т')


# In[81]:

X_train_max_sp, X_test_max_sp, y_train_max_sp, y_test_max_sp = train_test_split(X_train_max, y_train_max, test_size=0.3)


# In[124]:

regr = RandomForestRegressor(n_estimators=100)
# regr.fit(X_train_max_sp, y_train_max_sp)


# In[125]:

regr.fit(X_train_max, y_train_max)


# In[126]:

regr.score(X_train_max, y_train_max)


# In[101]:

# regr.score(X_test_max_sp, y_test_max_sp)


# In[102]:

test = pd.read_excel('./data/test.xlsx')


# In[171]:

params_for_maximize = ['Обводненность (вес), %', 'Время работы, ч', 'Забойное давление', 'Давление на приеме', 
                       'Пластовое давление', 'Закачка, м3']

X_test = test[params_for_maximize].fillna(test[params_for_maximize].median())


# In[172]:

X_test.head()


# In[173]:

doptest.dtypes


# In[174]:

X_test['Забойное давление'] = X_test['Забойное давление'].apply(clean)


# In[175]:

X_test = X_test.fillna(X_test.median())


# In[176]:

doptest = X_test.copy()
X_test = X_test.drop(['Закачка, м3'], axis = 1)


# In[177]:

doptest['Нефть, т']= regr.predict(X_test)


# In[178]:

doptest = doptest.fillna(doptest.median())


# In[179]:

doptest = doptest.sort_values('Нефть, т', ascending= False)


# In[ ]:




# In[169]:

doptest.to_excel('dd.xlsx')


# In[99]:

plt.figure(figsize=(20,15))
plt.scatter(x= df3['Координата X'], y= df3['Координата Y'], marker='o', c='r', edgecolor='b')

for i in df3.index:
    plt.annotate(df3['Характер работы'][i], xy=(df3['Координата X'][i], df3['Координата Y'][i]))

plt.show()






