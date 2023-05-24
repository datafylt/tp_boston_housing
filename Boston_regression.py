#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# In[5]:


# Charger le data
#data = pd.read_csv('boston.csv')
#data = pd.read_csv('remote://mondataremote/boston.csv.dvc')

data = pd.read_csv('boston.csv.dvc')


# In[7]:


# Séparation des features et de la variable cible
features = data.columns[:-1]
X = data[features]
y = data['MEDV']


# In[8]:


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# In[9]:


# Définir le modèle
lr_model = LinearRegression()



# In[10]:


# Entraînement du modèle
model = lr_model.fit(X_train, y_train)



# In[11]:


# Prédiction sur les données de test
predictions = model.predict(X_test)



# In[12]:


# Évaluation
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)
r2 = r2_score(y_test, predictions)

print("MAE: {}".format(mae))
print("RMSE: {}".format(rmse))
print("R-squared: {}".format(r2))



# In[13]:


# Affichage des prédictions pour certaines colonnes
predictions_df = pd.DataFrame(predictions, columns=data.columns[13::2])
print(predictions_df)


# In[ ]:
# Write scores to a file
with open("metrics.txt", 'w') as outfile:
        outfile.write("MAE:  {0:2.1f} \n".format(mae))
        outfile.write("RMSE:  {0:2.1f} \n".format(rmse))
        outfile.write("R-squared: {0:2.1f}\n".format(r2))



