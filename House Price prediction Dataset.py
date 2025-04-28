#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import LabelEncoder


# In[2]:


df=pd.read_csv('Housing Dataset.csv')


# In[3]:


df


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.info()


# In[7]:


df['price'].max()


# In[8]:


df['price'].min()


# In[9]:


le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])


# In[10]:


df['log_price'] = np.log1p(df['price'])
df['log_airconditioning']= np.log1p(df['airconditioning'])
df['log_basement']= np.log1p(df['basement'])
df['log_parking']= np.log1p(df['parking'])
df['log_prefarea']= np.log1p(df['prefarea'])
df['log_stories']= np.log1p(df['stories'])
df['log_mainroad']= np.log1p(df['mainroad'])
df['log_guestroom']= np.log1p(df['guestroom'])
df['log_bathrooms']= np.log1p(df['bathrooms'])
df['log_bedrooms']= np.log1p(df['bedrooms'])
df['log_hotwaterheating']= np.log1p(df['hotwaterheating'])
df['log_furnishingstatus']= np.log1p(df['furnishingstatus'])
df['log_area']= np.log1p(df['area'])


# In[11]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df['price'], kde=True)
plt.title("Original Price Distribution")
plt.show()

sns.histplot(df['log_price'], kde=True)
plt.title("Log-Transformed Price Distribution")
plt.show()


# In[12]:


X = df[['log_airconditioning', 'log_stories','log_bathrooms', 'log_furnishingstatus', 'log_area']]
y = df['log_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)


# In[13]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[14]:


y_pred=model.predict(X_test)


# In[15]:


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R² Score: {r2}")


# In[16]:


# Scatter plot: Actual vs Predicted
plt.figure(figsize=(10, 6))
sns.regplot(x=y_test, y=y_pred, color='teal', scatter_kws={'s': 10})  # Increased scatter point size

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values with Regression Line')
# Annotate coefficients
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
for i, coef in enumerate(model.coef_):
  plt.annotate(f'{X.columns[i]}: {coef:.2f}', (y_test.min() + (i * 0.1), y_pred.min() + (i * 0.1)), color='black')
plt.show()

# Interpretation of coefficients
print("Coefficients:")
print(coefficients)
print("\nInterpretation of Coefficients:")
for index, row in coefficients.iterrows():
    print(f"A one unit increase in '{row['Feature']}' is associated with a {row['Coefficient']:.2f} unit change in 'log_price'.")


# In[ ]:




