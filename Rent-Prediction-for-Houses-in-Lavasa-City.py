#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Lab 
# 
# Name  : Prashant Gupta
# 
# Reg no: 21122044
# 
# Class : 2MScDS

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
import sklearn as skl
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import random


# # Data

# In[2]:


data= pd.read_csv("D:\Downloads\opera\HousePrices - Lab3.csv")
data.head()


# # EDA

# In[3]:


sns.distplot(data['RentPerMonth'])


# In[4]:


sns.countplot(y ='BuildingType', data = data)


# In[5]:


sns.countplot(y ='Location', data = data)


# In[6]:



fig = px.scatter(data, x="RentPerMonth", y="AreaSqFt", color="BuildingType",
                 size='NoOfBath', hover_data=['NoOfBalcony'])
fig.show()


# In[7]:



fig = px.pie(data, values='RentPerMonth', names='Size',
             title='Distribution According To number of Rooms', hole=0.4,
             hover_data=['NoOfBalcony'],)
fig.show()


# # Premodelling Feature Engineering

# In[8]:


data["Location"] = data["Location"].replace(to_replace =["Portofino A","Portofino B","Portofino C","Portofino D","Portofino E","Portofino F","Portofino G","Portofino H"], value ="Portofino")
data.head()


# In[9]:


label_encoder = preprocessing.LabelEncoder()
data['BuildingType']= label_encoder.fit_transform(data['BuildingType'])
data['Location']= label_encoder.fit_transform(data['Location'])
data['Size']= label_encoder.fit_transform(data['Size'])
data.head()



# In[10]:


X = data.drop("RentPerMonth",axis=1)

y = data["RentPerMonth"]


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)

LR = LinearRegression()
LR.fit(X_train,y_train)

y_prediction =  LR.predict(X_test)
# y_prediction
score=r2_score(y_test,y_prediction)
print("r2 socre is ",score)
print("mean_sqrd_error is==",mean_squared_error(y_test,y_prediction))
print("root_mean_squared error of is==",np.sqrt(mean_squared_error(y_test,y_prediction)))
score_50 =score


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=42)

LR = LinearRegression()
LR.fit(X_train,y_train)

y_prediction =  LR.predict(X_test)

score=r2_score(y_test,y_prediction)
print("r2 socre is ",score)
print("mean_sqrd_error is==",mean_squared_error(y_test,y_prediction))
print("root_mean_squared error of is==",np.sqrt(mean_squared_error(y_test,y_prediction)))
score_40 =score


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

LR = LinearRegression()
LR.fit(X_train,y_train)

y_prediction =  LR.predict(X_test)

score=r2_score(y_test,y_prediction)
print("r2 socre is ",score)
print("mean_sqrd_error is==",mean_squared_error(y_test,y_prediction))
print("root_mean_squared error of is==",np.sqrt(mean_squared_error(y_test,y_prediction)))
score_30 =score


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

LR = LinearRegression()
LR.fit(X_train,y_train)

y_prediction =  LR.predict(X_test)

score=r2_score(y_test,y_prediction)
print("r2 socre is ",score)
print("mean_sqrd_error is==",mean_squared_error(y_test,y_prediction))
print("root_mean_squared error of is==",np.sqrt(mean_squared_error(y_test,y_prediction)))
score_20 =score


# ### Compairing all Test Ratios

# In[15]:


score = pd.DataFrame()
score['Ratios'] = ['50:50',"60:40","70:30","80:20"]
score["Accuracy"] = [score_50*100,score_40*100,score_30*100,score_20*100]
score.head()


# ### Conclusion 

# #### we can see that 70:30 is the best ratio , so we will use that

# # Use Cases
# ### (Menu Driven)

# In[16]:


data.head()
data2=pd.read_csv("D:\Downloads\opera\HousePrices - Lab3.csv")
data2.head()


# In[ ]:


data3 = pd.DataFrame()
print("Please select the features from the list with their index numbers (Leave Empty For Random) ")
print("\n")

dict1 = {1: 'Minimum Budget Rooms',2: 'Semi Furnished Single Room',3:'Semi Furnished Flat',
                 4:'Fully Furnished Single Room',5:'Super Furnished Single Room',6:'Semi Furnished Villa',
                 7:'Fully Furnished Flat',8: 'Super Furnished Flat',9:'Fully Furnished Villa', 10:'Super Furnished Villa'}
print(dict1)
ip = int(input("Please select any one type of buidling type: "))

data = pd.read_csv("D:\Downloads\opera\HousePrices - Lab3.csv")
le = preprocessing.LabelEncoder()
le.fit(data['BuildingType'])
data3['BuildingType'] = [le.transform([dict1[ip]])[0]]



    
print("\n")
dict2={1:'Portofino', 2:'School Street', 3:'Clubview Road', 4:'Starter Homes'}
print(dict2)
ip = int(input("Please select any one type of location: "))
data = pd.read_csv("D:\Downloads\opera\HousePrices - Lab3.csv")
data["Location"] = data["Location"].replace(to_replace =["Portofino A","Portofino B","Portofino C","Portofino D","Portofino E","Portofino F","Portofino G","Portofino H"], value ="Portofino")
le = preprocessing.LabelEncoder()
le.fit(data['Location'])
data3['Location'] = [le.transform([dict2[ip]])[0]]


    
print("\n")    
dict3={1: '1 BHK', 2: '2 BHK', 3: '1 RK', 4: '3 BHK', 5: '4 BHK',6: '5 BHK', 7:'6 BHK',8:'8 BHK', 9:'7 BHK', 
               10:'9 BHK'}
print(dict3)
ip = int(input("Please select any one type of size: "))
data = pd.read_csv("D:\Downloads\opera\HousePrices - Lab3.csv")
le = preprocessing.LabelEncoder()
le.fit(data['Size'])
data3['Size'] = [le.transform([dict3[ip]])[0]]

    

    
print("\n")
ip = int(input("Please select square feet area available (min = 375, max=35000): "))
data3["AreaSqFt"] = ip

        

        
print("\n")    
ip = int(input("Please select number of baths you would like to have (min = 1, max = 11): "))
data3["NoOfBath"] = ip

        
        
        
print("\n")    
ip = int(input("How many people wish to stay in the property (min = 1, max = 6): "))
data3["NoOfPeople"] = ip

        
        
        
print("\n")    
ip = int(input("Please select number of balcony's you would like to have (min = 0, max = 3): "))
data3["NoOfBalcony"] = ip

        
        
print("\n")
print("\n")
print("_____________________________________________________________________________________________________________________________")
print("")
print("You selected following features with values: \n",data3.head())
print("_____________________________________________________________________________________________________________________________")
print("")
print("\n")

y_prediction =  LR.predict(data3)
print("Predicted Rent is:" ,y_prediction)

