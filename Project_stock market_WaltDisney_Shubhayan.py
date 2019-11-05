#!/usr/bin/env python
# coding: utf-8

# # Project: Shubhayan Bhattacharya, IIT Madras (GroupB, ML)

# ## Stock Market Data Analysis of Walt Disney

# ### Importing libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
my_year_month_fmt = mdates.DateFormatter('%m/%y')
from sklearn.linear_model import LinearRegression


# ### Reading the csv file for Walt Disney stock market data

# In[3]:


stock = pd.read_csv(r"C:\Users\Bisht\Downloads\My assignments\EOD-DIS.csv")


# ### Data shape and attributes

# In[4]:


print(stock.shape)
stock


# ### Description of the data

# In[5]:


stock.describe()


# ### Checking for any missing values in the data set

# In[6]:


stock.isnull().sum()


# In[7]:


sns.pairplot(stock[['Open',"High","Low","Close", "Volume"]],height=2)


# In[8]:


sns.pairplot(stock[['Adj_Open', "Adj_High", "Adj_Low", "Adj_Close", "Adj_Volume"]],height=2)


# In[9]:


import seaborn as sns
sns.heatmap(stock.corr(),cmap='jet')


# ### Adjusted closing price as a function of days

# In[10]:


# Define the figure size for the plot
plt.figure(figsize=(10, 7))
# Plot the adjusted close price
stock['Adj_Close'].plot()
# Define the label for the title of the figure
plt.title("Adjusted Close Price of WD", fontsize=16)
# Define the labels for x-axis and y-axis
plt.ylabel('Price', fontsize=14)
plt.xlabel('Days', fontsize=14)
# Plot the grid lines
plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
plt.show()


# In[11]:


stock_reduce=stock.drop("Date",axis=1)


# In[12]:


# simple moving average calculation
short_rolling = stock_reduce.rolling(window=10).mean()
short_rolling


# In[13]:


#long window simple moving average
long_rolling = stock.rolling(window=100).mean()
long_rolling.tail()


# In[14]:


df=stock


# In[15]:


# A variable for predicting 'n' days out into the future
forecast_out = 30 #'n=30' days
#Create another column (the target ) shifted 'n' units up
df['Prediction'] = stock[['Adj_Close']]
#print the new data set
print(df.tail())


# ## Independent variables and dependent variables

# In[16]:


### Create the independent data set (X)  #######
# Convert the dataframe to a numpy array
X =(df.drop(['Prediction','Date','Open','High','Low','Close','Dividend','Split','Volume','Adj_Close','Adj_Volume'],axis=1))

X2=X.values
y = np.array(df['Prediction'])


# ## Independent columns

# In[17]:


X.columns


# ## Spiliting data in training and testing set

# In[18]:


from sklearn.model_selection import train_test_split
# Split the data into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(X2, y, test_size=0.2)


# ## Logistic regression model

# In[19]:


#Create the Linear Regression Model
from sklearn.linear_model import LinearRegression
lr = LinearRegression() # Create the model
lr.fit(x_train, y_train) #Train the model
print(lr.coef_)
predictions = lr.predict(x_test) #Make predictions on the test data


# ### Accuracy of the model

# In[20]:


# Testing Model: Score returns the coefficient of determination R^2 of the prediction. 
# The best possible score is 1.0
lr_confidence = lr.score(x_test, y_test)
print("lr confidence (R^2): ", lr_confidence)
from sklearn.metrics import mean_squared_error
print("Mean Squared Error (MSE): ",mean_squared_error(y_test, predictions))

