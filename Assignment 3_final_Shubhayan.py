#!/usr/bin/env python
# coding: utf-8

# # Assignment 3: Shubhayan Bhattacharya, IIT Madras (GroupB, ML)

# ### Question 1:	List down the step of Model development

# #### Ans. The step of model development are as follows:
# - Data type (finding the nature of data)
# - Dependent variable type (classification/regression)
# - Independent variable (correlation between the independent variables)
# - Model description
# - Train and test the data
# - Fit the data
# - Prediction of data
# - Confusion matrix
# - ROC curve (for checking the model accuracy)

# ### Question 2: Define R square

# #### Ans. R square also called the coefficient of determination is defined as the proportion of variance in the dependent variable that is predicatble from independent variable(s) in a regression model. If the R square of a model is 0.65, 65% of the observed variation can be explained by the input values. 

# ### Question 3: Define adjusted R square

# #### Ans. R square as described above assumes that every single variable is dependent on the variation of the dependent variable. The adjusted R square gives the knowledge of the percentage of variation explained by only the indendent variables that actually affect the dependent variable. 

# ### Question 4: Define mean square error

# #### Ans. Mean squared error of an estimator is defined as the average squared difference between the estimated values and actual values measuring the average square o errors. 

# ### Question 5: Define root mean square error (RMSE)

# #### Ans. The RMSE of an estimator for an estimated parameter is the square root of the mean square error. 

# ### Question 6: Define mean absolute percentage error (MAPE)

# #### Ans. In statistics MAPE is used as a measure of the prediction of accuracy of a forecasting method. 
# \begin{equation}
# MAPE = (100/n)\sum_{t=1}^{n} [(A_t-F_t)/A_t]
# \end{equation}
# 
# #### where A and F represent the actual and forecast value. 

# ### Question 7: What are the assumptions of linear regression?
# 

# #### Ans. The assumptions of linear regression are as follows:
# - First and foremost, the relationship between the independent and dependent variables must be linear
# - The linear regression requires all variables to be multivariate normal
# - It assumes that no or very little multicollinearity is there in the data. If the independent variables are highly correlated with each other, multicollinearity occurs. 
# - No autocorrelation should be present i.e the degree of correlation between the values of the same variable across different observations in the data
# - Homoscedasticity should be present i.e the noise/random disturbance between the dependent and independent variables must be same across all observations. This can be checked by scatter plots. 

# ### Question 8: Define multicollinearity 

# #### Ans.  High inter association or inter correlation between the independent variables in the data is called multicollinearity. The inferences/conclusions made from a data having high multicollinearity may not be reliable. 

# ### Question 9: Make a wine prediction model through linear regression 

# #### Ans. Please see the end of this document. 

# ### Question 10: How decision tree works

# #### Ans. The main aim of decision tree is to create a training model which can predict value of target variables by learning decision rules as trained by previous data. It works in the following way: 
# - Decision tree algorithm aims to solve a problem by using tree representation
# - Each internal node of a decision tree corresponds to an attribute and each leaf node corresponds to a class label
# - The best attribute of a data is placed at the root of a tree
# - The training set is divided into various subsets (Splitting). The subsets are created in such a way that each subset contains data with the same value for an attribute
# - The last two steps are repeated until one finds leaf nodes in all branches of the tree

# ### Question 11: Define splitting and stopping criteria of a decision tree

# #### Ans. Splitting is defined as the partitioning of data into various subsets. Splits are formed on a particular variable.
# #### The stopping criteria of a decision tree are as follows:
# - Number of cases in the node is less than some pre-specified limit.
# - Purity of the node is more than some pre-specified limit. 
# - Depth of the node is more than some pre-specified limit.
# - Predictor values for all records are identical - in which no rule could be generated to split them.

# ### Question 12: What is entropy?

# #### Ans. The homogenity of a sample is found from the value of entropy calculated. If the value of entropy is 0, the sample is completely homogeneous, while if the sample is equally divided it has an entropy of 1. 

# ### Question 13: What is Gini?

# #### Ans. Gini index implies that, if two items are selected from a population at random, then these items must be from the same class and the probability for this is 1, if the population is pure. Gini index only performs binary splits, and works with categorical target variable such as "yes" or "no".  Higher the value of Gini index higher is the homogenity of the data class. 

# ### Question 14: What is information gain? 

# #### Ans. Information gain (IG) measures how much “information” a feature gives us about the class. The features that perfectly partition should give maximal information and the unrelated features should give no information.

# ### Question 15: How to identify overfitting? 

# #### Ans. Overfitting can be detected by finding whether the model fits new data as well as the data used to estimate the model. In the case of linear regression, overfitting can be easily predicted by the cross validation method called the predicted R squared (measured by the statistical software). 

# ## Question 9: Wine price prediction model 

# In[ ]:





# ### Importing libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ### Loading dataset

# In[85]:


wine = pd.read_csv(r"C:\Users\Bisht\Downloads\Python\Programs and Data\EDA-master\wine.csv")


# ### Data shape and attributes

# In[86]:


print(wine.shape)
wine.head()


# ### Description of the data

# In[87]:


wine.describe()


# ### Checking for any missing values in the data set

# In[88]:


wine.isnull().sum()


# ### Independent variables

# In[166]:


X = wine[[ 'WinterRain', 'AGST', 'HarvestRain', 'Age',
       'FrancePop']]


# ### Target variable is price

# In[167]:


y = wine[['Price']]


# In[181]:


sns.pairplot(wine[['WinterRain',"AGST","HarvestRain","Age", "FrancePop","Price"]],height=2)


# ### Checking correlation among the input variables

# In[182]:


import seaborn as sns
sns.heatmap(wine.corr(),cmap='jet')


# ### Importing train and test split library

# In[168]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state = 100 )


# In[169]:


X_test


# In[170]:


X_train


# In[171]:


y_train


# In[172]:


y_test


# In[173]:


import statsmodels.api as sm
lm1 = sm.OLS(y_train,X_train).fit()
# print the coefficients
lm1.params


# In[174]:


lm1.summary()


# ### Adj. R- squared value: 
# #### It's most useful as a tool for comparing different models.
# 
# #### Is has value 99.9%. It suggests that 99.9% wine price can be estimated by these predictor variables.

# ### Regression coefficient: 
# #### It represents the change in the target variable due to change of one predictor variable by one unit when other predictor variables are constant. 
# #### We have selected all predictor variables in this case
# #### The coefficient values for these variables are 0.001265, 0.538851, -0.004890, 0.005132 and -0.000041.
# 

# ### Standard error:
# 
# #### It measures the accuracy of coefficient by estimating the variation of the coefficient if the same test were run on a different sample of our population. The standard errors have been found to be 0, 0.121, 0.001, 0.021 and 0.0000352 respectively for 'WinterRain', 'AGST', 'HarvestRain', 'Age' and 'FrancePop' respectively. 

# ### Hypothesis testing and p-value
# 
# #### (1) We assume the null hypothesis that there is no linear relationship between predictor and target variables. So, linear coefficients would be zero.
# #### (2) But we get non-zero values of the coefficients of the predictor variables which rejects the null hypothesis. 
# 
# #### (3) The p-value corresponds to the probability that the coefficient is actually zero. Here, p-value is very small (<0.05) and it is zero for 'WinterRain', 'AGST' and 'HarvestRain' while the p-value is higher (>0.05) for the predictor variables 'Age' and 'FrancePop'. We can reject the null hypothesis. We can say that there is linear dependence of the house price on the predictor variables 'WinterRain', 'AGST' and 'HarvestRain'.

# ### Confidence interval:
# 
# #### Confidence interval is a range within which our coefficients are likely to fall. Here, coefficients are showing the probable range of the estimated coefficients.
# #### For example, the cofidence interval for 'WinterRain' is 0, 0.002 and for  'AGST' the interval is 0.281, 0.796.

# ### Training and testing the data

# In[175]:


from sklearn.linear_model import LinearRegression
X_train,X_test,y_train,y_test=train_test_split(X, y,train_size=0.8, random_state=100)
lrm= LinearRegression()
lrm.fit(X_train,y_train)


# ### Model coefficients

# In[176]:


print(lrm.intercept_)
print(lrm.coef_)


# In[177]:


y_pred = lrm.predict(X_test)
print(y_pred)


# In[178]:


lrm.fit(X_train,y_train)


# ### MSE, MAE, RMSE computation

# In[179]:


from sklearn import metrics
print("Mean square error =",(metrics.mean_squared_error(y_test, y_pred)))
print("Root mean square error =",np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print("Mean absolute error =",metrics.mean_absolute_error(y_test,y_pred))


# ### Conclusions:
# #### (1)  Price of the wine depends on all the variables but is largely dependent on 'WinterRain', 'AGST' and 'HarvestRain'. 
# #### (2)  p-value of the individual predictiors 'WinterRain', 'AGST' and 'HarvestRain' reject the null hypothesis of no linear relationship between predictors and the target value.

# # Please check the PDF file also for the model parameters to arrive at the conclusion. 

# In[ ]:




