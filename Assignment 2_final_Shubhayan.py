#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Shubhayan Bhattacharya, IIT Madras (GroupB, ML)

# #### Importing the libraries and data set

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
tips = sns.load_dataset("tips")


# #### Finding the number of rows and columns in the data frame

# In[2]:


tips.shape


# #### Observing the first five data

# In[3]:


tips.head ()


# #### Basic description of data

# In[4]:


tips.describe()


# #### Independent variables in the data frame

# In[5]:


tips.columns


# #### Independent variables 
# 
# 1. total_bill (numeric)
# 
# 2. tip (numeric)
# 
# 3. sex (categorical: "Male", "Female")
# 
# 4. smoker (categorical: "Yes", "No")
# 
# 5. day (categorical: "Thur", "Fri", "Sat", "Sun")
# 
# 6. time (categorical: "Lunch", "Dinner")
# 
# 7. size (numeric)

# #### For checking if there are missing values in the data set

# In[6]:


tips.isnull().sum()


# #### Number of male and female in the data set

# In[7]:


tips.sex.tolist().count('Male')


# In[8]:


tips.sex.tolist().count ('Female')


# ### Data visualization

# #### Distribution plot for total bill

# In[9]:



sns.distplot(tips.total_bill)
plt.title ('Distribution plot for total bill')


# #### Observations/comments from the above plot
# 
# - The total bill spans from less than 5 to nearly 60
# - The average total bill has a maxima less than 20 which can also be seen in the basic desciption of data but this gives us a visual insight to that
# 

# #### Distribution plot for tip

# In[10]:


sns.distplot(tips['tip'], kde=True, bins=15)
plt.title('Distribution plot for tip')


# #### Pair plot of tip, total bill and size

# In[11]:


sns.pairplot(tips[['tip','total_bill','size']])


# #### Observations/comments from the pair plot above
# - There is a clear linear relationship between the total bill and tip. We clearly see that the total_bill is above 20 the tip in most cases is above 2. 
# - Other pair plots do not show any such relationship between one another

# #### Checking the effect of a third variable
# 

# In[12]:


sns.relplot(x="total_bill", y="tip", hue="day", data=tips)
sns.relplot(x="total_bill", y="tip", hue="smoker", data=tips)
sns.relplot(x="total_bill", y="tip", hue="sex", data=tips)


# #### Checking independence between the independent variables

# In[13]:


sns.heatmap(tips.corr(),cmap='jet')


# #### Average total bill, tip and size according to day

# In[14]:


tips.groupby('day').mean()


# #### Observations from the above result
# 
# - The percentage of tip with respect to the total bill for Thur, Fri, Sat and Sun are 15.67%, 15.94%, 14.64% and 15.20 % respectively and hence the amount of tip does not vary much in the weekdays or weekend. 

# #### Average total bill, tip and size according to sex

# In[15]:


tips.groupby('sex').mean()


# #### Observations from the above result
# 
# - The percentage of tip with respect to the total bill for male and female are 14.89% and 15.69 % respectively
# - The average total_bill amount of females are less than that of males

# In[16]:


day_name=list(set(tips.day.tolist()))


# In[17]:


day_name[0]


# In[18]:


np.sum(np.array(tips.total_bill)>20)


# In[19]:


tips.groupby(['sex','day','smoker', 'time']).size()


# #### Scatter plot for tip for both genders

# In[20]:


sns.catplot(x="sex", y="tip", data=tips)


# #### Observations from the above result
# - We can see from the above plot that most of the tips lie between 2 and 4 irrespective of the gender and there are outlier values for both

# #### Other scatterplots

# In[21]:


sns.catplot(x="day", y="total_bill", hue="time", kind="swarm", data=tips)


# ### Observations from the above plot: 
# #### (1) It can be clearly seen from the above plot that there are no total_bill data (hence no customers) which implies that the restaurant/hotel is only open during dinner in the weekends. 
# #### (2) It can be observed that on Thursday ('Thur') customers only come during lunch time, hence the restaurant may think of shutting down during dinner time. Since there are no customers during dinner the restaurant only incur loss if they keep open at night on Thursday. 
# #### (3) It may be suggested that the restaurant can only operate on weekends during dinner time (if they are open during both lunch and dinner). 

# In[22]:


sns.catplot(x="day", y="total_bill", hue="sex", kind="swarm", data=tips)


# #### Observations from the above plot
# -The above plot gives us the total_bill according to the day for both male and female customers. 

# #### Box plot of total_bill as a function of day 

# In[23]:


sns.catplot(x="day", y="total_bill", hue="sex", kind="box", data=tips)


# #### Observations from the above plot
# - The box plot here actually gives all the results that we have already found out from "describe" in a graphical way. 
# - The bottom line and the top line of the plot denotes minima and maxima of the total_bill on a particular day. 
# - The bottom and top line of the box denotes the first and third quartile 
# - While the horizontal line within the box denotes the median of the data
# - Any value which lies outside the top line of the plot are outlier values
# - We can infer that the total_bill is higher on weekends (Sat, Sun) than on weekdays (Thur, Fri)
# - The total_bill on Fri of males are much higher than that of females

# #### Box plot of tip as a function of day and sex

# In[24]:


sns.catplot(x="day", y="tip", hue= "sex", kind="box", data=tips)


# #### Observations from the above plot of tip vs day
# - The tips are a little higher on Sun for both genders
# - There are quite a few outlier values on Sat
# - The tip value for males are higher than that of females

# ## Conclusions
# - There are no total_bill data (hence no customers) which implies that the restaurant/hotel is only open during dinner in the weekends. 
# - On Thursday ('Thur') customers only come during lunch time, hence the restaurant may think of shutting down during dinner time. Since there are no customers during dinner the restaurant only incur loss if they keep open at night on Thursday. 
# - It may be suggested that the restaurant can only operate on weekends during dinner time (if they are open during both lunch and dinner). 
# - The percentage of tip with respect to the total bill for Thur, Fri, Sat and Sun are 15.67%, 15.94%, 14.64% and 15.20 % respectively and hence the amount of tip does not vary much in the weekdays or weekend. 
# - Most of the tips lie between 2 and 4 irrespective of the gender and there are outlier values for bothWe can infer that the total_bill is higher on weekends (Sat, Sun) than on weekdays (Thur, Fri)
# - The total_bill on Fri of males are much higher than that of females
# - The tips are a little higher on Sun for both genders than on weekdays
