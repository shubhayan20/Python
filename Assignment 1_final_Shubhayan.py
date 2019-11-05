#!/usr/bin/env python
# coding: utf-8

# # Assignment 1: Shubhayan Bhattacharya, IIT Madras (GroupB, ML)

# In[1]:


import numpy as np
import pandas as pd


# ### Question 1: Convert tuple to list, set, Dictionary 
# 

# In[2]:


tuple_cars = (('USA','Ford'), ('Japan','Honda'), ('Italy','Alfa Romeo'), ('India', 'Maruti'))
print (tuple_cars)
type (tuple_cars)


# In[3]:


list1=list (tuple_cars)
print (list1)
type (list1)


# In[4]:


set1=set (tuple_cars)
print(set1)
type(set1)


# In[5]:


dict1=dict (tuple_cars)
print (dict1)
type (dict1)


# ### Question 2: Convert list to tuple, set, Dictionary 
# 

# In[18]:


list_cars =[('USA', 'Ford'),('Japan', 'Honda'),('Italy', 'Alfa Romeo'),('India', 'Maruti')]
print(list_cars)
type(list_cars)


# In[10]:


tuple1= tuple (list_cars)
print (tuple1)
type (tuple1)


# In[11]:


set2=set (list_cars)
print(set2)
type(set2)


# In[12]:


dict1=dict (list_cars)
print (dict1)
type(dict1)


# ### Question 3: Convert set to list, tuple, Dictionary 

# In[13]:


print (set2)
type (set2)


# In[14]:


list3=list (set2)
print (list3)
type (list3)


# In[15]:


tuple2= tuple (set2)
print (tuple2)
type (tuple2)


# In[16]:


dict2=dict(set2)
print (dict2)
type (dict2)


# ### Question 4: Convert Dictionary to list, set, tuple

# In[17]:


print (dict2)
type (dict2)


# In[18]:


list4 = [(k, v) for k, v in dict2.items()]
print (list4)
type (list4)


# In[19]:


set3= set (dict2)
print (set3)
type (set3)


# In[20]:


import collections
list5 = collections.namedtuple('List', 'name value') 
print (list5)
type (list5)


# ### Question 5: Create a list and perform pop, append, update, remove, delete, add element and sort operation

# In[21]:


list_fruits= ['apple', 'banana', 'guava', 'watermelon', 'orange']
type (list_fruits)


# In[22]:


print (list_fruits)


# In[23]:


z= list_fruits.pop()
print (z)


# In[24]:


list_fruits.append('apple')
print(list_fruits)


# In[114]:


list_fruits.remove('apple')
print(list_fruits)


# In[4]:


# sort option
listnumbers=[14, 56, 78, 93, 43]
listnumbers.sort()
listnumbers


# ### Question 6: Create a tuple and perform pop, append, update, remove, delete, add element and sort operation

# In[22]:


# tuple is immutable and most of the above operations cannot be performed
list_new=[1,3, 5, 87, 45, 22]
tuple_new=tuple(list_new)
type(tuple_new)


# In[26]:


sorted(tuple_new)


# ### Question 7: Create a set and perform pop, append, update, remove, delete, add element and sort operation

# In[6]:


set_numbers = {13, 34, 56, 78, 98, 108, 212}
type(set_numbers)


# In[7]:


#perform pop
pop1=set_numbers.pop()
set_numbers


# In[10]:


# performing add
set_numbers.add(68)
set_numbers


# In[15]:


# performing remove
set_numbers.remove('68')
set_numbers


# ### Question 9: How to make a copy of list, tuple and dictionary 

# In[29]:


# for list
old_list_primenumbers = [2, 3, 5, 7, 11]
new_list = old_list_primenumbers.copy()

# add element to list
new_list.append('a')

print('New List:', new_list )
print('Old List:', old_list_primenumbers )


# In[34]:


# for tuple
tuple_cars = (('USA','Ford'), ('Japan','Honda'), ('Italy','Alfa Romeo'), ('India', 'Maruti'))
newt = tuple_cars
# tuple is immutable


# In[36]:


#for dict
dict1 = {"fruit": "orange", "car": "Ford"}
dict2 = dict1.copy()
dict2


# ### Question 10: Write a function which accept input from keyboard and check if student grade should be First division, second division or Fail 

# In[38]:


print("Enter 'x' for exit.");
print("Enter marks obtained in 3 subjects: ");
mark1 = input();
if mark1 == 'x':
    exit();
else:
    sub1 = int(mark1);
    sub2 = int(input());
    sub3 = int(input());
    sum = sub1 + sub2 + sub3;
    average = sum/3;
    if(average>=80 and average<=100):
    	print("Your division is 1");
    elif(average>=61 and average<=79):
    	print("Your division is 2");
    elif(average>=41 and average<=60):
    	print("Your division is 3");
    elif(average>=0 and average<=39):
    	print("Your division is fail");


#  ### Question 11-13: 
# - teamA = {'India', 'Australia','Pakistan', 'England'}
# - teamB = {'Bangladesh', 'New Zealand', 'West Indies', 'India'}
# - use add, clear, copy, difference, differenc_update(), discard, intersection, intersection_update, 
# - remove, symmetric_difference,  union, update 
# 

# In[42]:


teamA = {'India', 'Australia','Pakistan', 'England'}
teamB = {'Bangladesh', 'New Zealand', 'West Indies', 'India'}
# using add
teamA.add('Netherlands')
teamA


# In[44]:


# using clear 
teamA.clear()
teamA


# In[45]:


# using copy
teamA = {'India', 'Australia','Pakistan', 'England'}
teamC=teamA.copy()
teamC


# In[46]:


# using difference
z=teamA.difference(teamB)
z


# In[49]:


# using differenc_update()
y=teamA.difference_update(teamB)
print(y)


# In[52]:


# using discard
teamA.discard('Australia')
teamA.discard('Pakistan')
teamA


# In[55]:


# using intersection
teamA = {'India', 'Australia','Pakistan', 'England'}
teamB = {'Bangladesh', 'New Zealand', 'West Indies', 'India'}
c = teamA.intersection (teamB)
print(c)


# In[57]:


# using intersection_update
d=teamA.intersection_update(teamB)
print(d)


# In[60]:


# using remove function
teamA.remove("India")
teamA


# In[61]:


# symmetric difference 
teamA = {'India', 'Australia','Pakistan', 'England'}
teamB = {'Bangladesh', 'New Zealand', 'West Indies', 'India'}
teamA.symmetric_difference(teamB)


# In[62]:


# using union
teamA = {'India', 'Australia','Pakistan', 'England'}
teamB = {'Bangladesh', 'New Zealand', 'West Indies', 'India'}
teamA.union(teamB)


# ### Question 14:	What is list comprehension explain with 2 examples

# #### List comprehensions in python is a short and concise way of creating new lists using sequences which have already been defined. 

# ### Example 1

# In[9]:


input_list_numbers = [61, 62, 73, 54, 84, 55, 61, 74, 47] 
  
output_list = [] 
# We wish to create a list with only even numbers 
for var in input_list_numbers: 
    if var % 2 == 0: 
        output_list.append(var) 
  
print("Output List using for loop:", output_list) 


# ### Example 2

# In[10]:


squarenumber=[i**2 for i in range(2,10)]
print (squarenumber)


# ### Question 15: What is the difference between List and tuple, explain with example

# #### Major differences between list and tuple are as follows:
# - List is mutable but tuple is immutable
# - Syntax for list and tuple are different
# - Tuples can consist of heterogeneous data structures but list consists of homogeneous data sequences

# In[ ]:


listdays =['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun']
type (listdays)


# #### Mutability

# In[25]:


listdays[2]='Wednesday'
print (listdays)


# In[19]:


tupledays=('Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun')
type (tupledays)


# #### Immutability

# In[26]:


tupledays(2)='Wednesday'


# ### Question 16: What is the output of

# In[116]:


if 1 and 0:
    print("Sucess 1 and 0")
elif 0 and 0:
    print("Sucess 0 and 0")
elif 0 and 1 or 1:
    print("Sucess 0 and 1 or 1")
elif 0 and 0 or 1 and 1:
    print("Sucess : 0 and 0 or 1 and 1")


# ### Question 17: What is the output of

# In[117]:


list_of_lists = [[10, 20, 30], [40, 50, 60], [70, 80, 90]]
for list1 in list_of_lists: 
    for x in list1: 
        print(x)
    print(list1)


# ### Question 18: Print  the numpy version and the configuration

# In[118]:


print(np.__version__)


# In[119]:


print(np.show_config())


# ### Question 19: Create a vector with values ranging from 10 to 29

# In[132]:


from numpy import array
w=np.arange(10,30)
v= array (w)
print(v)
type(v)


# ### Question 20: Create a 3x3 matrix with values ranging from 0 to 8 

# In[135]:


x =  np.arange(0, 9).reshape(3,3)
print(x)


# ### Question 21: Create a 10x10 array with random values and find the minimum and maximum values

# In[136]:


x = np.random.random((10,10))
print("Original Array:")
print(x) 
xmin, xmax = x.min(), x.max()
print("Min and Max Values:")
print(xmin, xmax)


# ### Question 22: Create a random vector of size 30 and find the mean and median value 

# In[157]:


y = np.random.random(30)
print("Original array:")
print(y)
import statistics
xmean,xmedian = statistics.mean(y),statistics.median(y)
print ("Meanvalue and Medianvalue")
print(xmean,xmedian)


# ### Question 23: Define an array and find the max value and index of max value

# In[163]:


s=np.array([14,17,87,105,64,89,12])
Maxvalue=np.amax(s)
print(Maxvalue)


# In[166]:


result = np.where(s == np.amax(s)) 
print('Returned tuple of arrays :', result)
print('List of Indices of maximum element :', result[0])


# ### Question 24: How to find the memory size of any array

# In[170]:


n = np.zeros((1440,6))
print("%d bytes" % (n.size * n.itemsize))


# ### Question 25: Create a 3x3 identity matrix 

# In[179]:


a = np.identity(3) 
print("\nMatrix a : \n", a) 


# In[ ]:




