#!/usr/bin/env python
# coding: utf-8

# In[1]:


class Variable:
    def __init__(self, data):
        self.data = data


# In[2]:


import numpy as np

data = np.array(1.0)
x = Variable(data)
print(x.data)


# In[3]:


x.data = 2.0
print(x.data)


# In[ ]:




