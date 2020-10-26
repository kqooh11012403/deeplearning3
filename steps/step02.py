#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np


# In[1]:


class Variable:
    def __init__(self, data):
        self.data = data


# In[15]:


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError()


# In[16]:


class Square(Function):
    def forward(self, x):
        return x ** 2


# In[17]:


x = Variable(np.array(10))
f = Square()
y = f(x)

print(type(y))
print(y.data)


# In[ ]:




