#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


class Variable:
    def __init__(self, data):
        self.data = data


# In[3]:


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError()


# In[5]:


class Square(Function):
    def forward(self, x):
        return x ** 2


# In[6]:


class Exp(Function):
    def forward(self, x):
        return np.exp(x)


# In[7]:


A = Square()
B = Exp()
C = Square()

x =  Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)
print(y.data)


# In[ ]:




