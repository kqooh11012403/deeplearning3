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


# In[4]:


class Square(Function):
    def forward(self, x):
        return x ** 2


# In[5]:


class Exp(Function):
    def forward(self, x):
        return np.exp(x)


# In[8]:


def numerical_diff(f, x, eps = 1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


# In[9]:


f = Square()
x = Variable(np.array(2.0))
dy = numerical_diff(f, x)
print(dy)


# In[10]:


def f(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))


# In[11]:


x = Variable(np.array(0.5))
dy = numerical_diff(f, x)
print(dy)


# In[ ]:




