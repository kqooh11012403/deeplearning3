#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np


# In[29]:


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
        
        self.data = data
        self.grad = None
        self.creator = None
    
    def set_creator(self, func):
        self.creator = func
        
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)
            
            if x.creator is not None:
                funcs.append(x.creator)


# In[42]:


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self)
        self.input = input
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError()
        
    def backward(self, gy):
        raise NotImplementedError()


# In[43]:


class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


# In[44]:


class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


# In[45]:


def square(x):
    f = Square()
    return f(x)

def exp(x):
    f = Exp()
    return f(x)


# In[48]:


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


# In[49]:


x = Variable(np.array(0.5))
a = square(x)
b = exp(a)
y = square(b)

y.backward()
print(x.grad)


# In[ ]:




