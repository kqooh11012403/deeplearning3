#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[40]:


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


# In[35]:


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys, )
        outputs = [Variable(as_array(y)) for y in ys]
        
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()
        
    def backward(self, gys):
        raise NotImplementedError()


# In[36]:


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy


# In[37]:


def add(x0, x1):
    return Add()(x0, x1)


# In[38]:


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


# In[39]:


x0 = Variable(np.array(2))
x1 = Variable(np.array(3))
ys = add(x0, x1)
print(ys.data)


# In[ ]:




