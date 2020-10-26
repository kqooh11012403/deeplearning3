#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[10]:


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
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            
            for x, gx in zip(f.inputs, gxs):
                x.grad = gx
                
                if x.creator is not None:
                    funcs.append(x.creator)


# In[11]:


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        
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


# In[12]:


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y
        
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


# In[13]:


def square(x):
    return Square()(x)


# In[14]:


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy


# In[15]:


def add(x0, x1):
    return Add()(x0, x1)


# In[16]:


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


# In[17]:


x = Variable(np.array(2))
y = Variable(np.array(3))

z = add(square(x), square(y))
z.backward()
print(z.data)
print(x.grad)
print(y.grad)


# In[ ]:




