#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# Create a tensor and set requires_grad=True to track computation with it

# In[2]:


import torch


# In[15]:


x = torch.ones(2, 2, requires_grad=True)
print(x)


# Do a tensor operation

# In[4]:


y = x + 2
print(y)


# y was created as a result of an operation, so it has a grad_fn.

# In[5]:


print(y.grad_fn)


# Do more operations on y

# In[6]:


z = y * y * 3
out = z.mean()

print(z, out)


# .requires_grad_( ... ) changes an existing Tensor's requires_grad flag in-place. 
# The input flag defaults to False if not given.

# In[7]:


a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)


# Gradients
# Let's backprop now. Because out contains a single scalar, out.backward() is equivalent to out.backward(torch.tensor(1.))

# In[8]:


out.backward()


# Print gradients d(out)/dx

# In[9]:


print(x.grad)


# Now in this case y is no longer a scalar. torch.autograd could not compute the full Jacobian directly,
# but if we just want the vector-Jacobian product, simply pass the vector to backward as argument:

# In[10]:


x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)


# In[11]:


v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)


# You can also stop autograd from tracking history on Tensors with .requires_grad=True either by wrapping the code block in with torch.no_grad():

# In[12]:


print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
	print((x ** 2).requires_grad)


# In[13]:


print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())


# In[ ]:




