#!/usr/bin/env python
# coding: utf-8

# # What is PyTorch?
# It’s a Python-based scientific computing package targeted at two sets of audiences:
# 
# A replacement for NumPy to use the power of GPUs
# a deep learning research platform that provides maximum flexibility and speed
# 
# 
# Tensors are similar to NumPy’s ndarrays, with the addition being that Tensors can also be used on a GPU to accelerate computing.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from __future__ import print_function
import torch


# In[3]:


x = torch.empty(5, 3)
print(x)


# # Note
# An uninitialized matrix is declared, but does not contain definite known values before it is used. When an uninitialized matrix is created, whatever values were in the allocated memory at the time will appear as the initial values.
# 
# Construct a 5x3 matrix, uninitialized:

# In[4]:


x = torch.rand(5, 3)
print(x)


# # Construct a randomly initialized matrix:

# In[5]:


x = torch.zeros(5, 3, dtype=torch.long)
print(x)


# # Construct a matrix filled zeros and of dtype long:

# In[6]:


x = torch.tensor([5.5, 3])
print(x)


# Construct a tensor directly from data:

# In[7]:


x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)    # override dtype!
print(x)


# or create a tensor based on an existing tensor. 
# These methods will reuse properties of the input tensor, e.g. dtype, unless new values are provided by user

# In[8]:


print(x.size())


# Note torch.Size is in fact a tuple, so it supports all tuple operations.
# 
# There are multiple syntaxes for operations. In the following example, we will take a look at the addition operation.
# 
# Addition: syntax 1

# In[9]:


y = torch.rand(5, 3)
print(x + y)


# Addition: syntax 2

# In[10]:


print(torch.add(x, y))


# Addition: providing an output tensor as argument

# In[11]:


result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)


# Addition: in-place

# In[12]:


# adds x to y
y.add_(x)
print(y)


# Note
# 
# Any operation that mutates a tensor in-place is post-fixed with an ``_``. For example: ``x.copy_(y)``, ``x.t_()``, will change ``x``.
# 
# You can use standard NumPy-like indexing with all bells and whistles!

# In[13]:


print(x[:, 1])


# Resizing: If you want to resize/reshape tensor, you can use torch.view:

# In[14]:


x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())


# If you have a one element tensor, use .item() to get the value as a Python number

# In[15]:


x = torch.randn(1)
print(x)
print(x.item())


# NumPy Bridge
# Converting a Torch Tensor to a NumPy array and vice versa is a breeze.
# 
# The Torch Tensor and NumPy array will share their underlying memory locations (if the Torch Tensor is on CPU), and changing one will change the other.
# 
# Converting a Torch Tensor to a NumPy Array

# In[16]:


a = torch.ones(5)
print(a)


# In[17]:


b = a.numpy()
print(b)


# how the numpy array changed in value.

# In[18]:


a.add_(1)
print(a)
print(b)


# Converting NumPy Array to Torch Tensor 
# See how changing the np array changed the Torch Tensor automatically

# In[19]:


import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)


# All the Tensors on the CPU except a CharTensor support converting to NumPy and back.
# 
# CUDA Tensors
# Tensors can be moved onto any device using the .to method.

# In[20]:


# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!


# In[ ]:




