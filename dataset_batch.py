#!/usr/bin/env python
# coding: utf-8

# In[1]:


# What version of Python do you have?
import sys
import tensorflow.keras
import pandas as pd
import sklearn as sk
import tensorflow as tf
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage

np.random.seed(1)


# This module will be to create batches for our dataset that we will use


# In[1]:


def minibatch_train(train_batch_size,seed):
    
    hdf5_path = 'C:/Users/Abo_Alon/Desktop/machine/Final Project/dataset_small.h5' # Change directory based on file you want to use
    dataset = h5py.File(hdf5_path, "r")
    
    np.random.seed(seed)
    
    # shuffle indexes,int numbers range from 0 to 8012 (This number depends on the train/test set created in the "Data Setup" file)
    permutation = list(np.random.permutation(8012))
    
    # get the "train_batch_size" indexes    
    train_batch_index=permutation[0:train_batch_size]
    
    # the shape of "train_labels" now is (8012,1)
    train_labels=np.array(dataset["train_labels"]).reshape(8012,-1)
    
    # get the corresponding labels according "train_batch_index"
    train_batch_labels=train_labels[train_batch_index]

    train_batch_labels= np.eye(2)[train_batch_labels.reshape(-1)] #convert to one_hot code
    
    train_batch_imgs=[]
    for i in range(train_batch_size):
        img=(dataset['train_img'])[train_batch_index[i]]
        img=img/255.
        train_batch_imgs.append(img)    
    train_batch_imgs=np.array(train_batch_imgs)
    
    dataset.close()
    
    return(train_batch_imgs,train_batch_labels)

def minibatch_test(test_batch_size,seed): 
     
    hdf5_path = hdf5_path = 'C:/Users/Abo_Alon/Desktop/machine/Final Project/dataset_small.h5'
    dataset = h5py.File(hdf5_path, "r")
    
    np.random.seed(seed)
    
    permutation = list(np.random.permutation(2003))    
    test_batch_index= permutation[0:test_batch_size]  
    test_labels= np.array(dataset["test_labels"]).reshape(2003,-1)
    test_batch_labels= test_labels[test_batch_index]
    test_batch_labels= np.eye(2)[test_batch_labels.reshape(-1)]
    
    test_batch_imgs=[]
    for i in range(test_batch_size):
        img=(dataset['test_img'])[test_batch_index[i]]
        img=img/255.
        test_batch_imgs.append(img)    
    test_batch_imgs=np.array(test_batch_imgs)
    
    dataset.close()  
    
    return(test_batch_imgs,test_batch_labels)


# In[ ]:




