#!/usr/bin/env python
# coding: utf-8

# ## NAME :  SONALI PATIL
# ## TASK 1 : Advance Level Task at LetsGrowMore
# ## TASK NAME : Develop A Neural Network That Can Read Handwriting
# 
# 
# 
# 
# 

# In[1]:


import tensorflow as tf ##pip install tensorflow 


# # Loading dataset : "MNIST DATASET"
# # Conataining training samples =60k imag
# ## testing samples=10k
# # tensorflow already conatining the MNIST dataset which is loaded by keras

# In[2]:


mnist=tf.keras.datasets.mnist # this is basically handwritten character based on 28*28 sized images of 0-9


# # After loading MNIST dataset , divide into train and test dataset 

# In[3]:


(x_train,y_train),(x_test,y_test)=mnist.load_data()


# In[4]:


x_train.shape # every img has 28*28 size


# In[5]:


# check the graph 
import matplotlib .pyplot as plt
plt.imshow(x_train[0])
plt.show()# execute the graph
plt.imshow(x_train[0],cmap=plt.cm.binary) ## binary image


# # checking value of each pixel
# # before normalization

# In[6]:


print(x_train[0]) # before normalization


# # AS images are in gray level
# # Normalizing the data / Pre-processing steps

# In[7]:


#gray imges and all values varies from 0-255
# for normalize two methods x_train/255 and 2nd is build in cmd

x_train=tf.keras.utils.normalize(x_train,axis=1) # cmd to normalize
x_test=tf.keras.utils.normalize(x_test,axis=1)
plt.imshow(x_train[0],cmap=plt.cm.binary)


# # AFETR NORMALIZATION

# In[8]:


print(x_train[0])# check the values(all the values are from 0-1)bcaz /255
##print(x_test[0])


# In[9]:


print(y_train[0])# chk that we have label our n/w


# # resizing img to make it suitable for apply Convolution operation

# In[10]:


import numpy as np
IMG_SIZE=28
x_trainr=np.array(x_train).reshape(-1,IMG_SIZE,IMG_SIZE,1)# increasing one dimension for kernal operation
x_testr=np.array(x_test).reshape(-1,IMG_SIZE,IMG_SIZE,1)# increasing one dimension for kernal operation
print("Trainig samples dimensions ",x_trainr.shape)
print("Testing samples dimensions ",x_testr.shape)


# # creating deep neural network
# # trainig on 60k samples of MNIST handwritten dataset

# In[11]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense ,Dropout,Activation,Flatten,Conv2D,MaxPooling2D


# In[12]:


#creating neural network
model = Sequential()

# first convolution layer 0,1,3,....  28-3+1=26
model.add(Conv2D(64, (3,3), input_shape = x_trainr.shape[1:]))
model.add(Activation("relu"))## activation function to make it non-linear 
model.add(MaxPooling2D(pool_size=(2,2)))## Maxpooling size pool is 2*2


# second convolution layer #13
model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))



# third convolution layer
model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))


## fully connected layer
model.add(Flatten())## before using fully connected layer need to flatten
model.add(Dense(64))
model.add(Activation("relu"))


## fully connected layer 2 o/p must be equal to no. of classes ,10 (0-9)
model.add(Dense(32))
model.add(Activation("relu"))

## last fully connected layer
model.add(Dense(10))# last dense layer must be equal to 10
model.add(Activation('softmax')) ## activation function is changed to softmax( class probabilities)


# In[13]:


model.summary()


# In[14]:


print("Total trainig sampes=",len(x_trainr))


# In[15]:


model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=['accuracy'])


# In[16]:


model.fit(x_trainr,y_train,epochs=5,validation_split=0.3)## trainig my model


# In[17]:


##evaluating  on testing data set MNIST
test_loss,test_acc=model.evaluate(x_testr,y_test)
print("Test Loss on 10,000 test samples",test_loss)
print("Validation Accuracy on 10,000 test samples",test_acc)


# In[18]:


predicions = model.predict([x_testr])


# In[19]:


print(predicions)


# In[20]:


print(np.argmax(predicions[0]))


# In[21]:


plt.imshow(x_test[0])


# In[22]:


print(np.argmax(predicions[128]))


# In[23]:


plt.imshow(x_test[128])


# In[24]:


import cv2# pip install opencv-pyhthon


# In[25]:


img=cv2.imread('C:\\Users\\sonali\\OneDrive\\Desktop\\eight.png')


# In[26]:


plt.imshow(img)


# In[27]:



img.shape


# In[28]:


gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


# In[29]:


gray.shape


# In[30]:


resized=cv2.resize(gray,(28,28),interpolation=cv2.INTER_AREA)


# In[31]:


resized.shape


# In[32]:


newimg=tf.keras.utils.normalize(resized, axis=1)


# In[33]:


newimg=np.array(newimg).reshape(-1,IMG_SIZE,IMG_SIZE,1)


# In[34]:


newimg.shape


# In[35]:


predictions=model.predict(newimg)


# In[36]:


print(np.argmax(predictions))


# # Thank you !!!!!!!

# In[ ]:




