import os
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from keras import backend as K
from keras.models import load_model
import statistics

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.333)
sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
Name = "Unet"
tensorboard=TensorBoard(log_dir="logs\{}".format(Name))

#Data Generator
class DataGenerator(keras.utils.Sequence):
    def __init__(self,ids,path,batch_size,image_width,image_height,image_file_type,image_file_name,masks_file_name):
        self.ids=ids
        self.path=path
        self.batch_size=batch_size
        self.image_width=image_width
        self.image_height=image_height
        self.image_file_type=image_file_type
        self.image_file_name=image_file_name
        self.masks_file_name=masks_file_name
        self.on_epoch_end()
        
    def __load__(self,id_name):
        image_path=os.path.join(self.path,self.image_file_name,id_name)+image_file_type
        mask_path=os.path.join(self.path,masks_file_name)
        
        ## Reading Image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (self.image_width, self.image_height))
        image = np.expand_dims(image, axis=-1)
        
        #Initializing mask value
        mask=np.zeros((self.image_width,self.image_height,1))
        
        
        _mask_path = os.path.join(mask_path, id_name) + "_label" + image_file_type
        
        

#################################################
# The code below can solve the wrong mask issue
#################################################
        if os.path.exists(_mask_path):
            #-1 for greyscale as the mask has a value of 255
            _mask_image = cv2.imread(_mask_path, -1)
            _mask_image = cv2.resize(_mask_image, (self.image_width, self.image_height)) #128x128
            _mask_image=  cv2.subtract(255, _mask_image) #convert the image to 
            _mask_image = np.expand_dims(_mask_image, axis=-1)
            mask = np.maximum(mask, _mask_image) #maximum because the white part is the label
        else:
            print(_mask_path)
            mask = cv2.resize(mask, (image_width, image_height)) #128x128
            mask=  cv2.subtract(255, mask) #convert the image to 
            cv2.imwrite(_mask_path, mask)


        #Normalising
        image=image/255.0
        mask=mask/255.0
    
        return image,mask
    
    def __getitem__(self,index):
        if(index+1)*self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index*self.batch_size
        
        files_batch = self.ids[index*self.batch_size : (index+1)*self.batch_size]
        
        image = []
        mask  = []
        
        for id_name in files_batch:
            _img, _mask = self.__load__(id_name)      
            image.append(_img)
            mask.append(_mask)
            
        image=np.array(image)
        mask=np.array(mask)
            
        return image,mask

    def on_epoch_end(self):
        pass
    
    def __len__(self):
        return int(np.ceil(len(self.ids)/float(self.batch_size)))

    
#Hyperparameters
image_width= 128
image_height= 128
train_path =r"D:\Project\FYP\2020\Dynamo\Dynamo without defects nogoodimages"

epochs=100
batch_size= 30 
image_file_type=".png"
image_file_name="Images"
masks_file_name="Masks/"

#Extracting and aligning the images with the mask via the file names
train_ids = []
for (filenames) in os.walk(os.path.join(train_path,image_file_name)):
    for filename in filenames[2]:
        if filename.endswith(image_file_type):
            train_ids.append(re.sub(image_file_type,'',filename))

val_data_size = 10

valid_ids = train_ids[:val_data_size]
train_ids = train_ids[val_data_size:]

train_gen = DataGenerator(train_ids, train_path, image_width=image_width, image_height = image_height, batch_size=batch_size , image_file_type=image_file_type , image_file_name= image_file_name , masks_file_name = masks_file_name)
valid_gen = DataGenerator(valid_ids, train_path, image_width=image_width, image_height = image_height, batch_size=batch_size , image_file_type=image_file_type , image_file_name= image_file_name , masks_file_name = masks_file_name)

x,y = train_gen.__getitem__(0)
x1,y1 = valid_gen.__getitem__(0)

#################################################
# you can uncomment the code below to check the raw image and its mask match
#################################################
#for r in range(len(x)):
#    fig = plt.figure()
#    fig.subplots_adjust(hspace=0.4, wspace=0.4)
#    ax = fig.add_subplot(1, 2, 1)
#    ax.imshow(np.reshape(x[r], (image_width, image_height)), cmap="gray")
#    ax = fig.add_subplot(1, 2, 2)
#    ax.imshow(np.reshape(y[r], (image_width, image_height)), cmap="gray")

###################################
#
def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
    return c, p

def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    us = keras.layers.UpSampling2D((2, 2))(x)
    concat = keras.layers.Concatenate()([us, skip])
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c
def UNet():
    f = [16, 32, 64, 128, 256]
#    f = [8, 16, 32, 64, 128]
    inputs = keras.layers.Input((image_width, image_height, 1))
    
    p0 = inputs
    c1, p1 = down_block(p0, f[0]) #128 -> 64
    c2, p2 = down_block(p1, f[1]) #64 -> 32
    c3, p3 = down_block(p2, f[2]) #32 -> 16
    c4, p4 = down_block(p3, f[3]) #16->8
    
    bn = bottleneck(p4, f[4])
    
    u1 = up_block(bn, c4, f[3]) #8 -> 16
    u2 = up_block(u1, c3, f[2]) #16 -> 32
    u3 = up_block(u2, c2, f[1]) #32 -> 64
    u4 = up_block(u3, c1, f[0]) #64 -> 128
    
    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u4)
    model = keras.models.Model(inputs, outputs)
    return model
model = UNet()
model.compile(optimizer="adam",loss="binary_crossentropy", metrics=["acc"])
model.summary()
train_gen = DataGenerator(train_ids, train_path, image_width=image_width, image_height = image_height, batch_size=batch_size , image_file_type=image_file_type , image_file_name= image_file_name , masks_file_name = masks_file_name)
valid_gen = DataGenerator(valid_ids, train_path, image_width=image_width, image_height = image_height, batch_size=batch_size , image_file_type=image_file_type , image_file_name= image_file_name , masks_file_name = masks_file_name)

train_steps = len(train_ids)//batch_size
valid_steps = len(valid_ids)//batch_size


#################################################
# I use below fit_generator, just change the epochs here
#################################################
model.fit_generator(train_gen, steps_per_epoch=33, epochs=epochs,validation_data=valid_gen,validation_steps=150,callbacks=[tensorboard])


#################################################
# you can save the result so that save you debuging time
#################################################
#### Save the Weights
#model.save_weights("Model/UNetW.h5")
#model.load_weights("Model/UNetW.h5")

# Dataset for prediction
x, y = valid_gen.__getitem__(0)
result = model.predict(x1)
result = result > 0.5


#################################################
# check all the validation result
#################################################
def iou_coeff(target,prediction):
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

array=[]
for r in range(len(x1)):
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(np.reshape(x1[r], (image_width, image_height)), cmap="gray")
    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(np.reshape(y1[r]*255, (image_width, image_height)), cmap="gray")
    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(np.reshape(result[r]*255, (image_width, image_height)), cmap="gray")
    x=iou_coeff(y1[r],result[r])
    array.append(x)
mean_iou_score = statistics.mean(array)
print(mean_iou_score)
plt.plot(array)

#####################################################
#Tensorboard: Type ====>tensorboard --logdir=D:\Project\FYP\Untitled_Message\logs\Unet --host=127.0.0.1
#into http://localhost:6006/
#tensorboard --logdir=D:\Project\FYP\FYP-code_and_images\logs\Unet --host=127.0.0.1