#importing relevant libraries
import os
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve
import tensorflow as tf
import h5py
###########################################################################

class DataGenerator(keras.utils.Sequence):
    def __init__(self,ids,path,batch_size,image_width,image_height,image_file_type,image_file_name):
        self.ids=ids
        self.path=path
        self.batch_size=batch_size
        self.image_width=image_width
        self.image_height=image_height
        self.image_file_type=image_file_type
        self.image_file_name=image_file_name
        self.on_epoch_end()
        
    def __load__(self,id_name):
        image_path=os.path.join(self.path,self.image_file_name,id_name)+image_file_type
        
        
        ## Reading Image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (self.image_width, self.image_height))
        image = np.expand_dims(image, axis=-1)
        
        #Normalising
        image=image/255.0
        return image
    
    def __getitem__(self,index):
        if(index+1)*self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index*self.batch_size
        
        files_batch = self.ids[index*self.batch_size : (index+1)*self.batch_size]
        image = []
        label = []
        
        types_of_defects=[]
        for id_name in files_batch:
            _img = self.__load__(id_name)
            image.append(_img)
            label.append(id_name.split("_")[0])
            for i in label:
                if i not in types_of_defects:
                    types_of_defects.append(i)
                    
        for n , i in enumerate(label):
            if i in types_of_defects:
                label[n]=types_of_defects.index(i)  
        image=np.array(image)
        label=np.asarray(label)
        return image,label

    def on_epoch_end(self):
        pass
    
    def __len__(self):
        return int(np.ceil(len(self.ids)/float(self.batch_size)))


####################################################################################
#Parameters
####################################################################################
image_width= 128
image_height= 128
train_path =r"D:\Project\FYP\Data For Gavin\batch1"

epochs=50
batch_size= 8 
image_file_type=".bmp"
image_file_name="Images"

train_ids = []
for (filenames) in os.walk(os.path.join(train_path,image_file_name)):
    for filename in filenames[2]:
        if filename.endswith(image_file_type):
            train_ids.append(re.sub(image_file_type,'',filename))
#print(train_ids)
val_data_size = 10

valid_ids = train_ids[:val_data_size]
train_ids = train_ids[val_data_size:]

train_gen = DataGenerator(train_ids, train_path, image_width=image_width, image_height = image_height, batch_size=batch_size , image_file_type=image_file_type , image_file_name= image_file_name )
valid_gen = DataGenerator(valid_ids, train_path, image_width=image_width, image_height = image_height, batch_size=batch_size , image_file_type=image_file_type , image_file_name= image_file_name )
x,y = train_gen.__getitem__(0)
x,label=valid_gen.__getitem__(5)
    
####################################################################################

model=tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(64,(3,3),input_shape=x.shape[1:],padding='same'))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Conv2D(64,(3,3),padding='same'))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64))
model.add(tf.keras.layers.Dense(1))
model.add(tf.keras.layers.Activation("softmax"))

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit_generator(train_gen, steps_per_epoch=32, epochs=1,validation_data=valid_gen,validation_steps=150)
model.summary
model.save_weights("Model/Classifier.h5")
model.load_weights("Model/Classifier.h5")






