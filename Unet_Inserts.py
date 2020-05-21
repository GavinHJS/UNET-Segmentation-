#######################################################################################################
#Import libraries
#######################################################################################################
import os
import re
import numpy             as np
import cv2
import matplotlib.pyplot as plt
import statistics
import tensorflow        as tf
from tensorflow.keras.callbacks import TensorBoard
from keras                      import backend as K
from keras.models               import load_model
from tensorflow                 import keras

###########################################################################################################################################
#For tensorboard
############################################################################################################################################

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.333)     #Split up GPU usage to 1/3 
sess        = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) 
Name        = "Unet"                                                     #Naming tensorboard directory
tensorboard = TensorBoard(log_dir="logs\{}".format(Name))                #Assigning tensorboard directory
############################################################################################################################################
#How to use tensorboard
############################################################################################################################################
# 1) Run the code, make sure model.fit_generator has callback assigned to tensorboard as shown below
# 2) After running the model, the information on the model should be saved in the assigned directory
# 3) Open up python's command prompt  
# 4) Type in tensorboard directory (..\logs\Unet) into YOUR_LOG_DIR======> tensorboard --logdir=YOUR_LOG_DIR --host=127.0.0.1
# 5) Type in =====> localhost:6006 to google chrome to asses dashboard
# 6) Tensorboard can be used to optimise the model
############################################################################################################################################
#Class for data generator
############################################################################################################################################
class DataGenerator(keras.utils.Sequence):
    def __init__(self,ids,path,batch_size,image_width,image_height,image_file_type,image_file_name,masks_file_name,masks_file_type):
        self.ids             = ids             #Assigning image name into constructor
        self.path            = path            #Assigning main directory into constructor
        self.batch_size      = batch_size      #Assigning batch size ~ set this to 8 ideally as there are many instances where it crashes as it increases
        self.image_width     = image_width     #Assigning image width to resize the image to match the Unet input width
        self.image_height    = image_height    #Assigning image height to resize the image to match the Unet input height
        self.image_file_type = image_file_type #Assign the image file type ~ for dynamo :".png" | for inserts:".bmp" 
        self.image_file_name = image_file_name #Assigning the subdirectory for images
        self.masks_file_name = masks_file_name #Assigning the subdirectory for masks
        self.masks_file_type = masks_file_type #Assigning the mask file type ~ for both dynamo and inserts are ".png"
        self.on_epoch_end()
        
    def __load__(self,id_name):
        #Structuring directory path for both image and mask
        image_path = os.path.join(self.path,self.image_file_name,id_name)+image_file_type
        mask_path  = os.path.join(self.path,masks_file_name)        
        
        #Reading Image
        image      = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image      = cv2.resize(image, (self.image_width, self.image_height))
        image      = np.expand_dims(image, axis=-1)
        
        #Initializing mask value with array of zeros
        mask=np.zeros((self.image_width,self.image_height,1))
        
        ###################################################################################################################################
        #Uncomment this if you are using this for dynamo
        #_mask_path = os.path.join(mask_path, id_name) + "_label" + masks_file_type
        ###################################################################################################################################
        
        ###################################################################################################################################
        #For inserts or sliced image dataset, comment this out if you are using it for dynamo
        file_name=id_name[:-5]+"label"+id_name[-6:]+masks_file_type
        _mask_path = os.path.join(mask_path, file_name)  
        ###################################################################################################################################
        

        if os.path.exists(_mask_path):
            #To check if a mask for the corresponding image exist and reading the mask into an array
            _mask_image = cv2.imread(_mask_path, -1)                                     #-1 for greyscale as the mask has extreme value of either 0 or 255
            _mask_image = cv2.resize(_mask_image, (self.image_width, self.image_height)) #Resize mask to image size            
            _mask_image = cv2.subtract(255, _mask_image)                                 #Convert image from (white to black) to (black to white)
            _mask_image = np.expand_dims(_mask_image, axis=-1)                           #Reshape the array to include an additional dimension
         
            mask        = np.maximum(mask, _mask_image)                                  #maximum because the white part is the label
        
        else:
            #For this portion, when the image in the dataset does not have a corrsponding mask, it creates an empty mask for the image
            print(_mask_path)                                                            
            mask = cv2.resize(mask, (image_width, image_height)) #Resize to new mask to match image size
            mask = cv2.subtract(255, mask)                       #Inverting the colours
            cv2.imwrite(_mask_path, mask)                        #Saving the image into the directory
        
        #Normalising
        #Getting image value down to range 0~1
        image = image/255.0
        mask  = mask/255.0
        
        return image,mask
    
    #Aligning the images with the masks with the corresponding image
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

############################################################################################################################################
#Hyperparameters
############################################################################################################################################
image_width   = 256 #Make sure the size corresponds to the unet settings. Use====>  f = [32, 62, 128, 256, 512] if image size is 256
image_height  = 256 #Make sure the size corresponds to the unet settings. Use====>  f = [32, 62, 128, 256, 512] if image size is 256
#Main data directory
train_path    = r"D:\Project\FYP\Data For Gavin\Inserts3" #Main directory for dataset
image_file_name="Images-split"                            #Subdirectory for image dataset
masks_file_name="Masks-split"                             #Subdirectory for mask dataset
image_file_type=".bmp"                                    #Image file type
masks_file_type=".png"                                    #Mask file type

epochs        = 2000                                      #Adjust epochs
batch_size    = 8                                         #batch size to be fed into the model ~ keep this at 8 as increasing it might cause the code to crash
optimizer     = "adam"                                    #Optimizer for the UNet model ~ can be changed to optimize model
loss          = "binary_crossentropy"                     #Calculation method for loss
metrics       = ["acc"]                                   #Indication of accuracy value 

############################################################################################################################################

#Extracting and aligning the images with the masks via the file names
train_ids = []
for (filenames) in os.walk(os.path.join(train_path,image_file_name)):
    for filename in filenames[2]:
        if filename.endswith(image_file_type):
            train_ids.append(re.sub(image_file_type,'',filename))

#Setting validation size ~ can use train_test_split from sklearn if there is a need to randomize order of validation result            
val_data_size = 10
valid_ids     = train_ids[:val_data_size]
train_ids     = train_ids[val_data_size:]

############################################################################################################################################
#Activating class function to generate data
############################################################################################################################################
train_gen = DataGenerator(train_ids,
                          train_path,
                          image_width=image_width,
                          image_height = image_height,
                          batch_size=batch_size ,
                          image_file_type=image_file_type ,
                          image_file_name= image_file_name ,
                          masks_file_name = masks_file_name,
                          masks_file_type=masks_file_type)

valid_gen = DataGenerator(valid_ids,
                          train_path,
                          image_width=image_width,
                          image_height = image_height,
                          batch_size=batch_size ,
                          image_file_type=image_file_type ,
                          image_file_name= image_file_name ,
                          masks_file_name = masks_file_name,
                          masks_file_type=masks_file_type)

#Generating image and mask and assigning values to them
x,y   = train_gen.__getitem__(0)
x1,y1 = valid_gen.__getitem__(0)
############################################################################################################################################
#You can uncomment the code below to check the raw image and its mask match
############################################################################################################################################
for r in range(len(x)):
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    ax = fig.add_subplot(1, 2, 1)    
    ax.imshow(np.reshape(x[r], (image_width, image_height)), cmap="gray")
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(np.reshape(y[r], (image_width, image_height)), cmap="gray")

############################################################################################################################################
#Unet model
############################################################################################################################################
#Downblock function
def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
    return c, p

#Upblock function
def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    us = keras.layers.UpSampling2D((2, 2))(x)
    concat = keras.layers.Concatenate()([us, skip])
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

#Bottleneck function
def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

#UNet function
def UNet():
    f = [32, 62, 128, 256, 512]                                   #Use this f value for inserts with image size 256,256
#    f = [16, 32, 64, 128, 256]                                   #Use this f value for dynamo with image size of 128,128
    inputs = keras.layers.Input((image_width, image_height, 1))   #This specifies the input dimensions for the UNet model
    
    p0 = inputs
    c1, p1 = down_block(p0, f[0]) #256 -> 128
    c2, p2 = down_block(p1, f[1]) #128 -> 64
    c3, p3 = down_block(p2, f[2]) #64  -> 32
    c4, p4 = down_block(p3, f[3]) #32  ->16
    
    bn = bottleneck(p4, f[4])
    
    u1 = up_block(bn, c4, f[3])   #16  -> 32
    u2 = up_block(u1, c3, f[2])   #32  -> 64
    u3 = up_block(u2, c2, f[1])   #64  -> 128
    u4 = up_block(u3, c1, f[0])   #128 -> 256
    
    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u4)
    model = keras.models.Model(inputs, outputs)
    return model

model = UNet()                                                #Assigning the model
model.compile(optimizer=optimizer,loss=loss, metrics=metrics) #Compiling the model 
model.summary()                                               #Outputs the architecture of the model

train_gen = DataGenerator(train_ids,
                          train_path,
                          image_width     = image_width,
                          image_height    = image_height,
                          batch_size      = batch_size,
                          image_file_type = image_file_type,
                          image_file_name = image_file_name,
                          masks_file_name = masks_file_name,
                          masks_file_type = masks_file_type)

valid_gen = DataGenerator(valid_ids,
                          train_path,
                          image_width     = image_width,
                          image_height    = image_height,
                          batch_size      = batch_size,
                          image_file_type = image_file_type,
                          image_file_name = image_file_name,
                          masks_file_name = masks_file_name,
                          masks_file_type = masks_file_type)

train_steps = len(train_ids)//batch_size
valid_steps = len(valid_ids)//batch_size


############################################################################################################################################
#Using fit generator
############################################################################################################################################
model.fit_generator(train_gen,                     #train generator as an input
                    steps_per_epoch=33,            
                    epochs=epochs,
                    validation_data=valid_gen,     #validation data to cross check accuracy
                    validation_steps=valid_steps,  
                    callbacks=[tensorboard])       #Include this to parse information of the model to tensorboard

############################################################################################################################################
#Saving the model
############################################################################################################################################
version=1 #Change the file name or it will overide the model with the same name
model.save("Model/UNetW{}.h5".format(version))

######
#Uncomment to load model
#model.load_model("Model/UNetW.h5")
######

#Dataset for prediction
threshold = 0.99
x, y      = valid_gen.__getitem__(0)
result    = model.predict(x1)
result    = result > threshold


############################################################################################################################################
#Check all the validation result
############################################################################################################################################
'''
For evaluation metric, the IoU method is used to evaluate the segmentation accuracy
'''

def iou_coeff(target,prediction):
    intersection = np.logical_and(target, prediction) #Finding the areas where the areas of mask and predicted image intersect
    union        = np.logical_or(target, prediction)  
    iou_score    = np.sum(intersection) / np.sum(union)
    return iou_score

array=[]
#Displaying the a sample of predicted results with its corresponding image and mask
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
iou_score = statistics.mean(array) #Getting the average value of IoU score of the validation data set to determine the accuracy
print(iou_score)
plt.plot(array)
############################################################################################################################################
#End
############################################################################################################################################