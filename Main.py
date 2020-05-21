##########################################################################################################################################
#Main: use this code segment defects from the desired image
##########################################################################################################################################
import tensorflow as     tf
import image_slicer
import math
import cv2
import h5py
import glob
import os
import numpy      as     np
from   PIL        import Image
from   tensorflow import keras
from keras.models import load_model,model_from_json

############################################################################################################################################
#Parameters
############################################################################################################################################
image_path          = r"D:\Project\FYP\Data For Gavin\Inserts\Images\Chipping_033_100ms" #Direct it to the exact image
resized_path        = r"D:\Project\FYP\Data For Gavin\Inserts\Test_image_resize"       
sliced_path         = r'D:\Project\FYP\Data For Gavin\Inserts\Test_image_split'
predicted_path      = r"D:\Project\FYP\Data For Gavin\Inserts\Results"
recombined_path     = r"D:\Project\FYP\Data For Gavin\Inserts\Test_image_recombined"     
image_type          = ".bmp"
dimensions          = (8192,8192)
desired_size        = 256
############################################################################################################################################
def preparation_of_image(dimensions,desired_size,image_path,resized_path,sliced_path,image_type,predicted_path,recombined_path,threshold):
    segmentation_model = tf.keras.models.load_model("Model/UNetW.h5")                              #Loading UNet Model
    #To determine how many portions to split to achieve desired size
    number             = dimensions[0]/desired_size
    x                  = math.log(number,2)
    if dimensions[0]  ==desired_size:
        split=2
    else:
        split=2**(x*2)
            
    imageName = os.path.basename(os.path.normpath(image_path))[:-4]                                #Getting the test image name from the test image path 
    img       = cv2.imread(image_path+image_type,cv2.IMREAD_GRAYSCALE)                             #Reading the test image in grayscale
    img       = cv2.resize(img,dimensions,interpolation=cv2.INTER_AREA)                            #Resizing the test image into the correct size 
    os.chdir(resized_path)                                                                         #Changing the directory for the test image to be saved
    cv2.imwrite(imageName+image_type,img)                                                          #Saving the test image
    tiles     = image_slicer.slice((resized_path+"/" +imageName+image_type), split , save=False)   #Slicing the test image up 
    image_slicer.save_tiles(tiles, directory=sliced_path, prefix=imageName, format=image_type[1:]) #Saving the sliced test images into the next folder
       

    for file in glob.glob(sliced_path+"/*"+image_type):
        
        img       = cv2.imread(file,cv2.IMREAD_GRAYSCALE)                                          #Reading the sliced image into an array
        base_name = os.path.basename(os.path.normpath(file))                                       #Getting image names 
        img       = np.expand_dims(img,-1)                                                         #Expanding dimensions
        img       = np.expand_dims(img,0)                                                          #Expanding dimensions
        img       = img/255.0                                                                      #Normalising values
                    
        predicted_image = segmentation_model.predict(img)                                          #Predicting the image
        predicted_image = predicted_image[0,:, :, 0]                                               #Removing the dimensions to (256,256)
        predicted_image = predicted_image> threshold                                               #Setting the threshold to filter out the noise ~ recommended is 0.99        
        predicted_image=predicted_image*255                                                        #Reverse normalising to get exact shades of colour        
        os.chdir(predicted_path)                                                                   #Changing directory for it to be saved
        cv2.imwrite(base_name,predicted_image)                                                     #Saving predicted image

    newtiles = image_slicer.open_images_in('D:/Project/FYP/Data For Gavin/Inserts/Results/.' )     #Opening image in tiled predicted images  
    imageAll = image_slicer.join(newtiles)                                                         #Joining the images together
    imageAll.save('D:/Project/FYP/Data For Gavin/Inserts/Results/' + imageName + '.png')           #Saving the combined image together
    
    return imageName

############################################################################################################################################
#Activation
############################################################################################################################################
preparation_of_image(dimensions,desired_size,image_path,resized_path,sliced_path,image_type,predicted_path,recombined_path,0.99)
