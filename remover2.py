import cv2
import numpy as np
import glob
import os

############################################################################################################################################
#Function to remove all fully white mask and corresponding image
############################################################################################################################################
def trim_images(image_path,mask_path,image_type,mask_type):
    for file in glob.glob(mask_path+"/*"+mask_type):
        
        #Get image name without ".png"
        base_name=os.path.basename(os.path.normpath(file))[:-4]
        #Mask name
        image_name=base_name[:-12]+base_name[-6:]
        #Read image
        mask=cv2.imread(file,-1)
        
        #Dynamo white areas are =255 but insert white areas are >=253
        #To check if the image is white, get the minimum value of the whole array 
        if np.amin(mask) >= 253:
           os.remove(mask_path + '/' + base_name+ mask_type)
           os.remove(image_path+"/"+image_name+image_type)

############################################################################################################################################
#Parameters
############################################################################################################################################
image_path = r"D:\Project\FYP\Data For Gavin\Inserts3\Images-split"    #Image directory
mask_path  = r"D:\Project\FYP\Data For Gavin\Inserts3\Masks-split"     #Mask diretory
image_type = ".bmp"                                                    #Image type for image dataset
mask_type  = ".png"                                                    #Mask type for mask dataset 
#################################
#Activation
#################################
trim_images(image_path,mask_path,image_type,mask_type)
