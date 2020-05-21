import cv2
import numpy as np
import glob
import os

############################################################################################################################################
#Function to remove images with no masks
############################################################################################################################################
def trim_images(image_path,mask_path,image_type,mask_type):
    for file in glob.glob(image_path+"/*"+image_type):
        base_name=os.path.basename(os.path.normpath(file))[:-4]

        if os.path.exists(mask_path + '/' + base_name+"_label"+ mask_type):
            print("A mask exist for this image")
        else:
            os.remove(image_path + '/' + base_name+ image_type)
            print("No mask exist for this image, proceeding with deletion of image")
            

############################################################################################################################################
#Parameters
############################################################################################################################################
image_path = r"D:\Project\FYP\Data For Gavin\Inserts2\Images" #Image directory
mask_path  = r"D:\Project\FYP\Data For Gavin\Inserts2\Masks"  #Mask diretory
image_type = ".bmp"                                           #Image type for image dataset
mask_type  = ".png"                                           #Mask type for mask dataset 
############################################################################################################################################
#Activation
############################################################################################################################################
trim_images(image_path,mask_path,image_type,mask_type)