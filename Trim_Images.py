import cv2
import numpy as np
import glob
import os

##################################################################
#Function to remove white images and corresponding masks
##################################################################
def trim_images(image_path,mask_path,image_type,mask_type):
    for file in glob.glob(image_path+"/*"+image_type):
        
        #Get image name without ".png"
        base_name=os.path.basename(os.path.normpath(file))[:-4]
        
        #Mask name
        mask_name=base_name[:-5]+"label"+base_name[-6:]
        #Read image
        img=cv2.imread(file,-1)
        
        #Dynamo white areas are =255 but insert white areas are >=253
        #To check if the image is white, get the minimum value of the whole array 
        if np.amin(img) >= 253:
            os.remove(image_path + '/' + base_name+ image_type)
            try:
                os.remove(mask_path + "/" + mask_name +  mask_type)
            except Exception:
                pass
            print('All white')
        else:
            print('Not all white')

#################################
#Parameters
#################################
image_path = r"D:\Project\FYP\Data For Gavin\Inserts\Images-split" #Image directory
mask_path  = r"D:\Project\FYP\Data For Gavin\Inserts\Masks-split"  #Mask diretory
image_type = ".bmp"                                                #Image type for image dataset 
mask_type  = ".png"                                                #Mask type for mask dataset
#################################
#Activation
#################################
trim_images(image_path,mask_path,image_type,mask_type)














