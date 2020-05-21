##########################################
#Preprocessing of images
##########################################
import image_slicer
import glob
import cv2
import os
import math

##########################################
#Resizing the image
##########################################
'''
For images, aim to have the dimensions closest or more than the original image size. This prevents any
loss of information by compression.
'''
##########################################
#Parameters
dimensions   = (8192,8192)
desired_size = 256
##########################################
#Images directory
##########################################
image_path          = r"D:\Project\FYP\Data For Gavin\Inserts2\Images"
resized_path        = r"D:\Project\FYP\Data For Gavin\Inserts2\Images-resized"
sliced_path         = r"D:\Project\FYP\Data For Gavin\Inserts2\Images-split"
image_type          = ".bmp"

##########################################
#Masks directory
##########################################
mask_path           = r"D:\Project\FYP\Data For Gavin\Inserts2\Masks"
mask_resized_path   = r"D:\Project\FYP\Data For Gavin\Inserts2\Masks-resized"
mask_sliced_path    = r"D:\Project\FYP\Data For Gavin\Inserts2\Masks-split"
mask_type           = ".png"

##############################################
#Function
##############################################

def slicing_of_image(dimensions,desired_size,image_path,resized_path,sliced_path,image_type):
    #Calculation to equally slice up the image to the desired size
    number=dimensions[0]/desired_size
    x=math.log(number,2)
    
    if dimensions[0]==desired_size:
        split=2
    else:
        split=2**(x*2)

    for file in glob.glob(image_path+"/*"+image_type):   
        imageName = file.split('\\')[6][0:-4]#Change the values in regards to the position of the directory
        img=cv2.imread(file,-1)
        img=cv2.resize(img,dimensions,interpolation=cv2.INTER_AREA)
        os.chdir(resized_path)
        cv2.imwrite(imageName+image_type,img)
        
    for file in glob.glob(resized_path+"/*"+image_type):   
        imageName = file.split('\\')[6][0:-4]#Change the values in regards to the position of the directory
        tiles = image_slicer.slice(file, split  , save=False)
        image_slicer.save_tiles(tiles, directory=sliced_path, prefix=imageName, format=image_type[1:])
     
#################################################
#Activation
#################################################        
slicing_of_image(dimensions,desired_size,image_path,resized_path,sliced_path,image_type)
slicing_of_image(dimensions,desired_size,mask_path,mask_resized_path,mask_sliced_path,mask_type)
