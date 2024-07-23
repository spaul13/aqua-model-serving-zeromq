# loading library 
#import cv2, sys 
import numpy as np
from PIL import Image, ImageEnhance
#import image_features as imf
#from scipy.stats import pearsonr
import torch, sys
from torchvision import datasets, transforms
import torchvision
import os, os.path, glob
#import image_preprocess as ip
import random
import math
from math import floor, ceil


extension = ".jpg"


def add_4dist(imfile,sfactor,csfactor,cfactor,bfactor,dest_fol):#factor>1--> more bright, factor<1--> more dark
	new_file_name = dest_fol + "/" + str(imfile.split("/")[-1]).split(".")[0] + "_sharp_" + str(sfactor) + "_color_"+ str(csfactor) + "_contrast_" + str(cfactor) + "_bright_" + str(bfactor) +  extension
	if(os.path.exists(new_file_name) and os.stat(new_file_name).st_size > 0):
		print("\n %s file exists" %new_file_name) 
		return new_file_name
	else:
		#read the image
		im = Image.open(imfile)
		
		sharp_enhancer = ImageEnhance.Sharpness(im)
		im_output1 = sharp_enhancer.enhance(sfactor)
		
		color_enhancer = ImageEnhance.Color(im_output1)
		im_output2 = color_enhancer.enhance(csfactor)
		
		contrast_enhancer = ImageEnhance.Contrast(im_output2)
		im_output3 = contrast_enhancer.enhance(cfactor)
		
		bright_enhancer = ImageEnhance.Brightness(im_output3)
		im_output4 = bright_enhancer.enhance(bfactor)
		im_output4.save(new_file_name)
		return new_file_name


xlist = list(range(6,16))
bright_list = [i/10 for i in xlist]

#xlist = list(range(6,37,3))
xlist = list(range(6,37,3))
contrast_list = [i/10 for i in xlist]

xlist = list(range(5,16))
sharpness_list = [i/10 for i in xlist]

#xlist = list(range(1,20,2))
xlist = list(range(1,20,2))
color_list = [i/10 for i in xlist]

imfile = sys.argv[1]
sharpness = float(sys.argv[2])
color = float(sys.argv[3])
contrast = float(sys.argv[4])
bright = float(sys.argv[5])

dest_fol = sys.argv[6]

add_4dist(imfile,sharpness,color,contrast,bright,dest_fol)
