# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 20:48:13 2017

@author: andy
"""

import os
import scipy.misc as misc
import sys
import pickle

# Get the size of all resized images. dump it in the "imageSizes.pkl"

def getsizes(img_path):
	sizes = {}
	#image_path = 'Caltech101/101_ObjectCategories/'
	subdirs = [x[0] for x in os.walk(img_path,True)]
	
	subdirs.pop(0)
	#count = 0
	#class_count = 0
	for subdir in subdirs:
		print(' ')
		sys.stdout.write("calculating " + subdir.split("/")[-1])
		s = {}
		imgs = [x[2] for x in os.walk(subdir,True)][0]
		
		#class_count +=1
		
		
		for img in imgs:
			#count += 1
			f = misc.imread(subdir+'/'+img,mode='L')
			height,width=f.shape
			ratio = 1.0*max(height,width)/300
			height = round(height/ratio)
			width = round(width/ratio)
			assert height<=301
			assert width <= 301
			s[img]=(height,width)
		sizes[subdir.split("/")[-1]]=s
	root_dir = img_path.split("/")[0]
	pickle.dump(sizes,open(root_dir+'/imageSizes.pkl','wb'))

#%%
#==============================================================================
# image_path = 'Caltech101/101_ObjectCategories/'
# subdirs = [x[0] for x in os.walk(image_path,True)]
# 
# subdirs.pop(0)
# count = 0
# class_count = 0
# for subdir in subdirs:
# 	imgs = [x[2] for x in os.walk(subdir,True)][0]
# 	
# 	class_count +=1
# 	sub_count = 0
# 	
# 	for img in imgs:
# 		count += 1
# 		sub_count += 1
# 	print(sub_count)
# 
# 
#==============================================================================
