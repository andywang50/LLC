# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 15:48:05 2017

@author: andy
"""
import pickle
import os
import sys
from pooling import sift_pooling
from pooling import hog_pooling
from scipy.sparse import hstack
import numpy as np

# Retrieve the unpooled codes in the code directory.
# pool them, and then return.

def retrieve_hog(code_path):
	
	# retrieve all pooled_codings:

	directory = code_path + 'hog_codes/'

	pooled_codes_list = []
	
	label_list = []
	
	label_count = 0
	
	subdirs = [x[0] for x in os.walk(directory,True)]
	subdirs.pop(0)
	for subdir in subdirs:
		print(" ")
		sys.stdout.write("pooling "+subdir)
		codes = [x[2] for x in os.walk(subdir,True)][0]
		for code in codes:
			unpooled,shape_3d = pickle.load(open(subdir+'/'+code,'rb'))
			#unpooled.shape = (1024,height*width) #因为sparse matrix存不了三维array
			#shape_3d = (dim,height,width) where  num = height*width
			pooled = hog_pooling(unpooled,shape_3d) # 21504-by-1
			label_list.append(label_count)
			pooled_codes_list.append(pooled)
		label_count+=1
	pooled_codes_list = hstack(pooled_codes_list)
	label_list = np.asarray(label_list)
	
	pickle.dump((pooled_codes_list,label_list),open(code_path+'pooled_hog.pkl','wb'))
	return pooled_codes_list,label_list
	
def retrieve_sift(code_path):
	
	root_dir = code_path.split("/")[0]

	directory = code_path + 'sift_codes/'
	
	image_sizes_dict = pickle.load(open(root_dir+'/imageSizes.pkl','rb'))

	pooled_codes_list = []
	label_list = []
	
	label_count = 0
	
	subdirs = [x[0] for x in os.walk(directory,True)]
	subdirs.pop(0)
	for subdir in subdirs:
		class_name = subdir.split("/")[-1]
		sub_dict = image_sizes_dict[class_name]
		print(" ")
		sys.stdout.write("pooling "+subdir)
		codes = [x[2] for x in os.walk(subdir,True)][0]
		for code in codes:
			img_file_name = code.split(".")[0]+".jpg"
			(height,width) = sub_dict[img_file_name]
			unpooled,coordinates = pickle.load(open(subdir+'/'+code,'rb'))
			pooled = sift_pooling(unpooled,coordinates,height,width)
			label_list.append(label_count)
			pooled_codes_list.append(pooled)
		label_count+=1
	pooled_codes_list = hstack(pooled_codes_list)
	label_list = np.asarray(label_list)

	pickle.dump((pooled_codes_list,label_list),open(code_path+'pooled_sift.pkl','wb'))

	return pooled_codes_list,label_list
