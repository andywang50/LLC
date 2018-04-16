# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 15:03:01 2017

@author: andy
"""

import numpy as np
import os
import pickle

def helper_func(length,cellsize=8,maximgsize= 300):
	result = np.zeros(length)
	for i in range(0,length):
		result[i] = 8*i
	assert result[-1] <= 300
	return result

source_dir = 'Caltech101/rgbhog/data/hog/'
target_dir = 'Caltech101/rgbhog_sifted/data/hog/'

src_subdirs = [x[0] for x in os.walk(source_dir,True)]
src_subdirs.pop(0)
for src_subdir in src_subdirs:
	print("transforming hog "+src_subdir)
	output_subdir = target_dir+src_subdir.split("/")[-1]+"/"
	if not os.path.exists(output_subdir):
		os.makedirs(output_subdir)
	hogs = [x[2] for x in os.walk(src_subdir,True)][0]
	for hog in hogs:
		data_point = pickle.load(open(src_subdir+'/'+hog,'rb'))
		(height,width,dim) = data_point.shape
		assert max(height,width) == 38
		assert dim == 31
		sifted = np.zeros((height*width,dim))
		frames = np.zeros((height*width,2))
		h_range = helper_func(height)
		w_range = helper_func(width)
		for y in range(0,height):
			for x in range(0,width):
				sifted[y*width+x,:]=data_point[y,x,:]
				frames[y*width+x,:]=np.array([h_range[y],w_range[x]])
		pickle.dump((frames,sifted),open(output_subdir+hog,'wb'))
				
		
