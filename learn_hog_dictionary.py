# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 10:15:37 2017

@author: andy
"""

from cyvlfeat import kmeans
import os
import pickle
from numpy import random
import numpy as np
from preprocess_hog_pkl import hog_preprocess
from cyvlfeat.kmeans import kmeans


def get_hog_dictionary(data_path,num_cluster = 1024, data_per_class=5):
	assert data_path.split("/")[-2] == 'hog'
	data_points = sample(data_path,data_per_class)
	centroids = kmeans(data_points,num_cluster,algorithm="ANN",max_num_comparisons=256,verbose=True)
	return centroids
	
	
def sample(data_path,data_per_class = 5):
			
	
	#random.seed(1)
	
	subdirs = [x[0] for x in os.walk(data_path,True)]
		
	subdirs.pop(0)
	
	data_points = []
	
	for subdir in subdirs:
		imgs = [x[2] for x in os.walk(subdir,True)][0]
		num_files = len(imgs)
		idx = np.arange(0,num_files)
		random.shuffle(idx)
		idx = idx[:data_per_class]
		for i in idx:
			img = imgs[i]
			data_point = pickle.load(open(subdir+'/'+img,'rb'))
			data_point = hog_preprocess(data_point,need_normalize=True, need_return_shape = False)
			data_points.append(data_point)
			
			
	data_points= np.hstack(data_points)
	data_points = data_points.T
	data_points = data_points[~np.all(data_points == 0, axis=1)]
	return data_points
	