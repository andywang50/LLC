# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 12:00:52 2017

@author: andy
"""

from cyvlfeat import kmeans
import os
import pickle
from numpy import random
import numpy as np
from preprocess_sift_pkl import sift_preprocess
from cyvlfeat.kmeans import kmeans


# From each class, sample 5 images. Use these images to get a codebook using KMeans.

def get_sift_dictionary(data_path,num_cluster = 1024, data_per_class=5,tol=1e-4):
	assert data_path.split("/")[-2] == 'sift'
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
			
			descriptors,coordinates,_ = sift_preprocess(data_point,True)
			shuffle_id = np.arange(0,descriptors.shape[0])
			np.random.shuffle(shuffle_id)
			descriptors = descriptors[shuffle_id[0:int(0.2*shuffle_id.shape[0])],:]
			data_points.append(descriptors)
			
			
	data_points= np.vstack(data_points)
	data_points = data_points[~np.all(data_points == 0, axis=1)]
	return data_points
	