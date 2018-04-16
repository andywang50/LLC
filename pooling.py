# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 19:00:29 2017

@author: andy
"""

import numpy as np
import pickle
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import scipy.misc as misc
import math



# Given an the coding of all descriptors in an image, get the pooled code.
# For hog, each descriptor is of dim 31.
# For an image, descriptors are stored in a (1024,H*W) sparse matrix. In other words, there are H*W descriptors in the image.
# in hog_pooling, shape_3d is (31,H,W)

# For sift, it is different. the vlfeat returns the descriptors each of size 128,
# together with the y,x coordinates of each descriptor (in the unit of pixels).
# This information is passed as the coordinates variable into the sift_pooling() function.
# Also, the size of the resized image (x,300) or (300,x) where x<=300, also needs to be passed in to the pooling function for sift.

def hog_pooling(code,shape_3d,pyramid=[1,2,4]):
	(dim31,height,width) = shape_3d #dim31 = 31 用不到了应该
	(num_clusters,num_descriptors) = code.shape 
	assert num_descriptors == height*width
	
	unpooled = np.asarray(code.todense())
		
	unpooled = unpooled.reshape(num_clusters,height,width)
	
	pool_dim = sum([i*i for i in pyramid]) # = 21
	
	pooling_result = np.zeros((num_clusters,pool_dim)) #(1024 by 21)
	
	counter = 0
	
	for grid_num in pyramid:
		
		height_partitions = partition(np.arange(0,height),grid_num)
		
		width_partitions = partition(np.arange(0,width),grid_num)
		
		for height_part in height_partitions:
			for width_part in width_partitions:
				#sub_height = height_part.shape[0]
				#sub_width = width_part.shape[0]
				sub_matrix = unpooled[:,height_part,:][:,:,width_part]
				sub_matrix= sub_matrix.reshape((num_clusters,-1))
				# max pooling:
				pooled = np.max(sub_matrix,axis=1)					
				# sum pooling:
				# pooled = np.sum(sub_matrix,axis=1)
				
				pooling_result[:,counter] = pooled
				counter += 1
		
		
		
	assert counter == pool_dim
	pooling_result = np.ravel(pooling_result,'F').reshape(-1,1)
	# l2 normalize
	pooling_result = normalize(pooling_result,norm='l2')
	
	# sum normalize
	# pooling_sum = np.sum(pooling_result)
	# pooling_result = div0(pooling_result,pooling_sum)
	
	pooling_result = csr_matrix(pooling_result)
	
	return pooling_result
	
def sift_pooling(code,coordinates,height,width,pyramid=[1,2,4]):
	(num_descriptors,dim) = code.shape
	
	pool_dim = sum([i*i for i in pyramid]) # tBins
	
	pooling_result = np.zeros((dim,pool_dim))
	counter = 0
	
	
	for grid_num in pyramid:
		bin_label = (-1)* np.ones(num_descriptors)
		y_step = height/grid_num
		x_step = width/grid_num
		y_coords = coordinates[:,0]#+0.01
		x_coords = coordinates[:,1]#+0.01
		
				
		yBin = np.ceil(y_coords/y_step)
		xBin = np.ceil(x_coords/x_step)
		bin_label =  (yBin - 1)*grid_num + xBin;
#==============================================================================
# 		for i in range(0,num_descriptors):
# 			y_coord,x_coord = coordinates[i]
# 			#current_code = code[i]
# 			current_label = math.ceil(y_coord/y_step-1)*grid_num+math.ceil(x_coord/x_step)
# 			bin_label[i] = current_label
#==============================================================================
		assert np.all(bin_label>=1)
		assert np.all(bin_label<=grid_num*grid_num)
		#assert np.max(bin_label)-np.min(bin_label)<grid_num*grid_num
		for label in np.arange(1,grid_num*grid_num+1):
			sidxBin = np.where(bin_label==label)[0]
			if sidxBin.shape[0] == 0:
				counter += 1
				continue
			sub_matrix = code[sidxBin,:]
			# max pooling:
			pooled = np.asarray(sub_matrix.max(axis=0).todense())	
			assert max(pooled.shape) == dim and min(pooled.shape) == 1
			# sum pooling:
			# pooled = np.sum(sub_matrix,axis=0)
			pooling_result[:,counter] = pooled

			counter += 1
		
	assert counter == pool_dim
	
	pooling_result = np.ravel(pooling_result,'F').reshape(-1,1)

	
	# l2 normalize
	pooling_result = normalize(pooling_result,norm='l2',axis=0)
	
	pooling_result = csr_matrix(pooling_result)

	
	# sum normalize
	# pooling_sum = np.sum(pooling_result,axis=0)
	# pooling_result = div0(pooling_result,column_Sum)
	
	
	return pooling_result


def partition(arr, n):
    division = arr.shape[0] / n
    return [arr[round(division * i):round(division * (i + 1))] for i in range(n)]

def div0( a, b ):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c				
	

#==============================================================================
# img=misc.imread(image_path+'accordion/image_0001.jpg',mode='L') 
# height,width=img.shape
# 			
# result_hog = hog_pooling(test_hog_code,shape)
# result_shift = sift_pooling(test_sift_code,coordinates,height,width)
#==============================================================================
#%%
#==============================================================================
# test = np.random.random((5,4,3))
# print(test)
# 
# test_height_part1 = np.array([0,1,2])
# test_width_part1 = np.array([0,1])
# 
# test_height_part2 = np.array([3,4])
# test_width_part2 = np.array([2,3])
# 
# left_upper = test[test_height_part1,:,:][:,test_width_part1,:]
# right_bot = test[test_height_part2,:,:][:,test_width_part2,:]
# 
#==============================================================================

#hog = pickle.load(open('Caltech101/data/hog/accordion/image_0001.pkl','rb'))
#%%
