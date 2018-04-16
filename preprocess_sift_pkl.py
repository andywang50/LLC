#-*- coding: utf-8 -*-
"""
Created on Mon Feb 20 12:01:32 2017

@author: andy
"""

import numpy as np
#from sklearn.preprocessing import normalize

# data_point is the raw data that vlfeat returns, in the form of
# (frames, descriptors).
# frames is of shape (#descriptors,4).
# The first two columns of frames is the y_coord and x_coord of the descriptors.
# The last two columns is not used.
# descriptors is of shape (#descriptors, 128)
# if need_normalize is true:
# we normalize each descriptor ONLY IF its l2 norm is larger than 1;
# if its l2 norm <=1, we don't normalize it. (keep it the same)

def sift_preprocess(data_point,need_normalize=True):
	descriptors = data_point[1].astype(float)
	frames = data_point[0]
	coordinates = frames[:,0:2]
	(num,dim) = descriptors.shape
	if need_normalize:
		#descriptors = normalize(descriptors,'l2',axis=1)
		norms = np.linalg.norm(descriptors,axis=1).reshape((num,1))
		
		# Only normalize when the norm is larger than 1
		for i in np.arange(0,num):
			if norms[i]>1:
				descriptors[i,:] = div0(descriptors[i,:],norms[i])
				
		#descriptors = div0(descriptors,norms)
#==============================================================================
# 		temp = data_point.reshape((height*width,dim))
# 		sk_normalized = normalize(temp,'l2',axis=1)
#==============================================================================

	return descriptors,coordinates,frames
	
	
def div0( a, b ):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c