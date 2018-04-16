# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 11:17:31 2017

@author: andy
"""

import numpy as np
#from sklearn.preprocessing import normalize

def hog_preprocess(data_point,need_normalize=True,need_return_shape = True):
	(height,width,dim) = data_point.shape
	#shape = (height,width,dim)
	transposed = np.transpose(data_point,(2,0,1)) # -> want (dim,height,width)
	return_shape = transposed.shape	
	flattened = transposed.reshape((dim,-1)) # -> want (dim, num = height*width)
	num = height*width
	if need_normalize:
		norms = np.linalg.norm(flattened,axis=0).reshape((1,height*width))
		#flattened = flattened/flattened
		for i in np.arange(0,num):
			if norms[:,i]>1:
				flattened[:,i] = div0(flattened[:,i],norms[:,i])
#==============================================================================
# 		temp = data_point.reshape((height*width,dim))
# 		sk_normalized = normalize(temp,'l2',axis=1)
#==============================================================================
	assert flattened.shape == (dim,height*width)
	if need_return_shape:
		return (flattened,return_shape) # return_shape = (dim,height,width)
	return flattened
	
	
def div0( a, b ):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c