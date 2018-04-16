# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 20:56:48 2017

@author: andy
"""
import pickle
import os
from cyvlfeat.hog import hog
from cyvlfeat.sift import sift
from cyvlfeat.sift import dsift

import scipy.misc as misc
import sys
import numpy as np
import cv2

def extract_sift(img_path,data_path):	
	
	subdirs = [x[0] for x in os.walk(img_path,True)]
	
	subdirs.pop(0)
	
	for subdir in subdirs:
		print(' ')
		sys.stdout.write("extracting sift " + subdir.split("/")[-1])
		sift_data_subdir = data_path+'sift/'+subdir.split('/')[-1]+'/'
		if not os.path.exists(sift_data_subdir):
			os.makedirs(sift_data_subdir)	
		imgs = [x[2] for x in os.walk(subdir,True)][0]
		num_files = len(imgs)
		count = 0
		for img in imgs:
			if count >= round(num_files/10):
				sys.stdout.write('.')
				count = 0
			gray_f=misc.imread(subdir+'/'+img,True)
			height,width=gray_f.shape
			ratio = 1.0*max(height,width)/300
			gray_f = misc.imresize(gray_f,(round(height/ratio),round(width/ratio))).astype(float)
		#这个雪崩 别用这个	#sift_frame_and_descriptor = sift(gray_f,compute_descriptor = True,float_descriptors=True, norm_thresh=1.0)				
			#sift_frame_and_descriptor = sift(gray_f,n_octaves = 5,n_levels = 10,compute_descriptor = True,float_descriptors=True,edge_thresh = 30)
			#dense sift
			sift_frame_and_descriptor = dsift(gray_f,step =4,size = (8,8),float_descriptors = True)
			pickle.dump(sift_frame_and_descriptor,open(sift_data_subdir+img.split('.')[-2]+'.pkl','wb'))
			count += 1
			
def extract_hog(img_path,data_path,cellsize=8,hog_orientations = 9):	
	
	subdirs = [x[0] for x in os.walk(img_path,True)]
	
	subdirs.pop(0)
	
#==============================================================================
# 	cell_size = (cellsize,cellsize)
# 	
# 	block_size = (2,2)
# 	
# 	nbins=hog_orientations
#==============================================================================
	
	for subdir in subdirs:
		print(' ')
		sys.stdout.write("extracting hog " + subdir.split("/")[-1])
		hog_data_subdir = data_path+'hog/'+subdir.split('/')[-1]+'/'
		if not os.path.exists(hog_data_subdir):
			os.makedirs(hog_data_subdir)	
		imgs = [x[2] for x in os.walk(subdir,True)][0]
		num_files = len(imgs)
		count = 0
		for img in imgs:
			if count >= round(num_files/10):
				sys.stdout.write('.')
				count = 0
#==============================================================================
# 			别用rgb，雪崩！
#			RGB_f=misc.imread(subdir+'/'+img,mode='RGB')			
# 			height,width,_=RGB_f.shape
# 			ratio = 1.0*max(height,width)/300
# 			RGB_f = misc.imresize(RGB_f,(round(height/ratio),round(width/ratio)),mode='RGB').astype(float)
# 			hog_descriptor = hog(RGB_f,cellsize,n_orientations=32,variant = 'DalalTriggs')
#==============================================================================

			gray_f = misc.imread(subdir+'/'+img,True)
			gray_f = cv2.imread(subdir+'/'+img,0)
			height,width=gray_f.shape
			ratio = 1.0*max(height,width)/300	
			gray_f = misc.imresize(gray_f,(round(height/ratio),round(width/ratio))).astype(float)
			gray_f = cv2.resize(gray_f,(round(height/ratio),round(width/ratio)))
			hog_descriptor = hog(gray_f,cellsize,n_orientations=hog_orientations)			
			hog_descriptor=hog_descriptor.astype(float)
#==============================================================================
# 			hogcv = cv2.HOGDescriptor(_winSize=(gray_f.shape[1] // cell_size[1] * cell_size[1],
#                                   gray_f.shape[0] // cell_size[0] * cell_size[0]),
#                         _blockSize=(block_size[1] * cell_size[1],
#                                     block_size[0] * cell_size[0]),
#                         _blockStride=(cell_size[1], cell_size[0]),
#                         _cellSize=(cell_size[1], cell_size[0]),
#                         _nbins=nbins)
# 
# 			n_cells = (gray_f.shape[0] // cell_size[0], gray_f.shape[1] // cell_size[1])
# 			hog_descriptor = hogcv.compute(gray_f).reshape(n_cells[1] - block_size[1] + 1,n_cells[0] - block_size[0] + 1,block_size[0], block_size[1], nbins).transpose((1, 0, 2, 3, 4))
# 			hog_descriptor = hog_descriptor.reshape((hog_descriptor.shape[0],hog_descriptor.shape[1],-1)).astype(float)			
#==============================================================================
			
			pickle.dump(hog_descriptor,open(hog_data_subdir+img.split('.')[-2]+'.pkl','wb'))
			count += 1


