# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 21:41:58 2017

@author: andy
"""

import os
import sys
import pickle
import numpy as np
from scipy import sparse
from preprocess_hog_pkl import hog_preprocess
from preprocess_sift_pkl import sift_preprocess

## Given Codebook, code each descriptor.

def code_hog_data_points(data_path,output_path,knn=5,miu = 1e-4):
	B = pickle.load(open(data_path+'hog_dictionary.pkl','rb'))
	num_clusters,dim = B.shape
	assert knn < num_clusters

	output_path = output_path+"hog_codes/"
	directory = data_path+"hog/"
	BBT = np.dot(B,B.T)
	Bnorms_untiled = np.diag(BBT).reshape((1,num_clusters))
	ones = np.ones((knn,1))	
	
	subdirs = [x[0] for x in os.walk(directory,True)]
	subdirs.pop(0)
	for subdir in subdirs:
		print(" ")
		sys.stdout.write("encoding hog "+subdir)
		output_subdir = output_path+subdir.split("/")[-1]+"/"
		if not os.path.exists(output_subdir):
			os.makedirs(output_subdir)
		imgs = [x[2] for x in os.walk(subdir,True)][0]
		count = 0
		total = len(imgs)
		for img in imgs:
			if (count > total/10):
				count = 0
				sys.stdout.write(".")
			count += 1
			codes= []
			data_point = pickle.load(open(subdir+'/'+img,'rb'))
			X,shape_3d = hog_preprocess(data_point,True,True) # X.shape = (dim,height*width) shape_3d = (dim,height,width)
			assert X.shape == (dim,shape_3d[1]*shape_3d[2])
			assert shape_3d[0] == dim
			_, num_descriptors = X.shape
			XTX = np.dot(X.T,X)
			Xnorms_untiled = np.diag(XTX).reshape((num_descriptors,1))
			Xnorms_2 = np.tile(Xnorms_untiled,(1,num_clusters))
			XTBT = np.dot(X.T,B.T)
			Bnorms_2 = np.tile(Bnorms_untiled,(num_descriptors,1))
			D = Xnorms_2+Bnorms_2-2*XTBT  # Notice this is the square of the distance!
			assert D.shape == (num_descriptors,num_clusters)
			for i in np.arange(0,num_descriptors):
				Xi = X[:,i].reshape((dim,1))
				Di = D[i,:]
				idx = np.argsort(Di)[0:knn]
				Bi = B[idx,:].T
				centralized_Bi =Bi-np.dot(Xi,ones.T)
				Cov_matrix = np.dot(centralized_Bi.T,centralized_Bi)
				#di2 = np.diag(np.diag(Cov_matrix))
				#Qi = Cov_matrix+miu*di2
				Qi = Cov_matrix+miu*np.trace(Cov_matrix)*np.eye(knn)
				ci = np.linalg.solve(Qi,ones).reshape((knn,))
				ci = ci/np.sum(ci,axis=0)
				ci = ci.reshape((knn,1))
				code_i = np.zeros((num_clusters,1))
				code_i[idx] = ci
				codes.append(code_i)			
			codes = np.hstack(codes)
			codes = sparse.csr_matrix(codes)
			assert codes.shape == (num_clusters,num_descriptors)
			pickle.dump((codes,shape_3d),open(output_subdir+img,'wb'))	

def code_sift_data_points(data_path,output_path,knn=5,miu = 1e-4):
	B = pickle.load(open(data_path+'sift_dictionary.pkl','rb'))
	num_clusters,dim = B.shape
	assert knn < num_clusters

	output_path = output_path+"sift_codes/"
	directory = data_path+"sift/"
	BBT = np.dot(B,B.T)
	Bnorms_untiled = np.diag(BBT).reshape((1,num_clusters))
	ones = np.ones((knn,1))	
	
	subdirs = [x[0] for x in os.walk(directory,True)]
	subdirs.pop(0)
	for subdir in subdirs:
		if (subdir.split("/")[-1] == ".DS_Store"):
			continue
		print(" ")
		sys.stdout.write("encoding "+subdir)
		output_subdir = output_path+subdir.split("/")[-1]+"/"
		if not os.path.exists(output_subdir):
			os.makedirs(output_subdir)
		imgs = [x[2] for x in os.walk(subdir,True)][0]
		count = 0
		total = len(imgs)
		for img in imgs:
			if (count > total/10):
				count = 0
				sys.stdout.write(".")
			count += 1
			codes= []
			data_point = pickle.load(open(subdir+'/'+img,'rb'))
			X,coordinates,_ = sift_preprocess(data_point,True)
			num_descriptors, _ = X.shape
			XXT = np.dot(X,X.T)
			Xnorms_untiled = np.diag(XXT).reshape((num_descriptors,1))
			Xnorms_2 = np.tile(Xnorms_untiled,(1,num_clusters))
			XBT = np.dot(X,B.T)
			Bnorms_2 = np.tile(Bnorms_untiled,(num_descriptors,1))
			D = Xnorms_2+Bnorms_2-2*XBT  # Notice this is the square of the distance!
			for i in np.arange(0,num_descriptors):
				Xi = X[i,:].reshape((dim,1))
				Di = D[i,:]
				idx = np.argsort(Di)[0:knn]
				Bi = B[idx,:].T
				centralized_Bi =Bi-np.dot(Xi,ones.T)
				Cov_matrix = np.dot(centralized_Bi.T,centralized_Bi)
				#di2 = np.diag(np.diag(Cov_matrix))
				#Qi = Cov_matrix+miu*di2
				Qi = Cov_matrix+miu*np.trace(Cov_matrix)*np.eye(knn)
				ci = np.linalg.solve(Qi,ones).reshape((knn,))
				ci = ci/np.sum(ci,axis=0) # axis=0 is redundant but does not hurt
				code_i = np.zeros((num_clusters,))
				code_i[idx] = ci
				codes.append(code_i)			
			codes = np.vstack(codes)
			codes = sparse.csr_matrix(codes)
			pickle.dump((codes,coordinates),open(output_subdir+img,'wb'))	
