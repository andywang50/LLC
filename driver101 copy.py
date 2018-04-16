# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 19:35:32 2017

@author: Xi Lin, Zhengqi Lin, Guoan Wang (andy)
"""

#from scipy.io import loadmat
import pickle
import numpy as np
import os

from extract_low_level import extract_sift
from extract_low_level import extract_hog
from learn_hog_dictionary import get_hog_dictionary
from learn_sift_dictionary import get_sift_dictionary
from encoding import code_hog_data_points
from encoding import code_sift_data_points
from getsizes import getsizes
from retrieve_pooled_codings import retrieve_hog
from retrieve_pooled_codings import retrieve_sift
from test import test
from test import print_performances_class
from test import plot_cfm
from datetime import datetime

# Ignore This
#==============================================================================
# dictionary_path = 'Caltech101_SIFT_Kmeans_1024.mat'
# 
# sift_codebook = loadmat(dictionary_path)['B']
# 
# (dim_descriptor,num_centroid) = sift_codebook.shape
# 
#==============================================================================

img_path = 'Caltech101/101_ObjectCategories/' #image files	

data_path = 'Caltech101/hogcv/data/' # low_level_features

code_path = 'Caltech101/hogcv/code/' # coded (unpooled yet)



## This is for sift
getsizes(img_path)
extract_sift(img_path,data_path)
sift_dictionary = get_sift_dictionary(data_path+"sift/",num_cluster = 1024, data_per_class=5)
pickle.dump(sift_dictionary,open(data_path+'sift_dictionary.pkl','wb'))
code_sift_data_points(data_path,code_path,knn=5)
pooled_sift,label_sift = retrieve_sift(code_path)

## This is for hog
#hog_cell_size = 8
#extract_hog(img_path,data_path,hog_cell_size)
#hog_dictionary = get_hog_dictionary(data_path+"hog/",num_cluster = 1024, data_per_class=5)
#pickle.dump(hog_dictionary,open(data_path+'hog_dictionary.pkl','wb'))
#code_hog_data_points(data_path,code_path,knn=5) # Example to read: (height,width),new_code = pickle.load(open('Caltech101/code/hog_codes/accordion/image_0001.pkl','rb'))
#pooled_hog,label_hog = retrieve_hog(code_path)


## testing part
training_num = 30
testing_num= 50
num_rounds = 5
svm_c_list = [30,10,3,1]
accuracies,cfms = test(pooled_sift,label_sift,training_num,testing_num,svm_c_list,num_rounds) #sift
#accuracies,cfms = test(pooled_hog,label_hog,training_num,testing_num,svm_c_list,num_rounds) #hog
print_performances_class(img_path,accuracies,svm_c=1)
cfm_1 = plot_cfm(cfms,accuracies,1)
#print(str(datetime.now()))

#%%
# AVG ACCURACY of same c
#==============================================================================
# 
# for c in svm_c_list:
#     accs = accuracies[c]
#     avgs = []
#     for acc in accs:
#         avg = np.sum(acc)/acc.shape[0]
#         avgs.append(avg)
#     avgs=np.asarray(avgs)
#     print(c, " ", np.sum(avgs)/avgs.shape[0])
# 				
#==============================================================================
#%%	
#==============================================================================
# import sys
# subdirs = [x[0].split("/")[-1] for x in os.walk(img_path,True)]
# 
# subdirs.pop(0)
# 
# cfm = cfm_10
# acc = np.diag(cfm)
# bad_to_good = np.argsort(acc)
# # #7 = beaver, acc = 18.75
# #beaver = cfm[7,:]
# #beaver_labels = np.argsort(beaver)[::-1] # 多到少
# for i in range(0,20):
# 	bad_class_id = bad_to_good[i]
# 	bad_class_name = subdirs[bad_class_id]
# 	print(bad_class_name,", overall acc: ",acc[bad_class_id])
# 	bad_class = cfm[bad_class_id]
# 	bad_class_predicts = np.argsort(bad_class)[::-1]
# 	for j in range(0,6):
# 		mistaken_class_id = bad_class_predicts[j]
# 		mistaken_prob = bad_class[mistaken_class_id]
# 		print(subdirs[mistaken_class_id]," ",mistaken_prob)
# 	print("\n\n")
# 	
#==============================================================================
