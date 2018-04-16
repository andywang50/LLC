# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 19:35:32 2017

@author: Xi Lin, Zhengqi Lin, Guoan Wang
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
from test_voc import test
from sample_train_test import sample_train_test


img_path = 'voc2007/image/' #image files
	
data_path = 'voc2007/data/' # low_level_features

code_path = 'voc2007/code/' # coded (unpooled yet)

training_label_dir = 'voc2007/train_class_label/'

test_label_dir = 'voc2007/test_class_label/'

training_set_txt = 'voc2007/trainval.txt'

test_set_txt = 'voc2007/test.txt'

def get_train_test_set(training_set_txt,test_set_txt):
	training_set= []
	test_set = []
	with open(training_set_txt) as f:
		train_content = f.readlines()
	for line in train_content:
		line = line.rstrip("\n")
		assert len(line) == 6
		img_id = int(line)
		training_set.append(img_id)
	with open(test_set_txt) as f:
		text_content = f.readlines()
	for line in text_content:
		line = line.rstrip("\n")
		assert len(line) == 6
		img_id = int(line)
		test_set.append(img_id)
	return training_set,test_set
	

## This is for sift
getsizes(img_path)
extract_sift(img_path,data_path)
sift_dictionary = get_sift_dictionary(data_path+"sift/",num_cluster = 4096, data_per_class=2000)
pickle.dump(sift_dictionary,open(data_path+'sift_dictionary.pkl','wb'))
code_sift_data_points(data_path,code_path,knn=5)
pooled_sift,label_sift = retrieve_sift(code_path)

## This is for hog
#==============================================================================
# hog_cell_size = 8
# extract_hog(img_path,data_path,hog_cell_size)
# hog_dictionary = get_hog_dictionary(data_path+"hog/",num_cluster = 1024, data_per_class=500)
# pickle.dump(hog_dictionary,open(data_path+'hog_dictionary.pkl','wb'))
# code_hog_data_points(data_path,code_path) # Example to read: (height,width),new_code = pickle.load(open('Caltech101/code/hog_codes/accordion/image_0001.pkl','rb'))
# pooled_hog,label_hog = retrieve_hog(code_path)
#==============================================================================


## testing part voc

print(" ")
svm_c_list = [30,10,3,1,0.3]
#svm_c_list = [0.7,0.3]
#svm_c_list = [0.9,0.8,0.7,0.6,0.5,0.4,0.3]
training_set,test_set = get_train_test_set(training_set_txt,test_set_txt)

train_dict,test_dict = sample_train_test(training_label_dir,test_label_dir)

accuracies,aps = test(pooled_sift,training_set,test_set,train_dict,test_dict,svm_c_list,'macro')

# AVG ACCURACY of same c
# print("average accuacy:") # Accuracy not used, AP used instead.
# for c in svm_c_list:
#    accs = accuracies[c]
#    avgs = []
#    for keys,acc in accs.items():
#        avgs.append(acc)
#    avgs=np.asarray(avgs)

#    assert avgs.shape[0] == 20
#    print(c, " ", np.sum(avgs)/avgs.shape[0])

print("average ap:")
for c in svm_c_list:
    aps_c = aps[c]
    avgs = []
    for keys,ap in aps_c.items():
        avgs.append(ap)
    avgs=np.asarray(avgs)

    assert avgs.shape[0] == 20
    print(c, " ", np.sum(avgs)/avgs.shape[0])
								
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
