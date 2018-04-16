# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 09:16:30 2017

@author: andy
"""

import pickle
import os
import numpy as np




def sample_train_test(training_label_dir,test_label_dir):
	
	test_suffix = "_test.txt"
	train_suffix = "_trainval.txt"
	
	labels = [x[2] for x in os.walk(test_label_dir,True)][0]
	num_labels = len(labels) 
	for i in range(0,num_labels):
		if (i == num_labels):break
		if(labels[i][0]=='.'):
			labels.pop(i)
			num_labels = len(labels)
			continue
		
	assert len(labels)==20
	
	training_id_dict = {}
	test_id_dict = {}
	
	for i in range(0,num_labels):
		label = labels[i]
		label = label.split(".")[0].split("_")[0]
		labels[i] = label
		train_filename = training_label_dir+label+train_suffix
		test_filename = test_label_dir + label+test_suffix
		train_in_this_class =[]
		test_in_this_class = []
		with open(train_filename) as f:
			train_content = f.readlines()
		for line in train_content:
			line = line.rstrip("\n")
			img_id,boolean = line.split()
			boolean = int(boolean)
			assert len(img_id) == 6
			assert boolean == -1 or boolean == 0 or boolean == 1
			img_id = int(img_id)
			if boolean > -1:
				train_in_this_class.append(img_id)
		training_id_dict[label] = train_in_this_class
		with open(test_filename) as f:
			test_content = f.readlines()
		for line in test_content:
			line = line.rstrip("\n")
			img_id,boolean = line.split()
			boolean = int(boolean)
			assert len(img_id) == 6
			assert boolean == -1 or boolean == 0 or boolean == 1
			img_id = int(img_id)
			if boolean > -1:
				test_in_this_class.append(img_id)	
		test_id_dict[label] = test_in_this_class
		#print(label,"  ",len(train_in_this_class),"  ",len(test_in_this_class))
	return training_id_dict,test_id_dict
		