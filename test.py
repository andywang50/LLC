# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 09:55:54 2017

@author: andy
"""

import numpy as np
from sklearn.svm import LinearSVC 
from collections import Counter
from scipy.sparse import hstack
from scipy.sparse import csr_matrix
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def testing_one(pooled_codes,label,training_num,testing_num,svm_C):
	assert training_num <= 30
	#np.random.seed(42)
	ll = label.tolist()
	z = Counter(ll)

	zkeys = z.keys()	
	
	training_set = []
	test_set = []
	
	training_label = []
	test_label = []
	
	num_classes = len(zkeys)
	
	pooled_codes = pooled_codes.tocsr()
	for i in zkeys:
		in_this_class_code = pooled_codes[:,np.where(label == i)[0]]
		_,length = in_this_class_code.shape
		arr = np.arange(0,length)
		np.random.shuffle(arr)
		training_index = arr[0:training_num]
		test_index = arr[training_num:min(training_num+testing_num,length)]
		for t in np.arange(0,training_num):
			training_label.append(i)
		for t in np.arange(training_num,min(training_num+testing_num,length)):
			test_label.append(i)
		training = in_this_class_code[:,training_index]
		test = in_this_class_code[:,test_index]
		training_set.append(training)
		test_set.append(test)
	training_set = hstack(training_set)
	test_set = hstack(test_set)
	training_set = csr_matrix.transpose(training_set.tocsr())
	test_set = csr_matrix.transpose(test_set.tocsr())
	
	#training_set_dense = training_set.todense()
	#test_set_dense = test_set.todense()
	
	sift_clf = LinearSVC(C=svm_C,multi_class='ovr')
	
	#sift_clf = SVC(C=svm_C,decision_function_shape='ovr')
	sift_clf.fit(training_set,training_label)
	
	predicted = sift_clf.predict(test_set)
	
	
	accuracy = np.zeros(num_classes)
	
	for i in zkeys:
		test_index = np.where(np.asarray(test_label)==i)[0]
		total_num = test_index.shape[0]
		predicted_in_this_class = predicted[test_index]
		correct_predictions = np.where(predicted_in_this_class == i)[0]
		correct_num = correct_predictions.shape[0]
		accuracy[i] = 1.0*correct_num/total_num
	cfm = confusion_matrix(test_label, predicted)
	cfm = cfm/np.sum(cfm,axis=1)[:,None]

	return accuracy,cfm
	
def test(pooled_codes,label,training_num,testing_num,svm_c_list,rounds):
	accuracies = {}
	cfms = {}
	for svm_c in svm_c_list:
		accuracies_c = []
		cfms_c= []
		for round_ in np.arange(0,rounds):
			accuracy,cfm = testing_one(pooled_codes,label,training_num,testing_num,svm_c)	
			accuracies_c.append(accuracy)	
			cfms_c.append(cfm)
			print('avg accuracy with c = ', svm_c,": ",np.sum(accuracy)/accuracy.shape[0])
		accuracies[svm_c] = accuracies_c
		cfms[svm_c] = cfms_c
	return accuracies,cfms
	
def print_performances_class(img_path,accuracies,svm_c):
	accuracies_c = accuracies[svm_c]
	accuracies_c = np.vstack(accuracies_c)
	avg_c = np.mean(accuracies_c,axis=0)
	
	subdirs = [x[0] for x in os.walk(img_path,True)]
	subdirs.pop(0)
	for i in range(0,len(subdirs)):
		print(subdirs[i].split("/")[-1],avg_c[i])
		
def plot_cfm(cfms,accuracies,svm_c):
	cfm_c = cfms[svm_c]
	accuracies_c = accuracies[svm_c]
	avg_acc_c = []
	for accuracy in accuracies_c:
		avg = np.sum(accuracy)/accuracy.shape[0]
		avg_acc_c.append(avg)
	avg_acc_c = np.asarray(avg_acc_c)
	best_performance_round = np.argmax(avg_acc_c)
	cfm = cfm_c[best_performance_round]
	fig = plt.figure(figsize=(7,7))
	ax = fig.add_subplot(1,1,1)
	ax.set_aspect('equal')
	plt.imshow(cfm, interpolation='nearest', cmap=plt.cm.Blues)#plt.cm.ocean
	plt.colorbar()
	plt.show()
	#plt.imsave('cfm.eps',cfm,cmap=plt.cm.Blues,format = 'eps',dpi=1000)
	fig.savefig('cfm.eps',format = 'eps',dpi=1000)

	return cfm
