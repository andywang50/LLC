# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 09:15:10 2017

@author: andy
"""

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
#from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve


def testing_one_class(training_codes,test_codes,training_set,test_set,train_in_this_class,test_in_this_class,svm_C,ap_type):
#==============================================================================
# 	assert training_num <= 30
# 	#np.random.seed(42)
# 	ll = label.tolist()
# 	z = Counter(ll)
# 
# 	zkeys = z.keys()	
# 	
# 	training_set = []
# 	test_set = []
# 	
# 	training_label = []
# 	test_label = []
# 	
# 	num_classes = len(zkeys)
# 	
# 	pooled_codes = pooled_codes.tocsr()
# 	for i in zkeys:
# 		in_this_class_code = pooled_codes[:,np.where(label == i)[0]]
# 		_,length = in_this_class_code.shape
# 		arr = np.arange(0,length)
# 		np.random.shuffle(arr)
# 		training_index = arr[0:training_num]
# 		test_index = arr[training_num:min(training_num+testing_num,length)]
# 		for t in np.arange(0,training_num):
# 			training_label.append(i)
# 		for t in np.arange(training_num,min(training_num+testing_num,length)):
# 			test_label.append(i)
# 		training = in_this_class_code[:,training_index]
# 		test = in_this_class_code[:,test_index]
# 		training_set.append(training)
# 		test_set.append(test)
# 	training_set = hstack(training_set)
# 	test_set = hstack(test_set)
# 	training_set = csr_matrix.transpose(training_set.tocsr())
# 	test_set = csr_matrix.transpose(test_set.tocsr())
# 	
# 	#training_set_dense = training_set.todense()
# 	#test_set_dense = test_set.todense()
#==============================================================================
	tt_label = np.zeros(len(training_set)+len(test_set))
	tt_label[train_in_this_class] =1
	tt_label[test_in_this_class] =1
	training_label = tt_label[training_set]
	test_label = tt_label[test_set]
	
	sift_clf = LinearSVC(C=svm_C,multi_class='ovr')
	
	#sift_clf = SVC(C=svm_C,decision_function_shape='ovr')
	sift_clf.fit(training_codes,training_label)
	
	predicted = sift_clf.predict(test_codes)
	confidence = sift_clf.decision_function(test_codes)
	
	correct = np.multiply(predicted,test_label)
	correct = np.where(correct == 1)[0]
	
	accuracy = correct.shape[0]/len(test_in_this_class)
	#By sklearn:
	#ap = average_precision_score(test_label, confidence,ap_type)
	#By VOC's ap:
	precision, recall, thresholds = precision_recall_curve( test_label, confidence)
	ap = 0
	for i in np.arange(0,1.1,0.1):
		p = np.max(precision[recall>=i])
		ap += p
	ap = ap/11

	
	#cfm = confusion_matrix(test_label, predicted)
	#cfm = cfm/np.sum(cfm,axis=1)[:,None]
	return accuracy,ap
	#return accuracy,cfm
	
def test(pooled_codes,training_set,test_set,train_dict,test_dict,svm_c_list,ap_type):
	
	#decrease each index by 1.
	pooled_codes = pooled_codes.todense()	
	training_set = [x-1 for x in training_set]
	test_set = [x-1 for x in test_set]
	
	training_codes = pooled_codes[:,training_set].T
	test_codes = pooled_codes[:,test_set].T
	

		
	accuracies = {}
	#cfms = {}
	aps={}
	for svm_c in svm_c_list:
		accuracies_c = {}
		aps_c={}
		#cfms_c= []
		for label in train_dict.keys():
			train_in_this_class = train_dict[label]
			train_in_this_class = [x-1 for x in train_in_this_class]
			test_in_this_class = test_dict[label]
			test_in_this_class = [x-1 for x in test_in_this_class]
			accuracy,ap = testing_one_class(training_codes,test_codes,training_set,test_set,train_in_this_class,test_in_this_class,svm_c,ap_type)	
			accuracies_c[label] = accuracy
			aps_c[label] = ap
			#cfms_c.append(cfm)
			print('avg accuracy with c = ', svm_c,", classname: ", label, ": ",accuracy, ' ap = ', ap)
		accuracies[svm_c] = accuracies_c
		aps[svm_c] = aps_c
		#cfms[svm_c] = cfms_c
	return accuracies,aps
	
		
#==============================================================================
# def plot_cfm(cfms,accuracies,svm_c):
# 	cfm_c = cfms[svm_c]
# 	accuracies_c = accuracies[svm_c]
# 	avg_acc_c = []
# 	for accuracy in accuracies_c:
# 		avg = np.sum(accuracy)/accuracy.shape[0]
# 		avg_acc_c.append(avg)
# 	avg_acc_c = np.asarray(avg_acc_c)
# 	best_performance_round = np.argmax(avg_acc_c)
# 	cfm = cfm_c[best_performance_round]
# 	fig = plt.figure()
# 	ax = fig.add_subplot(1,1,1)
# 	ax.set_aspect('equal')
# 	plt.imshow(cfm, interpolation='nearest', cmap=plt.cm.Blues)#plt.cm.ocean
# 	plt.colorbar()
# 	plt.show()
# 	return cfm
#==============================================================================
