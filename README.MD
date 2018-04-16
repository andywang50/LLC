
Caltech 101:

	Data Download: http://www.vision.caltech.edu/Image_Datasets/Caltech101/

	Needed Python Files: 
		Driver: driver101.py; 
		SIFT: getsizes.py; learn_sift_dictionary.py; preprocess_sift_pkl.py. 
		HOG:  learn_hog_dictionary.py; preprocess_hog_pkl.py.
		Both: encoding.py; extract_low_level.py; pooling.py; retrieve_pooled_codings.py; sample_train_test.py; test.py.
						 
	Directory: 
		img_path = 'Caltech101/101_ObjectCategories/' #image files	
		data_path = 'Caltech101/data/' # The extracting low features will be in this folder.
		code_path = 'Caltech101/code/' # The LLC Coding will be in this folder.
	
	Parameters:  
	'sift_dictionary = get_sift_dictionary(data_path+"sift/",num_cluster = 1024, data_per_class=5)'
		num_cluster: size of codebook; data_per_class: sample how many images per class.
	'code_sift_data_points(data_path,code_path,knn=5)'
		knn: size of local base
	Training number, testing number, number of rounds, svm_c for linear svm are below "##testing part".


	
	
VOC2007:

	Data Download: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/.
	Training: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
	Testing: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

	Needed Python Files: Almost the same as Caltech101, but the driver should be driver_voc.py, and test.py should be replaced by test_voc.py.
	
	Directory:  
		img_path = 'voc2007/image/' # Put all training and testing images into one folder, name this folder whatever name (e.g. ‘all’). Then put this folder under 'voc2007/image/'.
		data_path = 'voc2007/data/' # The extracting low features will be in this folder.
		code_path = 'voc2007/code/' # The LLC Coding will be in this folder.
		training_label_dir = 'voc2007/train_class_label/' # From the training data set you downloaded, find /VOCdevkit/VOC2007/ImageSets/Main/. Put all the xxx_trainval.txt files (except for trainval.txt itself) in this folder into 'voc2007/train_class_label/'. 
		test_label_dir = 'voc2007/test_class_label/' # Do the same for test data set. Copy all xxx_test.txt files, except for test.txt.
		training_set_txt = 'voc2007/trainval.txt' # Put the trainval.txt you found (two steps before) directly into voc2007/.
		test_set_txt = 'voc2007/test.txt' # Do the same for test.txt
	
	Parameters: same as Caltech-101. except for: in the dictionary-learning step, we are no longer sampling 5 images from each class, but sampling 2000 images from the whole.
				
	
