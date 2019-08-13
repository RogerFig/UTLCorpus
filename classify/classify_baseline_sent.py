'''
	sample 
	python3 classify.py -classifier lr -train_file data/corpus/trainTT -test_file data/corpus/testTT -v
'''

import os, sys
import numpy as np
import nlpnet
from random import shuffle as shuffle_list

from utils import vectorize
from model import get_model
from features import FeatureExtractor

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

from scipy import sparse

CLASSES = ['neg', 'neu', 'pos']
NEG, NEU, POS = 0, 1, 2 
THRESHOLD = 0

def help(*args):
	if args:
		print("-----------------------------------------------")
		print('error: ' + args[0])
		print('use -help for more information.')
		exit()

	print("-----------------------------------------------")
	print("Using sent_analyzer:\n")
	print("python3 sent_analyzer.py -classifier <CLASSIFIER> -options <OPTIONS> -emb <EMBEDDING> -md_options <MD_OPTIONS> -train_file <TRAIN_FILE> -test_file <TEST_FILE> -nb_classes <NUM> <MORE...>")
	print()
	print("\n-classifier <CLASSIFIER> .... which classifier is going to be used.")
	print(" * 'LinearSVM': SVM with a Linear kernel. - default - ")
	print(" * 'NB': Gaussian Naive Bayes classifier.")
	print(" * 'LR': Logistic Regression classifier.")
	print(" * 'MLP': Multi-layer Perceptron classifier.")
	print(" * 'TREES': Decision Tree classifier.")
	print(" * 'RANDFOR': Random Forest classifier.")
	print("\n-options <OPYIOND> .... how the data will be represented. Use the following options divided by ','s . Deafult options is everything.")
	print(" * 'bow': Bag-of-words.")
	print(" * 'negation': Negation words in the sentence.")
	print(" * 'emoticon': Emoticons for positive and negative emotions.")
	print(" * 'emoji': Emojis with pos/neg/neu prob values.")
	print(" * 'senti_words': Count of positive and negative words in sentence.")
	print(" * 'postag': Number of verbs, adjectives, nouns and adverbs in sentence.")
	print("\n-emb <EMBEDDING> .... use embeddings for representation. It overwrites '-option' flag.")
	print("\n-md_options <MD-OPTIONS> .... If you'd like to change the hyperparameters of the classifier.")
	print(" * Each classifier has its own parematers, check 'model.py' for further info.")
	print("\n-train_file <TRAIN_FILE> .... train file path.")
	print("\n-test_file <TEST_FILE> .... test file path.")
	print(" * Use 'none' for no test file.")
	print("\n-nb_classes <NUM> .... number of classes (2/pos,neg | 3/pos,neu,neg).")
	print("\n")
	print("<MORE...> - More options.\n")
	print(" * '-fold': 10-fold cross validation on train file.")
	print(" * '-fs': feature selection on representation.")
	print(" * '-balance': resample the train set using downsample in order to avoid data skewing.")
	print(" * '-verbose or -v': prints it all.")
	print("\n Thanks for using. Hope you can be satisfied to the results you may obtain.")
	exit()

''' load_data <file_name, shuffle=True>
	Receives the path to a file, reads it and returns a list with three lists.
	Input:
	 - file_name <str> : path to file.
	 - shuffle <bool> : shuffle or not the data.
	Output:
	 - list[sentences, labels, ids]
	   - sentences: the sentences of the data.
	   - labels   : the labels (0,1,2).
	   - ids      : the ids for each tweet.
'''

def load_data(file_name, shuffle=True, rmttsbr=False, verbose=False, balance=False):
	global CLASSES
	file_sentences = []
	file_labels = []
	file_ids = []
	corpus_removing = []
	
	minority_class_size = sys.maxsize
		
	if rmttsbr:
		if verbose: print('Removing train docs from TTsBR.')
		with open('data/corpus/tweetSentimentBR.txt','r') as corpus_ttsbr:
			for line in corpus_ttsbr:
				corpus_removing.append(int(line[:18].strip()))
				#print(corpus_removing[-1])

	for label, c in enumerate(CLASSES):
		if verbose: print('reading ' + file_name+'.'+c)
		with open('%s.%s' % (file_name, c), 'r', encoding='utf8') as f:
			cl_len = 0
			for line in f:
				sentence = line[19:].strip()
				if len(sentence) > 1:
					if int(line[:18].strip()) in corpus_removing:
						continue
					file_sentences.append(sentence)
					file_ids.append(int(line[:18].strip()))
					#file_sentences.append(data[1:])
					file_labels.append(label)
					#file_ids.append(int(data[0]))
					cl_len += 1
				else:
					if verbose: print('One empty line.')
			if minority_class_size > cl_len: minority_class_size = cl_len
	
	file_data = [file_sentences, file_labels, file_ids]
	if shuffle:
		file_indexes = list(range(len(file_sentences)))
		shuffle_data = lambda v, ind: [v[i] for i in ind]
		shuffle_list(file_indexes)
		for i in range(len(file_data)):
			file_data[i] = shuffle_data(file_data[i], file_indexes)

	if balance:
		new_file_data = [[],[],[]]
		cl_count = [0]*len(CLASSES)
		for i, y in enumerate(file_data[1]):
			if cl_count[y] < minority_class_size:
				new_file_data[0].append(file_data[0][i])
				new_file_data[1].append(file_data[1][i])
				new_file_data[2].append(file_data[2][i])
				cl_count[y] += 1
		if verbose: print('Done reading. Data balanced.\n')
		return new_file_data

	if verbose: print('Done reading.\n')
	return file_data

''' load_unlabeled <file_name, shuffle=True>
	Receives the path to a file, reads it and returns a list with three lists.
	Input:
	 - file_name <str> : path to file.
	 - shuffle <bool> : shuffle or not the data.
	Output:
	 - list[sentences, labels, ids]
	   - sentences: the sentences of the data.
	   - labels   : the labels (0,1,2).
	   - ids      : the ids for each tweet.
'''

def run(classifier='linearsvm', options='senti_words', verbose=False, train_file='data/corpus/trainTT',
	    test_file='data/corpus/testTT', fs=False, fold=True, nb_classes=3, md_options='', embedding_file=False, rmttsbr=False, force_balance=False):
	# PARAMS
	fold = fold
	fs = fs
	train_file = train_file
	test_file = test_file
	#embedding_file = 'data/embeddings/pt_word2vec_cbow_50.emb'
	nb_classes = nb_classes
	md_option = md_options

	if nb_classes == 2:
		global CLASSES
		global POS 
		global THRESHOLD

		CLASSES = ['neg', 'pos']
		POS = 1

	f_bow = False
	f_negation = False
	f_emoticon = False
	f_emoji = False
	f_senti_words = False
	f_postag = False

	for op in options:
		if 'bow' in options:
			f_bow = True
		if 'negation' in options:
			f_negation = True
		if 'emoticon' in options:
			f_emoticon = True
		if 'emoji' in options:
			f_emoji = True
		if 'senti_words' in options:
			f_senti_words = True
		if 'postag' in options:
			f_postag = True

	# LOAD DATA
	train_data = load_data(train_file, rmttsbr=rmttsbr, shuffle=True, verbose=False,balance=force_balance)
	train_sentences, train_labels, train_ids = train_data
	if test_file != None:
		test_data  = load_data(test_file, shuffle=True,verbose=verbose)
		test_sentences, test_labels, test_ids = test_data

	# INITIALIZE FEATURE EXTRACTOR PARAMS 
	
	if embedding_file != False:
		'''embeddings'''
		feats = FeatureExtractor(emb='Word2Vec', embedding_file=embedding_file,verbose=verbose)
	else:
		'''representation'''
		feats = FeatureExtractor(bow=f_bow, negation=f_negation, emoticon=f_emoticon, emoji=f_emoji, senti_words=f_senti_words,postag=f_postag,verbose=verbose)
	
		if test_file != None:
			feats.make_bow(train_sentences, test_sentences)
		else:
			feats.make_bow(train_sentences)

	print('classifier: ' + classifier)
		
	# PREPARE DATA
	'''embeddings'''
	#X_train = feats.get_avg_embeddings(train_sentences)
	#X_test = feats.get_avg_embeddings(test_sentences)
	'''bow'''
	X_train = feats.get_representation(train_sentences)
	Y_train = np.array(train_labels)
	
	if test_file != None:
		X_test  = feats.get_representation(test_sentences)
	
	if test_file != None:
		Y_test = np.array(test_labels)
	
	print(X_train.shape)
	#Feature selection
	if fs and embedding_file == False:
		fs_clf = LinearSVC(C=0.25, penalty="l1", dual=False, random_state=1).fit(X_train,Y_train)
		X_train = SelectFromModel(fs_clf,prefit=True).transform(X_train)
		if test_file != None: X_test = SelectFromModel(fs_clf,prefit=True).transform(X_test)
		print(X_train.shape)
		if test_file != None: print(X_test.shape)
	
	model = get_model(model_name=classifier,verbose=True, md_options=md_options,classify=True)

	if fold:
		print('Cross-validation:')
		skf = StratifiedKFold(n_splits=10)
		skf.get_n_splits(X_train,Y_train)

		avg_acc = 0
		avg_f1 = nb_classes*[0]
		avg_cm = nb_classes*[nb_classes*[0]]
		for n_fold, (train_idx, test_idx) in enumerate(skf.split(X_train,Y_train)):
			train_samples, train_classes = X_train[train_idx], Y_train[train_idx]
			test_samples, test_classes = X_train[test_idx], Y_train[test_idx]
			# model = get_model(model_name=classifier,verbose=False, md_options=md_options,classify=True)
			# clf = model.fit(train_samples, train_classes)
			
			predictions = []
			for x in test_samples:
				if x[0] > x[2]:
					predictions.append(0)
				else:
					predictions.append(1)
			predictions = np.array(predictions)

			# predictions = clf.predict(test_samples)
			# print(predictions.shape)

			acc_pred = np.mean(predictions == test_classes)
			f1 = f1_score(test_classes, predictions, average=None)
			cm = confusion_matrix(test_classes, predictions)
			#Fold statistics
			print('Fold: %i' % int(n_fold+1))
			#Acuracy
			print('Acc: %.4f' % acc_pred)
			#F1 and F-Measure
			print("F1.",end='')
			for cl in range(0,nb_classes): print(' %s: %.3f' % (CLASSES[cl].lower(), f1[cl]), end='')
			print(' -> %f' % np.mean(f1))
			#Confusion Matrix
			print(cm)
			
			for cl in range(0,nb_classes): avg_cm[cl] = [x+y for x, y in zip(cm[cl],avg_cm[cl])]
			avg_f1 = [x+y for x, y in zip(f1,avg_f1)]
			avg_acc += acc_pred
		avg_acc /= 10
		for el in range(0,len(avg_f1)): avg_f1[el] /= 10
		for el in range(0,nb_classes): avg_cm[el] = [x * 0.1 for x in avg_cm[el]]
		print('Average Acc: %.4f' % avg_acc)
		print("Average F1.",end='')
		for cl in range(0,len(CLASSES)): print(' %s: %.4f' % (CLASSES[cl].lower(), avg_f1[cl]), end='')
		print(' -> %f' % np.mean(avg_f1))
		
		#um monte de frescura pro print ficar bonitinho - aqui só printa a matriz confusão
		print('[', end='')
		for i in range(0,nb_classes): 
			if i == 0: print('[',end='')
			else: print(' [',end='')
			for j in range(0,nb_classes):
				if j == nb_classes-1: print('%.1f' % avg_cm[i][j],end='')
				else: print('%.1f' % avg_cm[i][j],end=', ')
			if i == len(avg_cm)-1: print(']',end='')
			else: print(']')
		print(']')

	if fold and test_file != None:
		print('-------------------------')

	if test_file != None:
		print('Test:')
	# TRAIN MODEL
		model.fit(X_train, Y_train)

		predictions = model.predict(X_test) # (n_samples,)
		
		acc_pred = np.mean(predictions == Y_test)
		f1 = f1_score(Y_test, predictions, average=None)
		print('Acc: %.4f' % accuracy_score(Y_test, predictions))
		#print('Acc: %.4f' % acc_pred)
		print("F1.",end='')
		for cl in range(0,len(CLASSES)): print(' %s: %.3f' % (CLASSES[cl].lower(), f1[cl]), end='')# + ' ' + ) neg: %.3f neu: %.3f pos: %.3f" % f1[0], f1[1],f1[2]))
		print(' -> %f' % np.mean(f1))
		#Confusion Matrix
		print(confusion_matrix(Y_test, predictions))

def report(distances, predictions, Y_test):
	acc_pred = np.mean(predictions == Y_test)
	f1 = f1_score(Y_test, predictions, average=None)
	print('Acc: %.4f' % accuracy_score(Y_test, predictions))
	#print('Acc: %.4f' % acc_pred)
	print("F1. neg: %.3f neu: %.3f post: %.3f" % (f1[0], f1[1],f1[2]))
	
	acc_dist_min = np.mean(np.argmin(distances, axis=-1) == Y_test)
	acc_dist_max = np.mean(np.argmax(distances, axis=-1) == Y_test)
	print('Acc dist min: %.4f' % acc_dist_min)
	print('Acc dist max: %.4f' % acc_dist_max)
	
	acc_dist_min_equal = np.mean((np.argmin(distances, axis=-1) == Y_test) & (predictions != Y_test))
	acc_dist_max_equal = np.mean((np.argmax(distances, axis=-1) == Y_test) & (predictions == Y_test))
	print('Acc dist min equal: %.4f' % acc_dist_min_equal)
	print('Acc dist max equal: %.4f' % acc_dist_max_equal)
	
if __name__ == '__main__':
	
	#default options
	classifier='linearsvm'
	options='senti_words'
	md_options=''
	train_file='data/corpus/trainTT'
	test_file='data/corpus/testTT'
	fold = False
	balance = False
	nb_classes = 3
	fs = False
	verbose = False
	rmttsbr = False
	embedding_file = False
	
	i = 0
	while i <= len(sys.argv[1:]):
		if sys.argv[i].lower() == '-help' or sys.argv[i].lower() == '-h':
			help()
		if sys.argv[i].lower() == '-v' or sys.argv[i].lower() == '-verbose':
			verbose = True
		if sys.argv[i].lower() == '-fold':
			fold = True
		# if sys.argv[i].lower() == '-fs':
		# 	fs = True
		if sys.argv[i].lower() == '-balance':
			balance = True
		if sys.argv[i].lower() == '-rmttsbr':
			rmttsbr = True
		# if sys.argv[i].lower() == '-classifier':
		# 	classifier = sys.argv[i+1]
		# 	if classifier not in ['linearsvm', 'polysvm','nb','lr','mlp','trees','randfor']:
		# 		help('not a valid classifier')
		# 	i += 1
		# if sys.argv[i].lower() == '-options' or sys.argv[i].lower() == '-opt':
		# 	options = sys.argv[i+1]
		# 	i += 1
		# 	options = options.replace(',',' ')
		if sys.argv[i].lower() == '-md_options':
			md_options = sys.argv[i+1]
			i += 1
		if sys.argv[i].lower() == '-nb_classes':
			nb_classes = int(sys.argv[i+1])
			i += 1
		# if sys.argv[i].lower() == '-emb':
		# 	embedding_file = sys.argv[i+1]
		# 	i += 1
		if sys.argv[i].lower() == '-train_file':
			train_file = sys.argv[i+1]
			i += 1
		if sys.argv[i].lower() == '-test_file':
			if sys.argv[i+1].lower() == 'none':
				test_file = None
			else:
				test_file = sys.argv[i+1]
			i += 1
		i += 1

	run(classifier=classifier, md_options=md_options, fs=fs,nb_classes=nb_classes, fold=fold, options=options, verbose=verbose,
		train_file=train_file, test_file=test_file, embedding_file=embedding_file, rmttsbr=rmttsbr, force_balance=balance)
	#new_run(sys.argv[1:])
