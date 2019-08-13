from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier


def getOptions(opt_lst):
	d = {}
	if len(opt_lst) == 0: return d
	for i in opt_lst:
		if len(i) == 0: continue
		e = i.strip().split(':')
		d[e[0].lower()] = e[1]
	return d

def printOptions(opt_dic):
	l = ''
	for i in opt_dic.keys():
		l += i + ':' + opt_dic[i] + ' '
	print(l)

def get_model(model_name='', verbose=False, md_options='',classify=False): 
	# Model options
	md_options = getOptions(md_options.split(','))
	if verbose: print('model_name: ' + model_name)
	if verbose: printOptions(md_options)
	if model_name.lower() == 'linearsvm':
		# SVM LINEAR
		#  penalty=’l2’, loss=’squared_hinge’, dual=True, tol=0.0001,
		#  C=1.0, multi_class=’ovr’, fit_intercept=True, intercept_scaling=1,
		#  class_weight=None, verbose=0, random_state=None, max_iter=1000
		C = 1.0	#best fit in eval
		dual=False	#best fit in eval
		if 'c' in md_options.keys(): C = float(md_options['c'])
		if 'dual' in md_options.keys(): dual = bool(md_options['dual'])
		if classify:
			return LinearSVC(C=C, dual=dual)
		return SVC(kernel='linear', C=C, probability=True, verbose=False, max_iter=1000)
	elif model_name.lower() == 'polysvm':
		# SVM - poly
		#  C=1.0, kernel=’rbf’, degree=3, gamma=’auto’, coef0=0.0, shrinking=True,
		#  probability=False, tol=0.001, cache_size=200, class_weight=None,
		#  verbose=False, max_iter=-1, decision_function_shape=’ovr’,
		#  random_state=None
		return SVC(C=10.0, kernel='poly', verbose=True)
	elif model_name.lower() == 'nb':
		# NAIVE BAYES
		alpha = 0.1 #best fit in eval
		if 'alpha' in md_options.keys(): alpha = float(md_options['alpha'])
		#  alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None
		return BernoulliNB(alpha=alpha)
	elif model_name.lower() == 'lr':
		# LOGISTIC REGRESSOR
		#  penalty=’l2’, dual=False, tol=0.0001, C=1.0, fit_intercept=True,
		#  intercept_scaling=1, class_weight=None, random_state=None, solver=’liblinear’,
		#  max_iter=100, multi_class=’ovr’, verbose=0, warm_start=False, n_jobs=1
		return LogisticRegression(n_jobs=-1)
	elif model_name.lower() == 'mlp':
		# MULTI-LAYER PERCEPTRON
		#  hidden_layer_sizes=(100, ), activation=’relu’, solver=’adam’, alpha=0.0001,
		#  batch_size=’auto’, learning_rate=’constant’, learning_rate_init=0.001, power_t=0.5,
		#  max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False,
		#  momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08
		activation = 'tanh' 
		learning_rate = 0.001 #best fit in eval
		alpha = 0.0001 #best fit in eval
		layers = 2 #best fit in eval
		neurons = (200,) #best fit in eval
		if 'lr' in md_options.keys(): 
			learning_rate = float(md_options['lr'])
		if 'alpha' in md_options.keys():
			alpha = float(md_options['alpha'])
		if 'layers' in md_options.keys():
			layers = int(md_options['layers'])
		if 'neurons' in md_options.keys():
			if layers > 1:
				neurons = (int(md_options['neurons']),int(md_options['neurons']),)
			else:
				neurons = (int(md_options['neurons']),)
		return MLPClassifier(activation=activation, learning_rate_init=learning_rate, learning_rate='adaptive',
			                 alpha=0.001, early_stopping=True, hidden_layer_sizes=neurons)
	elif model_name.lower() == 'trees':
		# DECISION TREE
		#  criterion=’gini’, splitter=’best’, max_depth=None, min_samples_split=2,
		#  min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None,
		#  max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False
		criterion='gini' # best fit
		max_depth=None #best fit
		if 'criterion' in md_options.keys():
			criterion = md_options['criterion']
		if 'max_depth' in md_options.keys():
			if md_options['max_depth'].lower() == 'none':
				max_depth = None
			else:
				max_depth = int(md_options['max_depth'])
		return DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
	elif model_name.lower() == 'randfor':
		# DECISION TREE
		#  criterion=’gini’, splitter=’best’, max_depth=None, min_samples_split=2,
		#  min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None,
		#  max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False
		criterion = 'entropy' # best fit
		estimators = 200 # best fit
		max_depth=None # best fit
		if 'estimators' in md_options.keys():
			estimators = int(md_options['estimators'])
		if 'max_depth' in md_options.keys():
			if md_options['max_depth'].lower() == 'none':
				max_depth = None
			else:
				max_depth = int(md_options['max_depth'])
		if 'criterion' in md_options.keys():
			criterion = md_options['criterion']
		return RandomForestClassifier(criterion=criterion, n_estimators=estimators, max_depth=max_depth)
	else:
		if verbose: print('no classifier given. What do you expect me to do?')
		exit()

def get_distances(model, X_test, model_name='', verbose=False):
	if model_name.lower() == 'linearsvm':
		return model.predict_proba(X_test)
	elif model_name.lower() == 'polysvm':
		return model.decision_function(X_test)
	elif model_name.lower() == 'nb':
		return model.predict_proba(X_test)
	elif model_name.lower() == 'lr':
		return model.predict_proba(X_test)
	elif model_name.lower() == 'mlp':
		return model.predict_proba(X_test)
	elif model_name.lower() == 'trees':
		return model.predict_proba(X_test)
	elif model_name.lower() == 'randfor':
		return model.predict_proba(X_test)
	else:
		if verbose: print('no classifier given. What do you expect me to do?')
		exit()
	