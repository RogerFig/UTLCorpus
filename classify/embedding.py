from abc import ABCMeta, abstractmethod
import numpy as np
import pickle # for glove

import gensim # for word2vec
#import glove  # pip3 install git+https://github.com/maciejkula/glove-python.git
#import fasttext # pip3 install git+https://github.com/salestock/fastText.py


class Embedding(metaclass=ABCMeta):

	def __init__(self, lowercase=False, dimensions=50, min_count=5, workers=4, window_size=5, epochs=5, oov_word='*RARE*'):
		self.lowercase = lowercase
		self.dimensions = dimensions
		self.min_count = min_count
		self.workers = workers
		self.window_size = window_size
		self.epochs = epochs
		self.oov_word = oov_word
		self.oov = dict()
		self.model = dict()
		self.random_vector = None
		self._vocabulary = None

	@abstractmethod
	def load(self, load_file): 
		pass

	@abstractmethod
	def save(self, save_file): 
		pass

	@abstractmethod
	def train(self, train_file, **kwargs): 
		pass

	@property
	def vocabulary(self): 
		if self._vocabulary is None:
			self._vocabulary = dict(zip(self.model.keys(), range(len(self.model))))
		return self._vocabulary

	@property
	def vocabulary_size(self):
		return len(self.vocabulary)

	def get_vector(self, word):
		if word in self.model:
			return self.model[word]
		if self.lowercase and word.lower() in self.model:
			return self.model[word.lower()]
		if word not in self.oov:
			self.oov[word] = 0
		self.oov[word] += 1
		if self.oov_word != None and self.oov_word in self.model:
			return self.model[self.oov_word]
		if self.random_vector is None:
			self.random_vector = self.generate_random_vector()
		return self.random_vector

	def generate_random_vector(self):
		epsilon = np.sqrt(6) / np.sqrt(self.dimensions)
		return np.random.random(self.dimensions) * 2 * epsilon - epsilon

	def statistics(self, top_k=10):
		nb_oovs = len(self.oov)
		nb_occur_oovs = sum(list(self.oov.values()))
		top_k_oovs = sorted(list(self.oov.items()), key=lambda x: x[1], reverse=True)
		return nb_oovs, nb_occur_oovs, top_k_oovs[:top_k]

	def oov_statistics(self, list_of_words, top_k=10):
		oovs = {}
		for x in list_of_words:
			word = x.lower() if self.lowercase else x
			if word not in self.model:
				if word not in oovs:
					oovs[word] = 0
				oovs[word] += 1
		nb_oovs = len(oovs)
		nb_occur_oovs = sum(list(oovs.values()))
		top_k_oovs = sorted(list(oovs.items()), key=lambda x: x[1], reverse=True)
		return nb_oovs, nb_occur_oovs, top_k_oovs[:top_k]


class LoadableEmbedding(Embedding):

	def save(self, save_file):
		raise Exception('This model is already saved.')

	def train(self, train_file):
		raise Exception('This class just load word embeddings.')




class Word2Vec(Embedding):
	@property
	def vocabulary(self): 
		if self._vocabulary is None:
			self._vocabulary = dict(zip(self.model.vocab.keys(), range(len(self.model.vocab))))
		return self._vocabulary

	def load(self, load_file):
		self.model = gensim.models.Word2Vec.load(load_file)
		self.dimensions = self.model.vector_size
		
	def train(self):
		pass
	
	def save(self, save_file):
		pass


'''
class DGlove(glove.Glove):
	
	@classmethod
	def load(cls, filename):
		instance = DGlove()
		with open(filename, 'rb') as savefile:
			instance.__dict__ = pickle.load(savefile)
		return instance

	def __contains__(self, x):
		return x in self.dictionary

	def __getitem__(self, x):
		return self.word_vectors[self.dictionary[x]]


class Glove(Embedding):

	@property
	def vocabulary(self): 
		if self._vocabulary is None:
			self._vocabulary = self.model.dictionary
		return self._vocabulary

	def load(self, load_file):
		self.model = DGlove.load(load_file)
		self.dimensions = self.model.no_components


class FastText(Embedding):

	@property
	def vocabulary(self): 
		if self._vocabulary is None:
			self._vocabulary = self.model.words
		return self._vocabulary

	def load(self, load_file):
		self.model = fasttext.load_model(load_file)
		self.dimensions = self.model.dim

	def get_vector(self, word):
		if word in self.model:
			return self.model[word]
		if self.lowercase and word.lower() in self.model:
			return self.model[word.lower()]
		if word not in self.oov:
			self.oov[word] = 0
		self.oov[word] += 1
		return self.model[word]


class Wang2Vec(LoadableEmbedding):
	"""
	The embeddings in `load_file` should be as plain text (not binary)
	"""
	def load(self, load_file):
		self.model = dict()
		ignore_first_line = True # metadata
		vocab_size, word_dim = 0, 0
		with open(load_file, 'r', encoding="utf-8") as f:
			for line in f:
				if ignore_first_line:
					vocab_size, word_dim = map(int, line.split())
					ignore_first_line = False
					continue
				data = line.split()
				self.model[data[0]] = list(map(float, data[1:]))
		self.dimensions = word_dim

'''