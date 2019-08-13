import numpy as np
from embedding import Word2Vec
import nlpnet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from scipy import sparse

BOW_SIZE = 4000

class FeatureExtractor:

	rep_size = 0

	def __init__(self, bow=True, negation=True, emoticon=True, emoji=True, senti_words=True,
						   postag=True, bow_grams=1, emb=None, embedding_file=None, verbose=True):
		self.verbose = verbose
		self.bow = bow
		self.negation = negation
		self.emoticon = emoticon
		self.emoji = emoji
		self.senti_words = senti_words
		self.postag = postag

		self.emb = emb
		#If embeddings are being used the rest is descarded
		if emb:
			if self.verbose:
				if self.verbose: print("using embeddings: " + emb)
			if emb == 'Word2Vec':
				self.emb_model = Word2Vec()
				if embedding_file:
					self.emb_model.load(embedding_file)
					self.rep_size = self.emb_model.dimensions
				else:
					if self.verbose: printl("** No embedding file defined. Am I supposed to guess? **")
					exit()
		else:
			if verbose: print('using representation: ',end='')
			if bow:
				if verbose: print('bow ',end='')
				self.bow_vectors = HashingVectorizer(analyzer='word', ngram_range=(1, bow_grams), n_features=BOW_SIZE) 
			if negation:
				if verbose: print('negation ',end='')
				self.make_negations()
				self.negation_index = self.rep_size
				self.rep_size += 1

			if emoticon:
				if verbose: print('emoticon ',end='')
				self.make_emoticons()
				self.emoticon_index = self.rep_size
				self.rep_size += 2
			if emoji:
				if verbose: print('emoji ',end='')
				self.make_emojis()
				self.emojis_index = self.rep_size
				self.rep_size += 3
			if senti_words:
				if verbose: print('senti_words ',end='')
				self.make_sentilex()
				self.sentilex_index = self.rep_size
				self.rep_size += 3
			if postag:
				if verbose: print('PoS ',end='')
				self.make_pos()
				self.pos_index = self.rep_size
				self.rep_size += 4
			if self.verbose: print()


	def make_bow(self, *args):
		if self.bow:
			all_docs = []
			for file in args:
				for doc in file:
					all_docs.append(doc)
			self.bow_vectors.fit(all_docs)
			self.bow_size = BOW_SIZE
			self.rep_size += self.bow_size
	'''
	def make_bow(self, *args):
		if self.bow:
			all_docs = []
			for file in args:
				for doc in file:
					all_docs.append(doc)
			self.bow_vectors.fit(all_docs)
			self.bow_size = len(self.bow_vectors.get_feature_names())
			self.rep_size += self.bow_size
	'''

	def make_negations(self):
		self.negation_words = ['jamais','nada', 'nem','nenhum', 'nenhures', 'ninguém',
				                 'ninguem', 'nonada', 'nulidade', 'nunca', 'não', 'nao',
				                 'tampouco', 'zero']
	
	def make_emoticons(self):
		self.pos_emoticons = [':-)',':)',':o)',':]',':3',':c)',':>','=]','8)','=)',':}',':^)',
				                      ':-))','|;-)',":'-)",":')",'\o/','*\\0/*',':-D',':D','8-D','8D',
				                      'x-D','xD','X-D','XD','=-D','=D','=-3','=3','B^D','<3',';-)',';)',
				                      '*-)','*)',';-]',';]',';D',';^)',':-,']
		self.neg_emoticons = ['>:\\','>:/',':-/',':-.',':/',':\\','=/','=\\',':L','=L',':S','>.<',
				                      ':-|','<:-|','>:[',':-(',':(',':-c',':c',':-<',':<',':-[',':[',':{',
				                      ':-||',':@',":'-(",":'(",'D:<','D:','D8','D;','D=','DX','v.v',"D-':",
				                      '(>_<)',':|']

	def make_emojis(self, path='./data/resources/Emoji_Sentiment_Data_v1.0.csv'):
		self.emoji_list = {}
		with open(path,'r') as emoji_file:
			for line in emoji_file:
				if line[0] != '#':
					line = line.strip().split(',')
					tot_oc = float(line[2])
					pos_oc = float(line[4])/tot_oc
					neu_oc = float(line[5])/tot_oc
					neg_oc = float(line[6])/tot_oc
					self.emoji_list[line[0]] = (pos_oc,neu_oc,neg_oc)

	def make_sentilex(self, path='./data/resources/sentilex-reduzido.txt'):
		self.sentilex = {}
		with open(path,'r') as sentilex_file:
			for line in sentilex_file:
				if line[0] != '#':
					line = line.strip().split(',')
					self.sentilex[line[0]] = int(line[1])

	def make_pos(self, path='./data/resources/pos-pt'):
		nlpnet.set_data_dir(path)
		#self.tagger = nlpnet.POSTagger()

	def get_representation(self, sentences):
		#if self.verbose: print('sentence representation size: ' + str(self.rep_size))
		if self.verbose: print(len(sentences))
		if self.verbose: print(self.rep_size)
		new_sentences = np.zeros((len(sentences), self.rep_size))
		# new_sentences = sparse.lil_matrix(new_sentences)
		if self.emb:
			for i, sent in enumerate(sentences):
				if self.verbose: print('%i/%i' % (i,len(sentences)),end='\r')
				for word in sent.split():
					new_sentences[i] += self.emb_model.get_vector(word)
				new_sentences[i] /= len(sent)
		else:
			for i, sent in enumerate(sentences):
				if self.verbose: print('%i/%i' % (i,len(sentences)),end='\r')
				sent = sent.split()
				if self.bow:
					for word in self.bow_vectors.transform(sent):
						new_sentences[i,:self.bow_size] += np.array(word.toarray()).flatten()

				if self.negation:
					neg_value = 0
					for word in sent:
						if word in self.negation_words:
							neg_value += 1
					new_sentences[i,self.negation_index] = neg_value

				if self.emoticon:
					pos_emot = 0
					neg_emot = 0
					for word in sent:
						if word in self.pos_emoticons:
							pos_emot += 1
						if word in self.neg_emoticons:
							neg_emot += 1
					new_sentences[i,self.emoticon_index] = pos_emot
					new_sentences[i,self.emoticon_index+1] = neg_emot

				if self.emoji:
					sent_emojis = [0,0,0]
					emoji_c = 0
					for word in sent:
						if word in self.emoji_list.keys():
							emoji_c += 1
							for j in range(0,2):
								sent_emojis[j] += self.emoji_list[word][j]
					if emoji_c > 0:
						for j in range(0,2): sent_emojis[j] = (sent_emojis[j]*100) / emoji_c
					new_sentences[i,self.emojis_index:self.emojis_index+3] += np.array(sent_emojis)

				if self.senti_words:
					sent_words = [0,0,0]
					for word in sent:
						if word in self.sentilex.keys():
							sent_words[self.sentilex[word]] += 1
					new_sentences[i,self.sentilex_index:self.sentilex_index+3] += np.array(sent_words)

				if self.postag:
					N_count = 0
					ADV_count = 0
					ADJ_count = 0
					V_count = 0
					# for tag in self.tagger.tag(' '.join(sent)):
					# 	for pair in tag:
					# 		if pair[1] == 'ADJ':
					# 			ADJ_count += 1
					# 		elif pair[1] == 'ADV':
					# 			ADV_count += 1
					# 		elif pair[1] == 'N':
					# 			N_count += 1
					# 		elif pair[1] == 'V':
					# 			V_count += 1
					new_sentences[i,self.pos_index:self.pos_index+4] += np.array([N_count,ADV_count,ADJ_count,V_count])
		if self.verbose: print('%i/%i' % (len(sentences),len(sentences)))
		return new_sentences