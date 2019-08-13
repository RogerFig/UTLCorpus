import nltk, sys

reviews = open(sys.argv[1],'r')

line_spacing = 5

if len(sys.argv)>2:
	line_spacing = int(sys.argv[2])

nb_sentencas = 0
nb_tokens = 0
types = set()
all_tokens = []
linhas = 0
for linha in reviews.readlines():
	texto = ' '.join(linha.split(' ')[line_spacing:])
	sentencas = nltk.sent_tokenize(texto)
	for sentenca in sentencas:
		nb_sentencas += 1
		tokens = nltk.word_tokenize(sentenca)
		all_tokens.extend(tokens)
		nb_tokens += len(tokens)
		types.update(tokens)
	linhas+=1
	#print(linhas,end='\r')


fdist = nltk.FreqDist(all_tokens)
print("Statistics ",sys.argv[1])
print("#Tokens: ", nb_tokens)
print("#Types: ", len(types))
print("#sentencas: ", nb_sentencas)
print(fdist.most_common(10000))