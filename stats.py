import sys

nb_tokens = 0.0
avg_token = []
words = {}
sentences = []
nb_sentences = 0

#otp_corpora = open('pstore_corpus_all.txt','w')

all = 0
repeated = 0
for arq in sys.argv[1:]:
  print('reading ',arq)
  with open(arq,'r') as input_file:
    for line in input_file:
      all += 1
      if all % 20000 == 0: print(all)
      if len(line) < 1: continue
      if line.split(' ')[0] in sentences:
        repeated += 1
        continue
      #print('%i/0' % all,end='\r')
      line = line.split(' ')
      sentences.append(line[0])
      nb_sentences += 1
      line = line[1:]
      avg_token.append(len(line))
      for word in line:
        nb_tokens += 1
        if word not in words.keys():
          words[word] = 1
        else:
          words[word] += 1
  print(all)
  print(len(sentences))
  print('repeated: ', repeated)
avg = float(sum(avg_token))/float(len(sentences))

print(all)
print('# words: ',len(words.keys()))
print('# tokens: ',nb_tokens)
print('# sentences: ',len(sentences))
print(nb_sentences)
print('Avg. tok/sent: ', avg)

#  print('words: \n',nb_words)
