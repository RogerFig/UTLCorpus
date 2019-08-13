import sys, ast, os
import numpy as np
from scipy import stats

neg_file = open('pstore_utl.neg','w')
pos_file = open('pstore_utl.pos','w')
err_file = open('pstore_utl.err','w')

path = sys.argv[1]
folders = os.listdir(path)

contador = 0
for folder in folders:
  total = []
  files = []
  for f in os.listdir(path+'/' +folder):
    for line in open(path+'/'+folder+'/'+f):
      contador += 1
      print('%i/1041738' % contador,end='\r')
      files.append(ast.literal_eval(line.strip()))
      if int(files[-1]['likes']) not in total:
        total.append(int(files[-1]['likes']))
        c_value = int(files[-1]['likes'])
        
      break
  total.sort()
  percentil = np.percentile(total,10)

  for d in files:
    d['texto'] = d['texto'].strip().replace('\n','').strip()
    d['id'] = str(d['id'])
    d_date = d['data'].split(' ')

    if float(d['likes']) > percentil:
      try:
        pos_file.write('0'*(18-len(d['id'])) + d['id'].strip() + ' ' + d['texto']+'\n')
      except:
        err_file.write(d['id'])
    else:
        if int(d_date[0]) > 25 and d_date[-1] == '2019' and d_date[2] == 'março':
          err_file.write(d['id']+'\n')
          continue
        try:
          neg_file.write('0'*(18-len(d['id'])) + d['id'].strip() + ' ' + d['texto']+'\n')
        except:
          err_file.write(d['id'])

exit()



with open(sys.argv[1],'r') as input_file:
      for line in input_file:
        d = ast.literal_eval(line)
        d['texto'] = d['texto'].strip().replace('\n','').strip()
        d['id'] = str(d['id'])
        d_date = d['data'].split(' ')
        if int(d_date[0]) > 25 and d_date[-1] == '2019' and d_date[2] == 'março':
            err_file.write(d['id']+'\n')
            continue

        if float(d['likes']) > 5:
          try:
            pos_file.write('0'*(18-len(d['id'])) + d['id'].strip() + ' ' + d['texto']+'\n')
          except:
            err_file.write(d['id'])
        else:
          try:
            neg_file.write('0'*(18-len(d['id'])) + d['id'].strip() + ' ' + d['texto']+'\n')
          except:
            err_file.write(d['id'])
neg_file.close()
pos_file.close()
err_file.close()
