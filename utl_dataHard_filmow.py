import sys, ast, os
import numpy as np
from scipy import stats


#neg_file = open('filmow_utl.neg','w')
#pos_file = open('filmow_utl.pos','w')
err_file = open('filmow_utl.err','w')

path = sys.argv[1]
folders = os.listdir(path)

contador = 0
for folder in folders:
  total = []
  files = []
  for f in os.listdir(path+'/' +folder):
    for line in open(path+'/'+folder+'/'+f):
      contador += 1
      print('%i/1839851' % contador,end='\r')
      files.append(ast.literal_eval(line.strip()))
      if int(files[-1]['likes']) not in total:
        total.append(int(files[-1]['likes']))
        c_value = int(files[-1]['likes'])
        
      break
  if len(total) == 0: continue
  total.sort()
  try:
    percentil = np.percentile(total,10)
  except:
    print(folder)
    exit()
  for d in files:
    d['texto'] = d['texto'].strip().replace('\n','').strip()
    
    if float(d['likes']) > percentil:
      try:
        pass
        #pos_file.write('0'*(18-len(d['id'])) + d['id'].strip() + ' ' + d['texto']+'\n')
      except:
        err_file.write(d['id']+'\n')
    else:
        
        if 'horas' in d['data'] or 'minutos' in d['data'] or '1 dia ' in d['data'] or ('dias' in d['data'] and int(d['data'].split(' ')[0]) < 5):
          err_file.write(d['id']+'\n')
          continue
        try:
          pass
          #neg_file.write('0'*(18-len(d['id'])) + d['id'].strip() + ' ' + d['texto']+'\n')
        except:
          err_file.write(d['id']+'\n')

