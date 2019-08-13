import sys, ast, os
import numpy as np
from scipy import stats
import xml.etree.ElementTree as ET

folders = ['play_store']#, 'play_store','buscape']
#folders = ['buscape']
size_fd = [1041738,1839851,1041738,85910]

full_data = []
inputs = []

for folder in folders:
  full_data.append(open(folder+'.all','w'))
  l_f = []
  for i, file in enumerate(['_sent','_util']):
    l = []
    for j, ext in enumerate(['.neg','.pos','.err']):
      l.append(open(folder+file+ext,'w'))
    l_f.append(l)
  inputs.append(l_f)


# creating write data
for id_f, folder in enumerate(folders):
  print('reading ',folder,'---------------')

  # reading data
  subfolders = os.listdir(folder+'/')
  
  contador = 0
  for sub in subfolders:
    print('r. ',sub)
    # if its's not buscape
    if folder != 'buscape':
      
      total = []
      files = []
    
      for f in os.listdir(folder+'/'+sub):

        for line in open(folder+'/'+sub+'/'+f):
          contador += 1
          print('%i/%d' % (contador,size_fd[id_f]),end='\r')
          obj_review = ast.literal_eval(line.strip())
          files.append(obj_review)
          obj_review['contador'] = '0'*(18-len(str(contador))) + str(contador)
          if int(files[-1]['likes']) not in total:
            total.append(int(obj_review['likes']))

        if len(total) != 0:
          total.sort()
        percentil = np.percentile(total,10)
        
      for d in files:
        polarity = '-'
        utility  = '-'
        
        d['id'] = str(d['id'])
        d['texto'] = d['texto'].strip().replace('\n','').strip()
        d['estrelas'] = float(d['estrelas'])

        # HELPFULNESS ---------------------------------------------------------------
        if len(total) > 0:
          if float(d['likes']) > percentil:
            try:
              inputs[id_f][1][1].write(d['contador'] + ' ' + d['texto']+'\n')
              utility = '1'
            except:
              inputs[id_f][1][2].write(d['id']+ ' 1\n')
          else:

            # Helpfulness in FILMOW
            if folder == 'filmow':
              if 'horas' in d['data'] or 'minutos' in d['data'] or '1 dia ' in d['data'] or ('dias' in d['data'] and int(d['data'].split(' ')[0]) < 5):
                inputs[id_f][1][2].write(d['id']+'-\n')
              else:
                try:
                  inputs[id_f][1][0].write(d['contador'] + ' ' + d['texto']+'\n')
                  utility = '0'
                except:
                  inputs[id_f][1][2].write(d['id']+'0\n')
            
            # Helpfulness in PLAY STORE
            else:
              d_date = d['data'].split(' ')
              if int(d_date[0]) > 25 and d_date[-1] == '2019' and d_date[2] == 'março':
                inputs[id_f][1][2].write(d['id']+'.\n')
              else:
                try:
                  inputs[id_f][1][0].write(d['contador'] + ' ' + d['texto']+'\n')
                  utility = '0'
                except:
                  inputs[id_f][1][2].write(d['id']+'0\n')

        # POLARITY ------------------------------------------------
        if d['estrelas'] != 0 and d['estrelas'] != 3 and d['estrelas'] != 5:
          try:
            if float(d['estrelas']) > 3:
              inputs[id_f][0][1].write(d['contador'] + ' ' + d['texto']+'\n')
              polarity = '1'
            else:
              inputs[id_f][0][0].write(d['contador'] + ' ' + d['texto']+'\n')
              polarity = '0'
            
          except:
            inputs[id_f][0][2].write(d['contador'] + ' ' + str(d['estrelas'])+'\n')
          
        if polarity != '-' or utility != '-':
          line_to_write = d['contador'] + ' ' + d['id'] + ' ' + polarity + ' ' + utility + ' ' + sub + ' ' + d['texto'] + '\n'
          #print(line_to_write)
          full_data[id_f].write(line_to_write)

    # BUSCAPE ----------
    else:
      for ssub in os.listdir(folder+'/'+sub+'/'):
        total = []
        files = []
        for f in os.listdir(folder+'/'+sub+'/'+ssub):
          xml_file = folder+'/'+sub+'/'+ssub+'/'+f
          if f[-3:] != 'xml': continue
          contador += 1
          print('%i/%d' % (contador,size_fd[id_f]),end='\r')
          tree = ET.parse(xml_file)
          t = tree.getroot()
          obj_review = dict()
          for child in t:
            if child.tag == 'stars':
              obj_review['estrelas'] = float(child.attrib['value'])
            if child.tag == 'thumbsUp':
              obj_review['likes'] = int(child.attrib['value'])
            if child.tag == 'opinion':
              if child.text == None:
                obj_review['texto'] = ''
              else:
                obj_review['texto'] = child.text.strip().replace('\n',' ').strip()
          obj_review['id'] = f.split('.')[0].split('_')[-1]

          files.append(obj_review)
          obj_review['contador'] = '0'*(18-len(str(contador))) + str(contador)
          if int(files[-1]['likes']) not in total:
            total.append(int(obj_review['likes']))
        
        if len(total) != 0:
          total.sort()
        percentil = np.percentile(total,10)  

        for d in files:
          polarity = '-'
          utility  = '-'
          
          d['id'] = str(d['id'])
          d['texto'] = d['texto'].strip().replace('\n','').strip()
          d['estrelas'] = float(d['estrelas'])

          # HELPFULNESS ---------------------------------------------------------------
          if len(total) > 0:
            if float(d['likes']) > percentil:
              try:
                inputs[id_f][1][1].write(d['contador'] + ' ' + d['texto']+'\n')
                utility = '1'
              except:
                inputs[id_f][1][2].write(d['id']+ ' 1\n')
            else:

              try:
                inputs[id_f][1][0].write(d['contador'] + ' ' + d['texto']+'\n')
                utility = '0'
              except:
                inputs[id_f][1][2].write(d['id']+'0\n')
              
              
          # POLARITY ------------------------------------------------
          if d['estrelas'] != 0 and d['estrelas'] != 3 and d['estrelas'] != 5:
            try:
              if float(d['estrelas']) > 3:
                inputs[id_f][0][1].write(d['contador'] + ' ' + d['texto']+'\n')
                polarity = '1'
              else:
                inputs[id_f][0][0].write(d['contador'] + ' ' + d['texto']+'\n')
                polarity = '0'
              
            except:
              inputs[id_f][0][2].write(d['contador'] + ' ' + str(d['estrelas'])+'\n')
            
          if polarity != '-' or utility != '-':
            line_to_write = d['contador'] + ' ' + d['id'] + ' ' + polarity + ' ' + utility + ' ' + folder + ' ' + d['texto'] + '\n'
            #print(line_to_write)
            full_data[id_f].write(line_to_write)


for i in inputs:
  for j in i:
    for k in j:
      k.close()
for i in full_data:
  i.close()