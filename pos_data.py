import sys, ast

neg_file = open(sys.argv[1][:-3]+'neg','w')
pos_file = open(sys.argv[1][:-3]+'pos','w')
err_file = open(sys.argv[1][:-3]+'err','w')

with open(sys.argv[1],'r') as input_file:
      for line in input_file:
        d = ast.literal_eval(line)
        d['texto'] = d['texto'].strip().replace('\n','').strip()
        if float(d['estrelas']) > 3:
          try:
            pos_file.write('0'*(18-len(d['id'])) + d['id'].strip() + ' ' + d['texto']+'\n')
          except:
            err_file.write(d['id'].strip() + '\t' + d['estrelas']+'\n')
        else:
          try:
            neg_file.write('0'*(18-len(d['id'])) + d['id'].strip() + ' ' + d['texto']+'\n')
          except:
            err_file.write(d['id'].strip() + '\t' + d['estrelas']+'\n')
neg_file.close()
pos_file.close()
err_file.close()
