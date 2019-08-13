import sys, ast

neg_file = open(sys.argv[1][:-3]+'neg','w')
pos_file = open(sys.argv[1][:-3]+'pos','w')
err_file = open(sys.argv[1][:-3]+'err','w')

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
