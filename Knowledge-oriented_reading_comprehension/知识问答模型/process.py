with open('input2.txt','r') as f:
    filling=f.readlines()
with open('input3.txt','w') as fp:
    fp.write('|||'.join([s.strip('\n') for s in filling]))            