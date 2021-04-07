# -*- coding: utf-8 -*-
import os
import getopt
import sys
import re

inputarg=""
outputarg=""
try:
    opts, args = getopt.getopt(sys.argv[1:],"hi:o:",["in=","out="])
except getopt.GetoptError:
    printUsage()
    sys.exit(-1)
for opt,arg in opts:
    if opt == '-h':
        printUsage()
    elif opt in ("-i", "--id"):
        idarg=arg
    elif opt in ("-n","--name"):
        namearg=arg
        
print ('id:'+inputarg)
#print ('name:'+outputarg)
print ('other param:'+",".join(args))

with open('./log/' + args[0], "r") as f:
    data1 = f.read()
    s1 = re.split('\n', data1)
    print(s1[0])
with open('./log/' + args[1], "r") as f:
    data2 = f.read()
    s2 = re.split('\n', data2)
    print(s2[0])

root = r'./data/nii_base/{}_{}'.format(s1[0],s2[0])

foldernames = os.listdir(root)
niis = []
nii_counts = []

for foldername in foldernames:
    filenames = os.listdir(os.path.join(root, foldername))
    for col, filename in enumerate(filenames):
        if filename not in niis:
            niis.append(filename)
            nii_counts.append(1)
        else:
            nii_counts[niis.index(filename)] += 1

zipped = zip(niis, nii_counts)
sort_zipped = sorted(zipped, key=lambda x:(x[1],x[0]))
result = zip(*sort_zipped)
niis_sorted, nii_counts_sorted = [list(x) for x in result]
niis_sorted.reverse()
nii_counts_sorted.reverse()

with open('./log/labels.txt', 'w') as f:
    f.write('label names: \n'.format(len(niis)))
    for i in range(len(niis_sorted)):
        f.write('{}->{}\n'.format(niis_sorted[i], nii_counts_sorted[i]))
