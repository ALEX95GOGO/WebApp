import os
import sys
import shutil
import random
import csv
import getopt
import re

def printUsage():
	print ('''usage: prepare_data.py -i <input> -o <output>
       test.py --in=<input> --out=<output>''')
       
if __name__ == '__main__':
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
    with open('./log/' + args[0], "r") as f:
        data1 = f.read()
        s1 = re.split('\n', data1)
        print(s1[0])
    with open('./log/' + args[1], "r") as f:
        data2 = f.read()
        s2 = re.split('\n', data2)
        print(s2[0])

    task_id = int(s1[0])
    task_name = str(s2[0])
    target_path = r'./data/nii_base/Task{}/'.format(task_id)
    train_images = r'./data/nii_base/Task{}/'.format(task_id)
    
    folder_list = os.listdir(train_images)
    #rint(folder_list)
    #
    k = len(folder_list) // 5  # number of test images
    
    index = [i for i in range(len(folder_list))]
    random.shuffle(index)
    
   
    for ii in range(k):
         shutil.move(train_images + folder_list[ii],
                         target_path + 'test/' + folder_list[ii])
    
    #for ii in range(k):
    #     shutil.copytree(train_images + folder_list[ii+k],
    #                     target_path + 'val/' + folder_list[ii+k])
    
    for ii in range(len(folder_list) - k):
         shutil.move(train_images + folder_list[ii+k],
                         target_path + 'train/' + folder_list[ii+k])





