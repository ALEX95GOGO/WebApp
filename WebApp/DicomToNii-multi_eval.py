#!/usr/bin/env python
# coding: utf-8
# nameDicom&rtss -> nii.gz
# date: 2021.02.02 auther: lecheng.jia
# requirements: python 3.7+, md
# added multiprocessing
'''
    dicom data should list like this, each case has a series of ct slice and ONLY one rtss
    - dcm
        - case1
            - ct001.dcm
            - ct002.dcm
            - ct003.dcm
            ...
            - rtss.dcm
        - case2
        - case3
'''

import md
import os
import numpy as np
from multiprocessing import Pool,Manager
import getopt
import sys
import re

def mkdir(path):

    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False

def convert_2_nii(input_path, output_path):
    num = 1
    
    if num > 0:        
        case_dir = input_path
        print(case_dir)
        # noinspection PyBroadException
        try:
            im, tags = md.read_dicom_series(case_dir)            
            save_dir = output_path
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            output_save_file = os.path.join(save_dir, 'image.nii.gz')
            axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            im.set_axes(axes)
            md.write_image(im, output_save_file)
        except Exception:
            print(num)
    num = num + 1


def read_dicom_contour(input_path, output_path):
    num = 1    
    if num > 0:
        #print(num)
        #print(case_name)        
        case_dir = input_path
        dicom_contour = md.read_dicom_rt_contours(case_dir)
        for key in dicom_contour:
            print(key)
            contour_name = key + '.nii.gz'
            save_case_dir = os.path.join(output_path, contour_name)
            md.write_image(dicom_contour[key], save_case_dir)
    num = num + 1

def printUsage():
	print ('''usage: DicomToNii-multi.py -i <input> -o <output>
       test.py --in=<input> --out=<output>''')

if __name__ == '__main__':
    inputarg=""
    outputarg=""
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hi:o:",["in=","out="])
            
        with open('./log/' + args[0], "r") as f:
            data1 = f.read()
            s1 = re.split('\n', data1)
            print(s1[0])
        input_dir  =  r"../data/dcm_base/Eval{}/".format(s1[0])
        output_dir = r"../data/nii_base/Eval{}/".format(s1[0])
        mkdir(output_dir)
        input_case,output_case = [],[]
        for case in os.listdir(input_dir):
            #index1 = case.find('-')
            #index2 = case.rfind('-')
            #name = case[index1+1:index2]
            name = case
            input_case.append(os.path.join(input_dir,case))
            output_case.append(os.path.join(output_dir,name))
            #mkdir(os.path.join(output_dir,name))
        # generate list of inputcase and corresponding outputcase
        
        process_pool = Pool(4)  # multiprocess = 4
        que = Manager().Queue()
        for i in range(len(input_case)):
            process_pool.apply_async(convert_2_nii,args=(input_case[i],output_case[i]))
            # 'apply_async' for non-blocking, process_pool.apply_async(function,args=(arg1,arg2))
        process_pool.close()
        process_pool.join()
        print('generate image processes finished')
        
        process_pool = Pool(4)  # multiprocess = 4
        que = Manager().Queue()
        for i in range(len(input_case)):
            process_pool.apply_async(read_dicom_contour,args=(input_case[i],output_case[i]))
            # 'apply_async' for non-blocking, process_pool.apply_async(function,args=(arg1,arg2))
        process_pool.close()
        process_pool.join()
        print('generate mask processes finished')
        
        print("**********************************************")    
        path=output_dir
        
        for root,dirs,files in os.walk(path):
            for name in files:
                if " " in name:
                    NewFileName=name.replace(" ",'');
                    print(NewFileName);
                    os.rename(os.path.join(root,name),os.path.join(root,NewFileName))

    except Exception as e:
        print('exception:')
        print(e)
        #print(e)
        with open('./log/prepare_data_status.file','w',encoding='utf-8') as f:
            f.write(str(e))
