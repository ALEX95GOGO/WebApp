#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from collections import OrderedDict
from nnunet.paths import nnUNet_raw_data
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import SimpleITK as sitk
import re
import os
from tqdm import tqdm
import getopt
import sys
import random
import time
    
def maybe_mkdir_p(path):
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False
        
def convert_for_submission(source_dir, target_dir):
    """
    I believe they want .nii, not .nii.gz
    :param source_dir:
    :param target_dir:
    :return:
    """
    files = subfiles(source_dir, suffix=".nii.gz", join=False)
    maybe_mkdir_p(target_dir)
    for f in files:
        img = sitk.ReadImage(join(source_dir, f))
        out_file = join(target_dir, f[:-7] + ".nii")
        sitk.WriteImage(img, out_file)

def printUsage():
	print ('''usage: Task121_BreCW.py -i <input> -o <output>
       test.py --in=<input> --out=<output>''')

if __name__ == "__main__":
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
    print ('other param:'+",".join(args))

    ids = os.listdir('{}/nnUNet_raw_data/'.format(os.getenv('nnUNet_raw_data_base')))
    id_array = [1]
    label_array = []
    trained_label = ''
    s2 = [0]
    
    with open('./log/' + args[3], "r") as f:
        data4 = f.read()
        s4 = re.split('\n', data4)
        print(s4[0])
        
    with open('./log/' + args[0], "r") as f:
        data1 = f.read()
        s1 = re.split('\n', data1)
        print(s1[0])
        
    for i in range(len(ids)):
        id_label = re.split('_', ids[i])
        id_array.append(id_label[0])
        label_array.append(id_label[1])
        if id_label[0] == 'Task{}'.format(s4[0]):
            trained_label = id_label[1]
    print(trained_label)
    with open('./log/train_label.file','w',encoding='utf-8') as f:
        f.write(trained_label)
    print(id_array)

    #s2[0] = trained_label
    #print(s2[0])
    with open('./log/' + args[1], "r") as f:
        data2 = f.read()
        s2 = re.split('\n', data2)
        print(s2[0])
    
    with open('./log/' + args[2], "r") as f:
        data3 = f.read()
        s3 = re.split('\n', data3)
        print(s3[0])
    
    
    nnUNet_raw_data = os.path.join(os.getenv('nnUNet_raw_data_base'),'nnUNet_raw_data')
    task_id = int(s1[0])
    task_name = str(s2[0])
    base = r'../data/nii_base/Eval{}/'.format(s1[0])

    foldername = "Eval{}_{}".format(task_id, trained_label)

    out_base = join(nnUNet_raw_data, foldername)
    maybe_mkdir_p(out_base)
    imagests = join(out_base, "imagesTs")
    labelsts = join(out_base, "labelsTs")
    
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelsts)

    test_patient_names = []
    
    folder_list = os.listdir(base)

    all_patients = subfolders(base, join=False)
    
    for p in tqdm(all_patients,ncols=50):
        old_time = time.perf_counter()
        name_p = re.sub("\D", "", p)
        name = "%06.0d" % int(name_p)
        #curr = join(base, "train", p)
        curr = join(base, p)
        if os.path.exists(os.path.join(curr, "{}.nii.gz".format(task_name))):
            label_file = join(curr, "{}.nii.gz".format(task_name))
            image_file = join(curr, "image.nii.gz")
            sitkimage = sitk.ReadImage(image_file)
            sitklabel = sitk.ReadImage(label_file)
            if sitkimage.GetSize() != sitklabel.GetSize():
                all_patients.remove(p)
        time_per_loop = time.perf_counter() - old_time
        estimate_time = time_per_loop * len(all_patients)
    print ('Assert image and label size to be the same')


    test_patients = all_patients
    for p in tqdm(test_patients,ncols=50):
        name_p = re.sub("\D", "", p)
        name = "%06.0d" % int(name_p)
        #curr = join(base, "test", p)
        curr = join(base, p)
        if os.path.exists(os.path.join(curr, "{}.nii.gz".format(task_name))):

            label_file = join(curr, "{}.nii.gz".format(task_name))
            image_file = join(curr, "image.nii.gz")

            shutil.copy(image_file, join(imagests, trained_label +'_'+ name + "_0000.nii.gz"))
            shutil.copy(label_file, join(labelsts, trained_label+ '_'+ name +".nii.gz"))
            patient_name = trained_label+ '_'+ name
            test_patient_names.append(patient_name)

    print ('Test set copy done')
    
    json_dict = OrderedDict()
    json_dict['name'] = ""
    json_dict['description'] = ""
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see challenge website"
    json_dict['licence'] = "see challenge website"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "mask",
    }
    json_dict['numTest'] = len(test_patient_names)

    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i.split("/")[-1] for i in test_patient_names]

    save_json(json_dict, os.path.join(out_base, "dataset.json"))
    
    
    