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
    
    nnUNet_raw_data = os.path.join(os.getenv('nnUNet_raw_data_base'),'nnUNet_raw_data')
    #nnUNet_raw_data = "../../nnUNet_data/nnUNet_raw/nnUNet_raw_data/"
    task_id = int(s1[0])
    task_name = str(s2[0])
    base = r'./data/nii_base/Task{}/'.format(task_id)
    #base = "./WebApp/upload/" + s1[0] + '_' + s2[0] + '/'
    #base = "H:/breast_k/cropped_data"


    foldername = "Task%03.0d_%s" % (task_id, task_name)

    out_base = join(nnUNet_raw_data, foldername)
    maybe_mkdir_p(out_base)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    labelsts = join(out_base, "labelsTs")
    
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(labelsts)
    
    #labels_clip = join(out_base, "labelsTs_clip")
    #maybe_mkdir_p(labels_clip)
    
    train_patient_names = []
    test_patient_names = []
    
    folder_list = os.listdir(base)

    k = len(folder_list) // 5  # number of test images
    
    all_patients = subfolders(base, join=False)
    random.shuffle(all_patients)
    
    for p in tqdm(all_patients,ncols=50):
        name_p = re.sub("\D", "", p)
        name = "%06.0d" % int(name_p)
        #curr = join(base, "train", p)
        curr = join(base, p)
        if os.path.exists(os.path.join(curr, "{}.nii.gz".format(task_name))):
            label_file = join(curr, "{}.nii.gz".format(task_name))
            image_file = join(curr, "image.nii.gz")
            sitkimage = sitk.ReadImage(image_file)
            image = sitk.GetArrayFromImage(sitkimage)
            sitklabel = sitk.ReadImage(label_file)
            label = sitk.GetArrayFromImage(sitklabel)
            if image.shape != label.shape:
                all_patients.remove(p)
    print ('Assert image and label size to be the same')

    train_patients = all_patients[index[k:-1]]
    #train_patients = subfolders(join(base, "train"), join=False)
    

    for p in tqdm(train_patients,ncols=50):
        name_p = re.sub("\D", "", p)
        name = "%06.0d" % int(name_p)
        #curr = join(base, "train", p)
        curr = join(base, p)
        if os.path.exists(os.path.join(curr, "{}.nii.gz".format(task_name))):
            label_file = join(curr, "{}.nii.gz".format(task_name))
            image_file = join(curr, "image.nii.gz")
            shutil.copy(image_file, join(imagestr, task_name+ '_'+ name +"_0000.nii.gz"))
            shutil.copy(label_file, join(labelstr, task_name+ '_'+ name +".nii.gz"))
            patient_name = task_name+ '_'+ name
            train_patient_names.append(patient_name)
    print ('Train set copy done')
    
    #test_patients = subfolders(join(base, "test"), join=False)
    test_patients = all_patients[index[0:k]]
    for p in tqdm(test_patients,ncols=50):
        name_p = re.sub("\D", "", p)
        name = "%06.0d" % int(name_p)
        #curr = join(base, "test", p)
        curr = join(base, p)
        if os.path.exists(os.path.join(curr, "{}.nii.gz".format(task_name))):

            label_file = join(curr, "{}.nii.gz".format(task_name))
            image_file = join(curr, "image.nii.gz")

            shutil.copy(image_file, join(imagests, task_name +'_'+ name + "_0000.nii.gz"))
            shutil.copy(label_file, join(labelsts, task_name+ '_'+ name +".nii.gz"))
        #clip_file = join(curr, "GTV-C.nii.gz")
        #shutil.copy(clip_file, join(labels_clip, task_name+ '_'+ name +".nii.gz"))
            patient_name = task_name+ '_'+ name
            test_patient_names.append(patient_name)

    print ('Test set copy done')
    
    json_dict = OrderedDict()
    json_dict['name'] = "Pelvis"
    json_dict['description'] = "PelvisMultiLabel"
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
        #"2": "lp",
        #"3": "ls",
    }
    json_dict['numTraining'] = len(train_patient_names)
    json_dict['numTest'] = len(test_patient_names)
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i.split("/")[-1], "label": "./labelsTr/%s.nii.gz" % i.split("/")[-1]} for i in
                             train_patient_names]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i.split("/")[-1] for i in test_patient_names]

    save_json(json_dict, os.path.join(out_base, "dataset.json"))
    
    with open('./log/prepare_data.file','w',encoding='utf-8') as f:
            text = 'success'
            f.write(text)
    
    