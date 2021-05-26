# -*- coding: utf-8 -*-
 
from django.shortcuts import render, redirect
from django.views.decorators import csrf
from django import forms
from django.http import HttpResponse,JsonResponse,FileResponse,StreamingHttpResponse
from subprocess import call
import os
import re
from . import settings
from django.views.generic import View
import subprocess

def download_file(request):
    response = HttpResponse('click download model!')
    if request.POST:
        with open('./log/mode.file', "r") as f:
            data = f.read()
            s0 = re.split('/', data)
        with open('./log/taskid.file', "r") as f:
            data = f.read()
            s1 = re.split('\n', data)
        with open('./log/train_label.file', "r") as f:
            data = f.read()
            s2 = re.split('\n', data)
        #file = open("{}/nnUNet/{}/Task{}_{}/nnUNetTrainerV2_noDeepSupervision__nnUNetPlansv2.1/fold_0/model_best.model".format(os.getenv('RESULTS_FOLDER'), s0[2], s1[0], s2[0]), 'rb')
        file = open("{}/compare/Task{}_{}/testdice_volume.xls".format(os.getenv('nnUNet_raw_data_base'), s1[0], s2[0]), 'rb')
        #file = open('/home/data/nnunet/nnUNet_trained_models/nnUNet/3d_fullres/Task155_LungL/nnUNetTrainerV2_noDeepSupervision__nnUNetPlansv2.1/fold_0/model_best.model', 'rb')
        response = FileResponse(file)
        response['Content-Type'] = 'application/octet-stream'
        #response['Content-Disposition'] = 'attachment;filename="Task{}.pth"'.format(s1[0])
        response['Content-Disposition'] = 'attachment;filename="Task{}_Dice.csv"'.format(s1[0])
        
        #file = open('/home/data/nnunet/nnUNet_trained_models/nnUNet/3d_fullres/Task155_LungL/nnUNetTrainerV2_noDeepSupervision__nnUNetPlansv2.1/fold_0/model_best.model', 'rb')
        #response = FileResponse(file)
        #response['Content-Type'] = 'application/octet-stream'
        #response['Content-Disposition'] = 'attachment;filename="Task{}.pth"'.format(s1[0])

    return response

def download_file_eval(request):
    response = HttpResponse('click download model!')
    if request.POST:
        with open('./log/mode.file', "r") as f:
            data = f.read()
            s0 = re.split('/', data)
        with open('./log/evaluation_id.file', "r") as f:
            data = f.read()
            s1 = re.split('\n', data)
        with open('./log/train_label.file', "r") as f:
            data = f.read()
            s2 = re.split('\n', data)
        #file = open("{}/nnUNet/{}/Task{}_{}/nnUNetTrainerV2_noDeepSupervision__nnUNetPlansv2.1/fold_0/model_best.model".format(os.getenv('RESULTS_FOLDER'), s0[2], s1[0], s2[0]), 'rb')
        file = open("{}/compare/Eval{}_{}/testdice_volume.csv".format(os.getenv('nnUNet_raw_data_base'), s1[0], s2[0]), 'rb')
        print(file)
        #file = open('/home/data/nnunet/nnUNet_trained_models/nnUNet/3d_fullres/Task155_LungL/nnUNetTrainerV2_noDeepSupervision__nnUNetPlansv2.1/fold_0/model_best.model', 'rb')
        response = FileResponse(file)
        response['Content-Type'] = 'application/octet-stream'
        #response['Content-Disposition'] = 'attachment;filename="Task{}.pth"'.format(s1[0])
        response['Content-Disposition'] = 'attachment;filename="Eval_{}_Dice.csv"'.format(s2[0])
        
        #file = open('/home/data/nnunet/nnUNet_trained_models/nnUNet/3d_fullres/Task155_LungL/nnUNetTrainerV2_noDeepSupervision__nnUNetPlansv2.1/fold_0/model_best.model', 'rb')
        #response = FileResponse(file)
        #response['Content-Type'] = 'application/octet-stream'
        #response['Content-Disposition'] = 'attachment;filename="Task{}.pth"'.format(s1[0])

    return response
    
'''
def upload(request):
    file = request.FILES  
    upload_status = {}
    upload_status['views_str'] = "<a href='../'>Return</a>"
    upload_status['upload_status'] = "Upload Done!"
    print(file)
    if request.method == "POST":
        myFile=request.FILES.get("file",None)
        print(myFile)
        if not myFile:
            return HttpResponse("no files for upload!")
        destination = open(os.path.join("../upload/", myFile.name),'wb+')
        for chunk in myFile.chunks():
            destination.write(chunk)
        destination.close()
    #return HttpResponse("upload done!")
    return render(request,'upload_status.html', upload_status)

def upload_files(request):
    upload_status = {}
    upload_status['views_str'] = "<a href='../'>Return</a>"
    return render(request,'upload_page.html', upload_status)
'''

def upload(request):
    file = request.FILES  
    if request.method == "POST":
        myFile=request.FILES.get("file",None)
        print(myFile)
        if not myFile:
            return HttpResponse("no files for upload!")
        destination = open(os.path.join("../upload/", myFile.name),'wb+')
        for chunk in myFile.chunks():
            destination.write(chunk)
        destination.close()
        with open('./log/unzip.file','w',encoding='utf-8') as f:
            f.write(str(myFile))
    return HttpResponse("upload done!")

def upload_files(request):
    upload_status = {}
    upload_status['views_str'] = "<a href='../'>Return</a>"
    return render(request,'upload_page.html',upload_status)

def upload_files_eval(request):
    upload_status = {}
    upload_status['views_str'] = "<a href='../'>Return</a>"
    return render(request,'upload_page_eval.html',upload_status)
# form
# retrun
#def search(request):  
#    request.encoding='utf-8'
#    if 'q' in request.GET and request.GET['q']:
#        message = 'your input is: ' + request.GET['q']
#    else:
#        message = 'empty input'
#    return HttpResponse(message)
    
#os.system('sh /home/zhuoli/data/train.sh>/home/zhuoli/data/webout.file 2>&1 &')

def check_existing_id(request):
    ids ={}
    if request.POST:
        ids['ids'] = os.listdir('{}/nnUNet_raw_data/'.format(os.getenv('nnUNet_raw_data_base')))
        id_array = [1]
        for i in range(len(ids['ids'])):
            if 'Task' in ids['ids'][i]:
                id_label = re.split('_', ids['ids'][i])
                id_array.append(int(id_label[0][4:]))
        with open('./log/max_id.file','w',encoding='utf-8') as f:
            f.write(str(max(id_array)))
    return render(request, "search_form.html", ids)

def check_existing_id_eval(request):
    ids ={}
    if request.POST:
        ids['ids'] = os.listdir('{}/nnUNet_raw_data/'.format(os.getenv('nnUNet_raw_data_base')))
        id_array = [1]
        for i in range(len(ids['ids'])):
            if 'Eval' in ids['ids'][i]:
                id_label = re.split('_', ids['ids'][i])
                id_array.append(int(id_label[0][4:]))
        with open('./log/max_id_eval.file','w',encoding='utf-8') as f:
            f.write(str(max(id_array)))
    return render(request, "evaluation.html", ids)
    
def check_label(request):
    top_labels ={}
    if request.POST:
        os.system('rm ./log/labels.txt')
        os.system('python3 ./print_labels.py taskid.file train_label.file')
        with open('./log/labels.txt', "r") as f:
            data = f.read()
            s1 = re.split('\n', data)
            if (len(s1)>10):
                top_labels['tlabel'] = s1[1:20]
            else:
                top_labels['tlabel'] = s1[1:]
    return render(request, "search_form.html", top_labels)

def choose_label(request):
    label_in ={}
    if request.POST:
        label_in['label'] = request.POST['q1']
        with open('./log/train_label.file','w',encoding='utf-8') as f:
            text = str(label_in['label'])
            f.write(text)
    return render(request, "search_form.html", label_in)


def rename(request):
    rename_in ={}
    if request.POST:
        rename_in['rename_in'] = request.POST['q1']
        with open('./log/rename.file','w',encoding='utf-8') as f:
            text = str(rename_in['rename_in'])
            f.write(text)
    return render(request, "search_form.html", rename_in)
    
def input_id(request):
    ctx ={}
    if request.POST:
        ctx['id'] = request.POST['q0']
        print(ctx['id'])
        with open('./log/taskid.file','w',encoding='utf-8') as f:
            text = str(ctx['id'])
            f.write(text)
    return render(request, "search_form.html", ctx)
    
def input_id_eval(request):
    ctx ={}
    if request.POST:
        ctx['id'] = request.POST['q0']
        print(ctx['id'])
        with open('./log/taskid.file','w',encoding='utf-8') as f:
            text = str(ctx['id'])
            f.write(text)
        
    ids = os.listdir('{}/nnUNet_raw_data/'.format(os.getenv('nnUNet_raw_data_base')))
    for i in range(len(ids)):
        if ctx['id']==re.split('_', ids[i])[0][4:]:
            trained_label=re.split('_', ids[i])[1]
    with open('./log/train_label.file','w',encoding='utf-8') as f:
        text = trained_label
        f.write(text)
    return render(request, "evaluation.html", ctx)

def mode(request):
    mode_in ={}
    if request.POST:
        #mode_in['mode'] = request.POST['q2']
        with open('./log/mode.file','w',encoding='utf-8') as f:
            text = str(request)
            f.write(text)
        with open('./log/train_label.file','r',encoding='utf-8') as f:
            label = f.read()
            mode_in['label'] = str(label)
        with open('./log/taskid.file','r',encoding='utf-8') as f:
            id = f.read()
            mode_in['id'] = str(id)
        with open('./log/mode.file','r',encoding='utf-8') as f:
            mode_all = f.read()
            s_mode = re.split('/', mode_all)
            mode_in['mode'] = str(s_mode[2])
    return render(request, "search_form.html", mode_in)

def mode_eval(request):
    mode_in ={}
    if request.POST:
        #mode_in['mode'] = request.POST['q2']
        with open('./log/mode.file','w',encoding='utf-8') as f:
            text = str(request)
            f.write(text)
        with open('./log/train_label.file','r',encoding='utf-8') as f:
            label = f.read()
            mode_in['label'] = str(label)
        with open('./log/taskid.file','r',encoding='utf-8') as f:
            id = f.read()
            mode_in['id'] = str(id)
        with open('./log/mode.file','r',encoding='utf-8') as f:
            mode_all = f.read()
            s_mode = re.split('/', mode_all)
            mode_in['mode'] = str(s_mode[2])
    return render(request, "evaluation.html", mode_in)


def overview(request):
    return render(request, "search_form.html")
    
def maybe_mkdir_p(path):
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False
        
def prepare_data(request):
    prepare_status = {}

    if request.POST:
        try:
            with open('./log/max_id.file', "r") as f:
                data4 = f.read()
                s4 = re.split('\n', data4)
                print(s4[0])
            with open('./log/taskid.file','w',encoding='utf-8') as f:
                f.write(str(int(s4[0])+1))
            with open('./log/prepare_data_status.file','w',encoding='utf-8') as f:
                f.write('Preparing data, please wait 30 minutes')            
            with open('./log/taskid.file', "r") as f:
                data1 = f.read()
                s1 = re.split('\n', data1)
                print(s1[0])
            with open('./log/train_label.file', "r") as f:
                data2 = f.read()
                s2 = re.split('\n', data2)
                print(s2[0])
                
            with open('./log/unzip.file', "r") as f:
                data3 = f.read()
                s3 = re.split('\.', data3)
                print(s3[0])
            print('base dir')
            print(settings.BASE_DIR)
            maybe_mkdir_p(r'../data/dcm_base/Task{}'.format(int(s4[0])+1))
            #os.system('tar -xvf ../upload/Task{}.tar -C ../data/dcm_base'.format(s1[0]))

            os.system('chmod +x prepare_data.sh')
            subprocess.Popen("bash prepare_data.sh {} {} > ./log/prepare_data.file 2>&1 &".format(s3[0],int(s1[0])), shell=True)
            #os.system('unzip -o ../upload/Task{}.zip -d ../data/dcm_base'.format(s1[0]))
            #os.system('python3 ./DicomToNii-multi.py taskid.file')
            #os.system('bash prepare_data.sh {} {} {}'.format(settings.BASE_DIR, s1[0], s1[1]))
            #run_script.delay()
            print('preparing data')
            
        except Exception as e:
            print('exception:')
            print(e)
            with open('./log/prepare_data_status.file','w',encoding='utf-8') as f:
                f.write(str(e))
            with open('./log/prepare_data_status.file', "r") as f:
                data1 = f.read()
                s0 = re.split('\n', data1)
                prepare_status['prepare_status']=s0
                
    with open('./log/prepare_data_status.file', "r") as f:
        data1 = f.read()
        s0 = re.split('\n', data1)
        prepare_status['prepare_status']=s0[0]
    return render(request, "search_form.html", prepare_status)

def prepare_data_eval(request):
    prepare_status = {}

    if request.POST:
        try:
            with open('./log/max_id_eval.file', "r") as f:
                data4 = f.read()
                s4 = re.split('\n', data4)
                print(s4[0])
            with open('./log/evaluation_id.file','w',encoding='utf-8') as f:
                f.write(str(int(s4[0])+1))
            with open('./log/prepare_data_status.file','w',encoding='utf-8') as f:
                f.write('Preparing data, please wait 30 minutes')            
            with open('./log/taskid.file', "r") as f:
                data1 = f.read()
                s1 = re.split('\n', data1)
                print(s1[0])
            with open('./log/train_label.file', "r") as f:
                data2 = f.read()
                s2 = re.split('\n', data2)
                print(s2[0])
                
            with open('./log/unzip.file', "r") as f:
                data3 = f.read()
                s3 = re.split('\.', data3)
                print(s3[0])
            print('base dir')
            print(settings.BASE_DIR)
            maybe_mkdir_p(r'../data/dcm_base/Eval{}'.format(int(s4[0])+1))
            os.system('chmod +x prepare_data.sh')
            subprocess.Popen("bash prepare_data_eval.sh {} {} > ./log/evaluation.file 2>&1 &".format(s3[0],int(s4[0])+1), shell=True)
            print('preparing data')
            
        except Exception as e:
            print('exception:')
            print(e)
            with open('./log/evaluation_status.file','w',encoding='utf-8') as f:
                f.write(str(e))
            with open('./log/evaluation_status.file', "r") as f:
                data1 = f.read()
                s0 = re.split('\n', data1)
                prepare_status['prepare_status']=s0
                
    with open('./log/evaluation_status.file', "r") as f:
        data1 = f.read()
        s0 = re.split('\n', data1)
        prepare_status['prepare_status']=s0[0]
    return render(request, "evaluation.html", prepare_status)
    
def read_log(url,keyword):
    count = 0
    with open(url,'r',encoding='utf-8') as f:    
        data = f.read()
        test_txt = re.findall(keyword, data)
    return test_txt
    
def plan(request):
    planned = {}
    
    with open('./log/mode.file', "r") as f:
        data = f.read()
        s0 = re.split('/', data)
    with open('./log/taskid.file', "r") as f:
        data = f.read()
        s1 = re.split('\n', data)
    with open('./log/train_label.file', "r") as f:
        data = f.read()
        s2 = re.split('\n', data)
        
    if request.POST:
        #os.system('python3 ./Task121_BreCW.py taskid.file train_label.file')
        os.system('bash prepare_data.sh {} {} {}'.format(settings.BASE_DIR, s1[0], s1[1]))
        p_plan = subprocess.Popen("nohup nnUNet_plan_and_preprocess -t {} > ./log/plan.file 2>&1 &".format(int(s1[0])), shell=True)
        #os.system('nohup nnUNet_plan_and_preprocess -t {} > ./log/plan.file 2>&1 &'.format(int(s1[0])))
    tr_file_num = len(os.listdir('{}/nnUNet_raw_data/Task{}_{}/imagesTr/'.format(os.getenv('nnUNet_raw_data_base'),s1[0],s2[0])))
    ts_file_num = len(os.listdir('{}/nnUNet_raw_data/Task{}_{}/imagesTs/'.format(os.getenv('nnUNet_raw_data_base'),s1[0],s2[0])))
    test_txt = read_log('./log/plan.file', r'saving')
    count = len(test_txt)
    planned['planned_cases'] = count
    planned['file_num'] = ' / ' + str(tr_file_num + ts_file_num)
    #views_str = "<a href='http://10.8.77.18:8000/progress'>Training progress</a>"
    return render(request, "search_form.html", planned)

def get_process_id(name):
    """Return process ids found by (partial) name or regex.
 
    >>> get_process_id('kthreadd')
    [2]
    >>> get_process_id('watchdog')
    [10, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61]  # ymmv
    >>> get_process_id('non-existent process')
    []
    """
    child = subprocess.Popen(['pgrep', '-f', name], stdout=subprocess.PIPE, shell=False)
    response = child.communicate()[0]
    return [int(pid) for pid in response.split()]
    
def train(request):
    trained = {}
    mode_in = {}
    

        
    with open('./log/mode.file', "r") as f:
        data = f.read()
        s0 = re.split('/', data)
    with open('./log/taskid.file', "r") as f:
        data = f.read()
        s1 = re.split('\n', data)
    with open('./log/train_label.file', "r") as f:
        data = f.read()
        s2 = re.split('\n', data)
    
    if request.POST:
      os.system('chmod +x train.sh')

      subprocess.Popen("bash train.sh {} {} {} {}> ./log/train.file 2>&1 &".format(int(s1[0]),s2[0],s0[2],os.getenv('nnUNet_raw_data_base')), shell=True)
      #os.system('python3 ./Task121_BreCW.py taskid.file train_label.file')
      #os.system('python3 ./Task121_BreCW.py taskid.file train_label.file')
      #print('start planning')
      #p_plan = subprocess.Popen("nnUNet_plan_and_preprocess -t {} > ./log/plan.file 2>&1 &".format(int(s1[0])), shell=True)      
      #with open('./log/training_status.file','w',encoding='utf-8') as f:
      #    f.write('planning\n')
      #os.system('nnUNet_plan_and_preprocess -t {}'.format(int(s1[0])))
    
    '''
    with open('./log/mode.file', "r") as f:
        data = f.read()
        s0 = re.split('/', data)
        print(s0[2])
    with open('./log/taskid.file', "r") as f:
        data = f.read()
        s1 = re.split('\n', data)
        print(s1[0])
    with open('./log/train_label.file', "r") as f:
        data = f.read()
        s2 = re.split('\n', data)
    
    
    with open('./log/train_label.file','r',encoding='utf-8') as f:
        label = f.read()
        mode_in['label'] = str(label)
    with open('./log/taskid.file','r',encoding='utf-8') as f:
        id = f.read()
        mode_in['id'] = str(id)
    with open('./log/mode.file','r',encoding='utf-8') as f:
        mode_all = f.read()
        s_mode = re.split('/', mode_all)
        mode_in['mode'] = str(s_mode[2])
    
    
    if request.POST:
        with open('./log/training_status.file','w',encoding='utf-8') as f:
            f.write('training\n')
        if os.path.exists('{}/nnUNet/{}/Task{}_{}/nnUNetTrainerV2_noDeepSupervision__nnUNetPlansv2.1/fold_0/model_best.model'.format(os.getenv('RESULTS_FOLDER'),s0[2],s1[0],s2[0])):
            #p_train = subprocess.Popen('nnUNet_train -c {} nnUNetTrainerV2_noDeepSupervision {} 0 --npz > ./log/train.file 2>&1 &'.format(s0[2],int(s1[0])), shell=True)
            os.system('nnUNet_train -c {} nnUNetTrainerV2_noDeepSupervision {} 0 --npz'.format(s0[2],int(s1[0])))
            print('continue training')
            #pid_ = p_train.pid + 1
            trained['button'] = 'stop'
            with open('./log/training_flag.file','w',encoding='utf-8') as f:
                f.write('stop\n')
        else:
            #p_train = subprocess.Popen('nnUNet_train {} nnUNetTrainerV2_noDeepSupervision {} 0 --npz > ./log/train.file 2>&1 &'.format(s0[2],int(s1[0])), shell=True)
            os.system('nnUNet_train {} nnUNetTrainerV2_noDeepSupervision {} 0 --npz'.format(s0[2],int(s1[0])))
            #pid_ = p_train.pid + 1
            flag = 'stop'
            trained['button'] = 'stop'
            with open('./log/training_flag.file','w',encoding='utf-8') as f:
                f.write('stop\n')
                #f.write(str(pid_))
        
    with open('./log/mode.file', "r") as f:
        data = f.read()
        s0 = re.split('/', data)
    with open('./log/taskid.file', "r") as f:
        data = f.read()
        s1 = re.split('\n', data)
    with open('./log/train_label.file', "r") as f:
        data = f.read()
        s2 = re.split('\n', data)
                
        
    if request.POST:
        print("start prediction")
        with open('./log/training_status.file','w',encoding='utf-8') as f:
            f.write('predicting\n')
        #p_predict = subprocess.Popen('nnUNet_predict -i {}/nnUNet_raw_data/Task{}_{}/imagesTs/ -o {}/result/Task{}_{} -t {} -m {} -f 0 -tr nnUNetTrainerV2_noDeepSupervision --disable_tta -chk model_best > ./log/predict.file 2>&1 &'.format(os.getenv('nnUNet_raw_data_base'),s1[0],s2[0],os.getenv('nnUNet_raw_data_base'),s1[0],s2[0],s1[0],s0[2]))
        os.system('nnUNet_predict -i {}/nnUNet_raw_data/Task{}_{}/imagesTs/ -o {}/result/Task{}_{} -t {} -m {} -f 0 -tr nnUNetTrainerV2_noDeepSupervision --disable_tta -chk model_best'.format(os.getenv('nnUNet_raw_data_base'),s1[0],s2[0],os.getenv('nnUNet_raw_data_base'),s1[0],s2[0],s1[0],s0[2]))
    epoch = read_log('./log/train.file', r'epoch: .{4}')
    trained['epoch'] = epoch
    trained['total_epoch'] = ' / '+str(250)
    '''
    with open('./log/training_status.file', "r") as f:
        data = f.read()
        s2 = re.split('\n', data)
    trained['train_status'] = s2[0]
        
    views_str = "<a href='./progress'>Training progress</a>"
    trained['views_str'] = views_str
    return render(request, "search_form.html", trained)

def stop_training(request):
    with open('./log/training_flag.file', "r") as f:
        data = f.read()
        sf = re.split('\n', data)
        flag = sf[0]
        
    if request.POST:
        #p_train.kill()
        process_ids = get_process_id("nnUNet_plan")

        for pid in range(len(process_ids)):
            os.system("kill -9 {}".format(process_ids[pid]))
            
        process_ids = get_process_id("nnUNet_train")

        for pid in range(len(process_ids)):
            os.system("kill -9 {}".format(process_ids[pid]))
            
        process_ids = get_process_id("nnUNet_predict")

        for pid in range(len(process_ids)):
            os.system("kill -9 {}".format(process_ids[pid]))
            
        flag = 'train'

        with open('./log/training_flag.file','w',encoding='utf-8') as f:
            f.write('train\n')
            #f.write(str(pid_))
        #os.system('nohup nnUNet_train {} nnUNetTrainerV2_noDeepSupervision {} 0 --npz > ./log/train.file 2>&1 &'.format(s0[2],int(s1[0])))    
    return render(request, "search_form.html")

def predict(request):
    predicted ={}
    with open('./log/mode.file', "r") as f:
        data = f.read()
        s0 = re.split('/', data)
    with open('./log/taskid.file', "r") as f:
        data = f.read()
        s1 = re.split('\n', data)
    with open('./log/train_label.file', "r") as f:
        data = f.read()
        s2 = re.split('\n', data)
        
    if request.POST:
        os.system('nohup nnUNet_predict -i {}/nnUNet_raw_data/Task{}_{}/imagesTs/ -o {}/result/Task{}_{} -t {} -m {} -f 0 -tr nnUNetTrainerV2_noDeepSupervision --disable_tta -chk model_best > ./log/predict.file 2>&1 &'.format(os.getenv('nnUNet_raw_data_base'),s1[0],s2[0],os.getenv('nnUNet_raw_data_base'),s1[0],s2[0],s1[0],s0[2]))

        #print('number of cases predicted')
    cases = read_log('./log/predict.file', r'number of cases:. {4}')
    #count = len(os.listdir('{}/result/Task{}_{}/'.format(os.getenv('nnUNet_raw_data_base'),s1[0],s2[0])))
    #predicted['num_predicted'] = count
    #ts_image_num = len(os.listdir('{}/nnUNet_raw_data/Task{}_{}/imagesTs/'.format(os.getenv('nnUNet_raw_data_base'),s1[0],s2[0])))
    #predicted['ts_image']= ' / '+ str(ts_image_num+1)
    return render(request, "search_form.html")
    
def evaluate(request):
    views_str = ''
    if request.POST:
        #os.system('sh /home/zhuoli/data/train.sh>/home/zhuoli/data/webout.file 2>&1 &')
        os.system('python3 ./eval_DL_result.py taskid.file train_label.file')
        with open('./log/dice.file', "r") as f:
            data = f.read()
            views_str = 'average dice is: ' + str(data)
    return render(request, "search_form.html", {"views_str": views_str})
    
def predict_eval(request):
    views_str = ''
    if request.POST:
        os.system('python3 ./predict.py evaluation_id.file train_label.file mode.file taskid.file')
        with open('./log/dice.file', "r") as f:
            data = f.read()
            views_str = 'average dice is: ' + str(data)
    return render(request, "evaluation.html", {"views_str": views_str})

def eval_mode(request):
    views_str = ''
    return render(request, "evaluation.html", {"views_str": views_str})
    
def check_result(request):
    if request.POST:
        os.system('python3 ./WebApp/generate_html.py')
    return render(request,"images.html")

def check_result_eval(request):
    if request.POST:
        os.system('python3 ./WebApp/generate_html_eval.py')
    return render(request,"images.html")

num_progress = 0 
 
 
