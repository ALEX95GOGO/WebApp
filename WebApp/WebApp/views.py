# -*- coding: utf-8 -*-
 
from django.shortcuts import render, redirect
from django.views.decorators import csrf
from django import forms
from django.http import HttpResponse,JsonResponse
from disk.models import User
from subprocess import call
import os
import re
from . import settings
from WebApp.tasks import run_script
from django.views.generic import View


class UserForm(forms.Form):
    username = forms.CharField()
    headImg = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': True}))
    
def upload(request):
    file = request.FILES  
    if request.method == "POST":
        myFile=request.FILES.get("file",None)
        print(myFile)
        if not myFile:
            return HttpResponse("no files for upload!")
        destination = open(os.path.join("./upload/", myFile.name),'wb+')
        for chunk in myFile.chunks():
            destination.write(chunk)
        destination.close()
    return HttpResponse("upload done!")

def upload_files(request):
    return render(request,'upload_page.html')
    
def get_media_upload_to(instance, filename):
    """
    A simple way to return the full path      
    """
    paths = { 'I':'images/', 'V':'videos/', 'A':'audio/', 'D':'documents/' }
    return settings.MEDIA_ROOT + 'content/' + paths[instance.content_type] + filename

def register(request):
    if request.method == "POST":
        uf = UserForm(request.POST,request.FILES)

        if uf.is_valid():
            #ger form info
            username = uf.cleaned_data['username']
            headImg = uf.cleaned_data['headImg']
            print(headImg)
            #write to the database
            user = User()
            user.username = username
            user.headImg = headImg
            user.save()
            return HttpResponse('upload ok!')
    else:
        uf = UserForm()
    return render(request,'register.html',{'uf':uf})

    
def progress(request):
    views_str = "<a href='http://10.8.77.18:8000/progress'>Training progress</a>"
    return render(request, "search_form.html", {"views_str": views_str})

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
        ids['ids'] = os.listdir('/home/zhuoli.zhuang/nnUNet_data/nnUNet_raw/nnUNet_raw_data/')
    return render(request, "search_form.html", ids)
    
def check_label(request):
    top_labels ={}
    if request.POST:
        os.system('python3 ./print_labels.py taskid.file train_label.file')
        with open('./log/labels.txt', "r") as f:
            data = f.read()
            s1 = re.split('\n', data)
            if (len(s1)>10):
                top_labels['tlabel'] = s1[1:10]
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
    
def input_id(request):
    ctx ={}
    if request.POST:
        ctx['id'] = request.POST['q0']
        print(ctx['id'])
        with open('./log/taskid.file','w',encoding='utf-8') as f:
            text = str(ctx['id'])
            f.write(text)
    return render(request, "search_form.html", ctx)

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
    
#def task_name(request):
#    ctx_name ={}
#    if request.POST:
#        ctx_name['rlt_name'] = request.POST['q']
#        print(ctx_name['rlt_name'])
#        with open('./log/taskname.file','w',encoding='utf-8') as f:
#            text = str(ctx_name['rlt_name'])
#            f.write(text)
#
#    return render(request, "search_form.html", ctx_name)

def overview(request):
        #os.system('sh /home/zhuoli/data/train.sh>/home/zhuoli/data/webout.file 2>&1 &')
    #views_str = "<a href='http://10.8.77.18:8000/progress'>Training progress</a>"
    return render(request, "search_form.html")
    
def import_data(request):
    if request.POST:
        return redirect("/disk/")
        #os.system('sh /home/zhuoli/data/train.sh>/home/zhuoli/data/webout.file 2>&1 &')
    #views_str = "<a href='http://10.8.77.18:8000/progress'>Training progress</a>"
    return render(request, "search_form.html")
    
def prepare_data(request):
    if request.POST:
        #os.system('bash nnunet_env.sh')
        with open('./log/taskid.file', "r") as f:
            data1 = f.read()
            s1 = re.split('\n', data1)
            print(s1[0])
        with open('./log/train_label.file', "r") as f:
            data2 = f.read()
            s2 = re.split('\n', data2)
            print(s2[0])
        print('base dir')
        print(settings.BASE_DIR)
        #os.system('cp {}/upload/{}_{}.tar /home/img/data/'.format(settings.BASE_DIR,s1[0],s1[1]))
        #os.system('tar -xvf /home/img/data/{}_{}.tar'.format(s1[0],s1[1]) )
        #os.system('tar -xvf ./upload/Task{}.tar -C ./data/dcm_base'.format(s1[0]))
        #os.system('python3 ./DicomToNii-multi.py taskid.file')
        #os.system('python3 ./prepare_data.py taskid.file train_label.file')
        os.system('python3 ./Task121_BreCW.py taskid.file train_label.file')
        #os.system('bash prepare_data.sh {} {} {}'.format(settings.BASE_DIR, s1[0], s1[1]))
        #run_script.delay()
        print('preparing data')
    views_str = "<a href='http://10.8.77.18:8000/progress'>Training progress</a>"
    return render(request, "search_form.html", {"views_str": views_str})
    
def plan(request):
    if request.POST:
        with open('./log/taskid.file', "r") as f:
            data = f.read()
            s1 = re.split('\n', data)
            print(s1[0])
        #print(os.system('ls'))         
        #call('bash nnunet_env.sh', shell=True)

#        cmd = """source activate seg;
#                  export nnUNet_raw_data_base="/home/img/data/nnUNet_raw_data_base/";
#                  export nnUNet_preprocessed="/home/img/data/nnUNet_preprocessed/";
#                  export RESULTS_FOLDER="/home/img/data/nnUNet_trained_models/";
#                  nnUNet_plan_and_preprocess -t {}
#              """.format(int(s1[0]))
#        call(cmd, shell=True)
        os.system('nohup nnUNet_plan_and_preprocess -t {} > ./log/plan.file 2>&1 &'.format(int(s1[0])))
    views_str = "<a href='http://10.8.77.18:8000/progress'>Training progress</a>"
    return render(request, "search_form.html", {"views_str": views_str})

def train(request):
    if request.POST:
        with open('./log/mode.file', "r") as f:
            data = f.read()
            s0 = re.split('/', data)
            print(s0[2])
        with open('./log/taskid.file', "r") as f:
            data = f.read()
            s1 = re.split('\n', data)
            print(s1[0])
            
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
            
        os.system('nohup nnUNet_train -c {} nnUNetTrainerV2_noDeepSupervision {} 0 --npz > ./log/train.file 2>&1 &'.format(s0[2],int(s1[0])))
    views_str = "<a href='http://10.8.77.18:8000/progress'>Training progress</a>"
    return render(request, "search_form.html", {"views_str": views_str})

def predict(request):
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
        os.system('nohup nnUNet_predict -i /home/zhuoli.zhuang/nnUNet_data/nnUNet_raw/nnUNet_raw_data/Task{}_{}/imagesTs/ -o /home/zhuoli.zhuang/nnUNet_data/result/Task{}_{} -t {} -m {} -f 0 -tr nnUNetTrainerV2_noDeepSupervision --disable_tta -chk model_best > ./log/predict.file 2>&1 &'.format(s1[0],s2[0],s1[0],s2[0],s1[0],s0[2]))

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
    
    
def check_result(request):
    if request.POST:
        os.system('python3 ./WebApp/generate_html.py')
    return render(request,"images.html")
    

num_progress = 0 
 
 

#def show_progress1(request):
#    # return JsonResponse(num_progress, safe=False)
#    return render(request, 'search_form.html')
 
 


def process_data(request):
    # ...
    global num_progress
 
    for i in range(12345666):
        num_progress = i * 100 / 12345666; 
        # print 'num_progress=' + str(num_progress)
        # time.sleep(1)
        res = num_progress
        # print 'i='+str(i)
        # print 'res----='+str(res)
    return JsonResponse(res, safe=False)
 

#def show_progress(request):
#    print('show_progress----------'+str(num_progress))
#    return JsonResponse(num_progress, safe=False)
