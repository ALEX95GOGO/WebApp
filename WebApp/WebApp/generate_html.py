#coding:utf-8
import os,sys

def showImageInHTML(imageTypes,imagedir,savedir):
    folder_list = os.listdir(imagedir)
    #print(folder_list)
    files=getAllFiles(imagedir + '/' + folder_list[0])
    images=[f for f in files if f[f.rfind('.')+1:] in imageTypes]
    #images.sort(key=lambda x:int(x[:-4]))
    #print(files)
    images=[item for item in images if os.path.getsize(item)>5*1024]
    images=[item[item.rfind('/'):] for item in images]
    images.sort(key=lambda x:int(x[1:-4]))
    images=['/prediction/' + folder_list[0] + item for item in images]
    newfile='%s/%s'%(savedir,'images.html')
    with open(newfile,'w') as f:
        f.write('<div>')
        f.write('<h1>Image ID is :'+folder_list[0]+'</h1>')
        for image in images:
            f.write("<img src='%s'>\n"%image)
        f.write('</div>')
    print ('success,images are wrapped up in %s'%newfile)

def getAllFiles(directory):
    files=[]
    for dirpath, dirnames,filenames in os.walk(directory):
        if filenames!=[]:
            for file in filenames:
                files.append(dirpath+'/'+file)
    files.sort(key=len)
    return files


def cur_file_dir():
    path = sys.path[0]
    if os.path.isdir(path):
        return path
    elif os.path.isfile(path):
        return os.path.dirname(path)
     
if __name__ == '__main__':
    #savedir=cur_file_dir()
    imagedir=r'/home/img/data/compare/Task121_optic_r/'
    savedir=r'./templates'
    showImageInHTML(('bmp','png','gif'),imagedir, savedir)