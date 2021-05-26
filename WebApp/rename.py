import os
  

print("**********************************************")    
path=r"/home/file/web/data/nii_base/Task333/"

oldname = {}
newname = ''
for root,dirs,files in os.walk(path):
    for name in files:
        NewFileName=name.replace("mask","skin");
        NewFileName=os.path.join(root,NewFileName);
        print(NewFileName);
        os.rename(os.path.join(root,name),os.path.join(root,NewFileName))


