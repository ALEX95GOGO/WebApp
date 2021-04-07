import os
  

print("**********************************************")    
path=r"/home/zhuoli.zhuang/web/WebApp/data/nii_base"

for root,dirs,files in os.walk(path):
    for name in files:
        NewFileName=name.replace(" ",'');
        NewFileName=os.path.join(root,NewFileName);
        print(NewFileName);
        os.rename(os.path.join(root,name),os.path.join(root,NewFileName))


