import os,csv
import numpy as np
from PIL import Image
#import nibabel as nib
import re
import shutil
from sklearn.metrics import *
import SimpleITK as sitk
import getopt
import sys
import cv2
from medpy import metric

'''
    1. using the result of nnunet, generate dice result and compared with the ground truth
    2. nii_path for images, nnunet_path for prediction and mask_path for ground truth are needed
'''
'''
def OverlayMaskOnCT(ct_array, mask_gt, mask_predict,save_path,slice_num):

    contour_gt, hierarchy = cv2.findContours(
                            mask_gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                                                                             
    contour_predict, hierarchy = cv2.findContours(
                            mask_predict, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)                              
    
    final_image = cv2.drawContours(ct_array, contour_gt, -1, (0, 0, 255), 1) # index=-1
                                                                             #(0,0,255)
    final_image = cv2.drawContours(final_image, contour_predict, -1, (255, 0, 0), 1) 
    cv2.imwrite(save_path+'/'+str(slice_num)+'.bmp', final_image.T) 
'''    

def printUsage():
	print ('''usage: eval_DL_result.py -i <input> -o <output>
       test.py --in=<input> --out=<output>''')
       
def CalculateDice(mask_gt, mask_pred):
 
    if (mask_gt.sum() == 0) and (mask_pred.sum() == 0):
        dice = 2.0 
    else:
        intersection = np.sum(mask_gt * mask_pred)
        union = np.sum(mask_gt) + np.sum(mask_pred)
        dice = (2. * intersection) / (union + 0.00001)   

    return dice
    
def mkdir(path):
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False
def MaskPost(mask_predict,mask_gt):
    size = mask_gt.shape
    for i in range(size[0]):
        if (np.sum(mask_gt[i,:,:])==0):
            mask_predict[i,:,:] = 0
        
    return mask_predict
def MaskClip(mask_predict,gtv_clip):
    '''
    find the range of mask due to the information of clip
    '''
    nonb = gtv_clip.nonzero()
    size = mask_predict.shape
    z_slice = nonb[0]
    x_slice = nonb[1]
    y_slice = nonb[2]
    z = int((z_slice[0]+z_slice[-1])/2)
    x = int((x_slice[0]+x_slice[-1])/2)
    y = int((y_slice[0]+y_slice[-1])/2)
    z_min = max(np.min(z_slice)-2,0)
    z_max = min(np.max(z_slice)+2,size[0])
    tmp = np.zeros(shape=(size[0],size[1],size[2]))
    #tmp[z_min:z_max,x-90:x+90,y-90:y+90] = 1
    tmp[z_min:z_max,:,:] = 1
    new_data = np.multiply(mask_predict,tmp)
    new_data = new_data.astype(np.uint8)
    return new_data   

def MaskTrue(mask_predict,gtv_clip):
    '''
    find the range of mask due to the information of clip
    '''
    nonb = gtv_clip.nonzero()
    size = mask_predict.shape
    z_slice = nonb[0]
    x_slice = nonb[1]
    y_slice = nonb[2]
    z = int((z_slice[0]+z_slice[-1])/2)
    x = int((x_slice[0]+x_slice[-1])/2)
    y = int((y_slice[0]+y_slice[-1])/2)
    z_min = max(np.min(z_slice),0)
    z_max = min(np.max(z_slice)+1,size[0])
    tmp = np.zeros(shape=(size[0],size[1],size[2]))
    #tmp[z_min:z_max,x-90:x+90,y-90:y+90] = 1
    tmp[z_min:z_max,:,:] = 1
    new_data = np.multiply(mask_predict,tmp)
    new_data = new_data.astype(np.uint8)
    return new_data  
    
def Normalization(image, window_level,window_width):
    """
    Normalize an image to 0 - 1, due to the window level and width
    """    
    image = image.astype(np.float32)
    image = (image-window_level)/window_width
    image[image>1] = 1
    image[image<0] = 0 
    return image
    
def Itk2array(nii_path):
    '''
    read nii.gz and convert to numpy array
    '''
    itk_data = sitk.ReadImage(nii_path)
    data = sitk.GetArrayFromImage(itk_data)
    #data_array = np.transpose(data, (2, 1, 0))
    return data
    
def dice_score(gt,pre):
    '''
    calculate dice, the input should be [0,1] mask
    '''
    intersection = np.sum(gt * pre)
    union = np.sum(gt) + np.sum(pre)
    dice = (2. * intersection) / (union + 0.00001) 
    return dice

def get_metrics(gt,pre):
    '''
    calculate metrics by sklearn, input should be [0,1] mask
    '''
    gt_s = gt.flatten()
    pre_s = pre.flatten()
    precision = precision_score(gt_s,pre_s)
    recall = recall_score(gt_s,pre_s)
    f1 = f1_score(gt_s,pre_s)
    dice = dice_score(gt,pre)
    return precision,recall,f1,dice    
def Mask2single(mask):
    mask1 = mask.copy()
    mask2 = mask.copy()
    mask1[mask1>0]=1
    mask2[mask2==1]=0
    mask2[mask2==2]=1
    return mask1,mask2
def generate_ctv(ctvb,gtv):
    itk_ctvb = sitk.ReadImage(ctvb)
    itk_gtv = sitk.ReadImage(gtv)
    gtv = sitk.BinaryDilate(itk_gtv==1, 8)
    d1 = sitk.GetArrayFromImage(itk_ctvb)
    d2 = sitk.GetArrayFromImage(gtv)
    data = np.multiply(d1,d2)
    #new_data = sitk.GetImageFromArray(data)
    return data
def OverlayMaskOnCT(ct_array, mask_gt, mask_predict,save_path):
    ct_array = (ct_array-np.min(ct_array))/(np.max(ct_array)-np.min(ct_array))
    # normalization to 0-1
    ct_array = ct_array*255
    ct_array = ct_array.astype(np.uint8)
    mask_gt = mask_gt*255
    mask_gt[mask_gt>125] = 255
    mask_gt[mask_gt<=125] = 0
    mask_gt = mask_gt.astype(np.uint8)
    mask_predict = mask_predict*255
    mask_predict[mask_predict>125] = 255
    mask_predict[mask_predict<=125] = 0
    mask_predict = mask_predict.astype(np.uint8)

    for i in range(ct_array.shape[0]):
        img1 = cv2.cvtColor(ct_array[i], cv2.COLOR_GRAY2RGB)
        img2 = mask_gt[i]

        img3 = mask_predict[i]

        contour_gt, hierarchy = cv2.findContours(
                          img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
                                                                             
        contour_predict, hierarchy = cv2.findContours(
                          img3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
    
        final_image = cv2.drawContours(img1, contour_gt, -1, (0, 0, 255), 1) 
                                                                             
        final_image = cv2.drawContours(final_image, contour_predict, -1, (0, 255, 0), 1) 
        cv2.imwrite(save_path+'/'+str(i)+'.bmp', final_image) 
        
def load_image(root, series, key):

    img_file = os.path.join(root, series)

    img = load_volfile_sitk(img_file, 0, key, True)
    z, y, x = np.shape(img)
    img = img.reshape((z, y, x))
    #image_dict[series] = utils.truncate(img, MIN_BOUND, MAX_BOUND)
    #stats_dict[series] = itk_img.GetOrigin(), itk_img.GetSpacing()

    return img

def load_volfile_sitk(path, depthdim, keywords,NORMALIZATION):

    sitkimage = sitk.ReadImage(path + keywords)
    image = sitk.GetArrayFromImage(sitkimage)
    if depthdim == 2:
        image = np.swapaxes(image,0,2)
    if NORMALIZATION:
        #image = normalization(image,[-150,400])
        image = normalization(image, [-250, 1350])
    else:
        image = image.astype(np.float32)
    return image


#window = (minimum,fullwidth)
def normalization(image,window):
    image = image.astype(np.float32)
    image = (image-window[0])/window[1]
    image[image>1] = 1
    image[image<0] = 0
    return image

def assert_shape(test, reference):

    assert test.shape == reference.shape, "Shape mismatch: {} and {}".format(
        test.shape, reference.shape)

class ConfusionMatrix:

    def __init__(self, test=None, reference=None):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.reference_empty = None
        self.reference_full = None
        self.test_empty = None
        self.test_full = None
        self.set_reference(reference)
        self.set_test(test)

    def set_test(self, test):

        self.test = test
        self.reset()

    def set_reference(self, reference):

        self.reference = reference
        self.reset()

    def reset(self):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.test_empty = None
        self.test_full = None
        self.reference_empty = None
        self.reference_full = None

    def compute(self):

        if self.test is None or self.reference is None:
            raise ValueError("'test' and 'reference' must both be set to compute confusion matrix.")

        assert_shape(self.test, self.reference)

        self.tp = int(((self.test != 0) * (self.reference != 0)).sum())
        self.fp = int(((self.test != 0) * (self.reference == 0)).sum())
        self.tn = int(((self.test == 0) * (self.reference == 0)).sum())
        self.fn = int(((self.test == 0) * (self.reference != 0)).sum())
        self.size = int(np.prod(self.reference.shape, dtype=np.int64))
        self.test_empty = not np.any(self.test)
        self.test_full = np.all(self.test)
        self.reference_empty = not np.any(self.reference)
        self.reference_full = np.all(self.reference)

    def get_matrix(self):

        for entry in (self.tp, self.fp, self.tn, self.fn):
            if entry is None:
                self.compute()
                break

        return self.tp, self.fp, self.tn, self.fn

    def get_size(self):

        if self.size is None:
            self.compute()
        return self.size

    def get_existence(self):

        for case in (self.test_empty, self.test_full, self.reference_empty, self.reference_full):
            if case is None:
                self.compute()
                break

        return self.test_empty, self.test_full, self.reference_empty, self.reference_full

def hausdorff_distance(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, voxel_spacing=None, connectivity=1, **kwargs):

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty or test_full or reference_empty or reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0

    test, reference = confusion_matrix.test, confusion_matrix.reference

    return metric.hd(test, reference, voxel_spacing, connectivity)

def Itk2array(nii_path):
    '''
    read nii.gz and convert to numpy array
    '''
    itk_data = sitk.ReadImage(nii_path)
    data = sitk.GetArrayFromImage(itk_data)
    #data_array = np.transpose(data, (2, 1, 0))
    return data
	

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
            
    nii_path = r"{}/nnUNet_raw_data/Task{}_{}/imagesTs/".format(os.getenv('nnUNet_raw_data_base'),s1[0],s2[0])
    mask_path = r"{}/nnUNet_raw_data/Task{}_{}/labelsTs/".format(os.getenv('nnUNet_raw_data_base'),s1[0],s2[0])
    nnunet_path = r"{}/result/Task{}_{}/".format(os.getenv('nnUNet_raw_data_base'),s1[0],s2[0])
    task_name = '{}'
    
    output_dir = r"{}/compare/Task{}_{}/".format(os.getenv('nnUNet_raw_data_base'),s1[0],s2[0])
    files = os.listdir(nii_path)
    #ctv_path = r"H:/breast_k/cropped_data/test/"
    mkdir(output_dir)
    
    v_name = output_dir+'/testdice_volume.xls'
    vfile = open(v_name,"w",newline='')
    vwriter = csv.writer(vfile)
    vwriter.writerow(['case','dice'])
    dice_ALN = []
    dice_sum = 0
    h_dist_sum = 0
    for file in files:
        index = file.rfind('_')
        name_p = file[:index]
        name = re.sub("\D", "", name_p)
        name = file[:-12]
        mask_gt_path = mask_path + name + '.nii.gz'
        ct_path = nii_path + file    
        predict_path = nnunet_path + name + '.nii.gz'
        #ctvb_path = ctv_path + name + '/CTV-B-Crop.nii.gz'
        #ctvtb_path = ctv_path + name + '/CTV-TB.nii.gz'
        
        
        #mask_gt = Itk2array(ctvtb_path)
        #predict = generate_ctv(ctvb_path,predict_path)
        predict = Itk2array(predict_path)
        predict_c = Itk2array(predict_path)
        mask_gt = Itk2array(mask_gt_path)
        ct_image = Itk2array(ct_path)
        
        compare_path = os.path.join(output_dir,name)
        #mkdir(compare_path)
        
        predict = predict.astype(np.uint8)
        mask_gt = mask_gt.astype(np.uint8)
        #predict[predict!=1] = 0
        #mask_gt[mask_gt!=1] = 0
        window_level = -150
        window_width = 400
        ct_array = Normalization(ct_image, window_level,window_width)
    
        ct_array = ct_array*255
        ct_array = ct_array.astype(np.uint8)
        
        ct_array1 = ct_array.copy() 
        #predict_mask,predict_clip = Mask2single(predict)
        #mask_gt,gt_clip = Mask2single(mask_gt)
        #if (np.sum(gt_clip)==0):
        #    print ("%s does not have clip" %name)
        #mask_gt[mask_gt>1] = 1    
        
        #if np.sum(predict_clip)>0:
            #predict_mask = MaskClip(predict_mask,predict_clip)
        #predict_tmp,clip = Mask2single(predict_c)
        #predict_mask = MaskClip(predict_mask,clip)
        #predict_mask = MaskTrue(predict_mask,mask_gt)
        size = ct_array.shape
        '''
        for index in range(size[0]):
    
            pre = predict_mask[index,:,:]
            gt = mask_gt[index,:,:]
            if np.sum(pre+gt)>0:
                dice = dice_score(gt,pre)
                #if np.sum(precision+recall+f1+dice)>0:
                csvwriter.writerow([str(name),str(index),str(dice)])
        '''    
        
            
        dice = dice_score(mask_gt,predict)
        #h_dist = hausdorff_distance(test=mask_gt,reference=predict)
        dice_sum += dice
        #h_dist_sum += h_dist
        vwriter.writerow([str(name), str(dice)])
        result = predict*2+mask_gt
        new_img = sitk.GetImageFromArray(result)
        sitk.WriteImage(new_img, output_dir+str(name)+'_result.nii.gz')
        shutil.copy(ct_path, output_dir+str(name)+'_image.nii.gz')
        ct_array = load_image(ct_path, output_dir, '/'+str(name)+'_image.nii.gz')
        mkdir(output_dir+'/'+str(name))
        OverlayMaskOnCT(ct_array,mask_gt,predict,output_dir+'/'+str(name))
        print ("%s Dice is %s " %(name,dice))
    print("Average dice is %s" % (dice_sum/len(files)))
    vwriter.writerow(['Average Dice',str(dice_sum/len(files))])

    with open('./log/dice.file','w',encoding='utf-8') as f:
        text = str(dice_sum/len(files))
        f.write(text)
    #f.close()
    vfile.close()
    
	
