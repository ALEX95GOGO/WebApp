#coding=utf-8
import cv2
import numpy as np
import pydicom
import SimpleITK as sitk
import glob
import copy
import os
import SimpleITK as sitk
from multiprocessing import Pool,Manager


def dcm2nii(dcms_path, nii_path):
	# 1.构建dicom序列文件阅读器，并执行（即将dicom序列文件“打包整合”）
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dcms_path)
    reader.SetFileNames(dicom_names)
    image2 = reader.Execute()
	# 2.将整合后的数据转为array，并获取dicom文件基本信息
    image_array = sitk.GetArrayFromImage(image2)  # z, y, x
    origin = image2.GetOrigin()  # x, y, z
    spacing = image2.GetSpacing()  # x, y, z
    direction = image2.GetDirection()  # x, y, z
	# 3.将array转为img，并保存为.nii.gz
    image3 = sitk.GetImageFromArray(image_array)
    image3.SetSpacing(spacing)
    image3.SetDirection(direction)
    image3.SetOrigin(origin)
    sitk.WriteImage(image3, nii_path)


## SOP Class UID
CTSOPClassUID =         '1.2.840.10008.5.1.4.1.1.2'
EnhancedCTSOPClassUID = '1.2.840.10008.5.1.4.1.1.2.1'
MRSOPClassUID =         '1.2.840.10008.5.1.4.1.1.4'
EnhancedMRSOPClassUID = '1.2.840.10008.5.1.4.1.1.4.1'
RTSSSOPClassUID =       '1.2.840.10008.5.1.4.1.1.481.3'


def slice_contour_to_mask(shape_x, shape_y, contours: list):
    """
    args:
        contours: [contour_array1, contour_array2, ...]
    return:
        mask response to contours
    """
    mask = np.zeros((shape_y, shape_x), np.uint8)
    cv2.fillPoly(mask, contours, (1))
    return mask

def get_ct_names_property(dcm_path, SOPClassUID):
    """
    get the info in dcm file
    """
    dcm_fns = glob.glob(dcm_path+'/*.dcm')
    
    fn_SOPInstanceUID_Position = []
    for dcm_fn in dcm_fns:
        ds_CT = pydicom.dcmread(dcm_fn, force = True)
        if ds_CT.SOPClassUID == SOPClassUID:  # for image SOPClassUID
            fn_SOPInstanceUID_Position.append([dcm_fn, ds_CT.SOPInstanceUID, ds_CT.ImagePositionPatient])

    fn_SOPInstanceUID_Position.sort(key=lambda x: float(x[-1][-1]))

    image_fns = [x[0] for x in fn_SOPInstanceUID_Position]
    SOPInstanceUID = [x[1] for x in fn_SOPInstanceUID_Position]
    Position = [x[2] for x in fn_SOPInstanceUID_Position]

    image_one = pydicom.dcmread(image_fns[0], force = True)
    SOPClassUID = image_one.SOPClassUID
    Spacing = image_one.PixelSpacing
    spacing_z = image_one.SpacingBetweenSlices if getattr(image_one, 'SpacingBetweenSlices', None) else image_one.SliceThickness
    Spacing = [Spacing[0], Spacing[1], spacing_z]
    Shape = (int(image_one.Rows), int(image_one.Columns), len(image_fns))
    Origin = Position[0]
    return image_fns, SOPClassUID, SOPInstanceUID, Position, Spacing, Shape, Origin

def get_contours_point_from_rtss(contour_sequence):
    """
    get the contour array in rtss contour sequence.
    return:
        contours: [[[x11, y11, z11], [x12, y12, z12], ...], [[x21, y21, z21], [x22, y22, z22], ...], ...]
                  the [x11, y11, z11] means point place in real world.
    """
    contours = []
    for contour in contour_sequence:
        contour_data = copy.deepcopy(contour.ContourData)
        contour_data = np.array(contour_data)
        contour_data = contour_data.reshape(-1, 3)
        contours.append(contour_data)
    # contours.sort(key=lambda x: x[-1, -1])
    return contours

def get_slice_contours(contour_sequence, Position, Spacing, shape_x, shape_y):
    """
    get contour in every z slice
    return:
        slice_contours: [slice1_contours, slice2_contours, ...]
                         slice1_contours: [[[x11, y11], [x12, y12], ...], [[x11, y11], [x12, y12], ...], ...]
                                          the order of points in the same z slice.        
    """
    contours = get_contours_point_from_rtss(contour_sequence)
    slice_contours = []
    for p in Position:
        slice_contour = []
        for contour in contours:
            if abs(contour[-1][-1] - p[-1]) < 1e-6:
                contour[:, 0] = (contour[:, 0] - p[0]) / Spacing[0]
                contour[:, 1] = (contour[:, 1] - p[1]) / Spacing[1]
                contour = contour[:, :2]
                contour = np.around(contour, decimals=0)
                contour[:, 0] = np.clip(contour[:, 0], 0, shape_x-1)
                contour[:, 1] = np.clip(contour[:, 1], 0, shape_y-1)
                contour = contour.astype(np.int32) 
                # print(contour.shape)
                slice_contour.append(contour)
        slice_contours.append(slice_contour)
    return slice_contours

def rtss_single_roi_2_nii(current_contour_sequence, Position, Spacing, Origin, Shape):
    """
    roi contours convert to mask
    """
    shape_x, shape_y, shape_z = Shape
    contours = get_slice_contours(current_contour_sequence, Position, Spacing, shape_x, shape_y)
    mask = np.zeros((shape_z, shape_y, shape_x), np.uint8)
    for i in range(shape_z):
        if len(contours[i])==0: continue
        mask[i] = slice_contour_to_mask(shape_x, shape_y, contours[i])

    mask = sitk.GetImageFromArray(mask)
    mask.SetOrigin(Origin)
    mask.SetSpacing(Spacing)
    return mask

def rtss2nii(rtss_path, dcm_path, nii_folder):
    _, _, _, Position, Spacing, Shape, Origin = get_ct_names_property(dcm_path, CTSOPClassUID)
    
    rtss = pydicom.dcmread(rtss_path, force=True)
    
    for current_roi_sequence in rtss.StructureSetROISequence:
        try:
            current_roi_number = current_roi_sequence.ROINumber
            current_roi_name = current_roi_sequence.ROIName
            current_roi_contour_sequence = None
            for roi_contour_sequence in rtss.ROIContourSequence:
                if roi_contour_sequence.ReferencedROINumber == current_roi_number:
                    current_roi_contour_sequence = roi_contour_sequence
                    break
            assert current_roi_contour_sequence is not None
            current_contour_sequence = current_roi_contour_sequence.ContourSequence
    
            mask = rtss_single_roi_2_nii(current_contour_sequence, Position, Spacing, Origin, Shape)
    
            sitk.WriteImage(mask, os.path.join(nii_folder, current_roi_name+'.nii.gz'))
        except Exception as e:
            err = str(e)
            print(err)
            pass
        
def findfiles(path):
    result = []
    count = 1
    for root, dirs, files in os.walk(path):
        for filename in files:
            if "RS" in filename:
                result.append(root + "/" + filename)
                count += 1
    return result[0]
    
def mkdir(path):
    # 判断路径是否存在
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False
        
if __name__ == '__main__':
    # rtss_path: the file name of patient's rtss
    # dcm_path: the dicom folder of patient's dicom
    # nii_folder: the nii folder to save mask
    
    rtss_path = r'./WebApp/upload/Chest_dcm/'
    dcm_path = r'./WebApp/upload/Chest_dcm/'
    nii_folder = r'./WebApp/upload/Chest_nii/'
    folder_list = os.listdir(dcm_path)
    mkdir(nii_folder)
    process_pool = Pool(4)  # multiprocess = 4
    que = Manager().Queue()
    for i in range(len(folder_list)):
        print(folder_list[i])
        rtss_name = findfiles(os.path.join(rtss_path,folder_list[i]))
        mkdir(os.path.join(nii_folder,folder_list[i]))
        process_pool.apply_async(rtss2nii, args=(rtss_name, os.path.join(dcm_path,folder_list[i]), os.path.join(nii_folder,folder_list[i])))
        dcm2nii(os.path.join(dcm_path,folder_list[i]), os.path.join(nii_folder,folder_list[i],'./image.nii.gz'))
        
    process_pool.close()
    process_pool.join()