import torch
import nnunet
import pickle
import os,csv
import sys
import pkgutil
import importlib
from multiprocessing import Process, Queue
import numpy as np
import SimpleITK as sitk
from typing import Union, Tuple
from copy import deepcopy
from skimage.transform import resize
import getopt
import re
import shutil
import cv2

def load_pickle(file, mode='rb'):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a

def subfiles(folder, join=True, prefix=None, suffix=None, sort=True):
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))
            and (prefix is None or i.startswith(prefix))
            and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res	

def recursive_find_python_class(folder, trainer_name, current_module):
    tr = None
    for importer, modname, ispkg in pkgutil.iter_modules(folder):
        # print(modname, ispkg)
        if not ispkg:
            m = importlib.import_module(current_module + "." + modname)
            if hasattr(m, trainer_name):
                tr = getattr(m, trainer_name)
                break

    if tr is None:
        for importer, modname, ispkg in pkgutil.iter_modules(folder):
            if ispkg:
                next_current_module = current_module + "." + modname
                tr = recursive_find_python_class([os.path.join(folder[0], modname)], trainer_name, current_module=next_current_module)
            if tr is not None:
                break

    return tr
def preprocess_multithreaded(trainer, list_of_lists, output_files, num_processes=2, segs_from_prev_stage=None):
    if segs_from_prev_stage is None:
        segs_from_prev_stage = [None] * len(list_of_lists)

    num_processes = min(len(list_of_lists), num_processes)

    classes = list(range(1, trainer.num_classes))
    #assert isinstance(trainer, nnUNetTrainer)
    q = Queue(1)
    processes = []
    for i in range(num_processes):
        pr = Process(target=preprocess_save_to_queue, args=(trainer.preprocess_patient, q,
                                                            list_of_lists[i::num_processes],
                                                            output_files[i::num_processes],
                                                            segs_from_prev_stage[i::num_processes],
                                                            classes, trainer.plans['transpose_forward']))
        pr.start()
        processes.append(pr)

    try:
        end_ctr = 0
        while end_ctr != num_processes:
            item = q.get()
            if item == "end":
                end_ctr += 1
                continue
            else:
                yield item

    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()  # this should not happen but better safe than sorry right
            p.join()

        q.close()

def preprocess_save_to_queue(preprocess_fn, q, list_of_lists, output_files, segs_from_prev_stage, classes,
                             transpose_forward):
    # suppress output
    # sys.stdout = open(os.devnull, 'w')

    errors_in = []
    for i, l in enumerate(list_of_lists):
        try:
            output_file = output_files[i]
            print("preprocessing", output_file)
            d, _, dct = preprocess_fn(l)
            # print(output_file, dct)
            if segs_from_prev_stage[i] is not None:
                assert isfile(segs_from_prev_stage[i]) and segs_from_prev_stage[i].endswith(
                    ".nii.gz"), "segs_from_prev_stage" \
                                " must point to a " \
                                "segmentation file"
                seg_prev = sitk.GetArrayFromImage(sitk.ReadImage(segs_from_prev_stage[i]))
                # check to see if shapes match
                img = sitk.GetArrayFromImage(sitk.ReadImage(l[0]))
                assert all([i == j for i, j in zip(seg_prev.shape, img.shape)]), "image and segmentation from previous " \
                                                                                 "stage don't have the same pixel array " \
                                                                                 "shape! image: %s, seg_prev: %s" % \
                                                                                 (l[0], segs_from_prev_stage[i])
                seg_prev = seg_prev.transpose(transpose_forward)
                seg_reshaped = resize_segmentation(seg_prev, d.shape[1:], order=1, cval=0)
                seg_reshaped = to_one_hot(seg_reshaped, classes)
                d = np.vstack((d, seg_reshaped)).astype(np.float32)
            """There is a problem with python process communication that prevents us from communicating obejcts 
            larger than 2 GB between processes (basically when the length of the pickle string that will be sent is 
            communicated by the multiprocessing.Pipe object then the placeholder (\%i I think) does not allow for long 
            enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually 
            patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will 
            then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either 
            filename or np.ndarray and will handle this automatically"""
            print(d.shape)
            if np.prod(d.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save, 4 because float32 is 4 bytes
                print(
                    "This output is too large for python process-process communication. "
                    "Saving output temporarily to disk")
                np.save(output_file[:-7] + ".npy", d)
                d = output_file[:-7] + ".npy"
            q.put((output_file, (d, dct)))
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            print("error in", l)
            print(e)
    q.put("end")
    if len(errors_in) > 0:
        print("There were some errors in the following cases:", errors_in)
        print("These cases were ignored.")
    else:
        print("This worker has ended successfully, no errors to report")
    # restore output
    # sys.stdout = sys.__stdout__

def save_segmentation_nifti_from_softmax(segmentation_softmax: np.ndarray, out_fname: str,
                                         properties_dict: dict, order: int = 1,
                                         region_class_order: Tuple[Tuple[int]] = None,
                                         seg_postprogess_fn: callable = None, seg_postprocess_args: tuple = None,
                                         resampled_npz_fname: str = None,
                                         non_postprocessed_fname: str = None, force_separate_z: bool = None,
                                         interpolation_order_z: int = 0, verbose: bool = True):
    """
    This is a utility for writing segmentations to nifto and npz. It requires the data to have been preprocessed by
    GenericPreprocessor because it depends on the property dictionary output (dct) to know the geometry of the original
    data. segmentation_softmax does not have to have the same size in pixels as the original data, it will be
    resampled to match that. This is generally useful because the spacings our networks operate on are most of the time
    not the native spacings of the image data.
    If seg_postprogess_fn is not None then seg_postprogess_fnseg_postprogess_fn(segmentation, *seg_postprocess_args)
    will be called before nifto export
    There is a problem with python process communication that prevents us from communicating obejcts
    larger than 2 GB between processes (basically when the length of the pickle string that will be sent is
    communicated by the multiprocessing.Pipe object then the placeholder (\%i I think) does not allow for long
    enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually
    patching system python code.) We circumvent that problem here by saving softmax_pred to a npy file that will
    then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either
    filename or np.ndarray for segmentation_softmax and will handle this automatically
    :param segmentation_softmax:
    :param out_fname:
    :param properties_dict:
    :param order:
    :param region_class_order:
    :param seg_postprogess_fn:
    :param seg_postprocess_args:
    :param resampled_npz_fname:
    :param non_postprocessed_fname:
    :param force_separate_z: if None then we dynamically decide how to resample along z, if True/False then always
    /never resample along z separately. Do not touch unless you know what you are doing
    :param interpolation_order_z: if separate z resampling is done then this is the order for resampling in z
    :param verbose:
    :return:
    """
    if verbose: print("force_separate_z:", force_separate_z, "interpolation order:", order)


    # first resample, then put result into bbox of cropping, then save
    current_shape = segmentation_softmax.shape
    shape_original_after_cropping = properties_dict.get('size_after_cropping')
    shape_original_before_cropping = properties_dict.get('original_size_of_raw_data')
    # current_spacing = dct.get('spacing_after_resampling')
    # original_spacing = dct.get('original_spacing')

    if np.any([i != j for i, j in zip(np.array(current_shape[1:]), np.array(shape_original_after_cropping))]):
        if force_separate_z is None:
            if get_do_separate_z(properties_dict.get('original_spacing')):
                do_separate_z = True
                lowres_axis = get_lowres_axis(properties_dict.get('original_spacing'))
            elif get_do_separate_z(properties_dict.get('spacing_after_resampling')):
                do_separate_z = True
                lowres_axis = get_lowres_axis(properties_dict.get('spacing_after_resampling'))
            else:
                do_separate_z = False
                lowres_axis = None
        else:
            do_separate_z = force_separate_z
            if do_separate_z:
                lowres_axis = get_lowres_axis(properties_dict.get('original_spacing'))
            else:
                lowres_axis = None

        if verbose: print("separate z:", do_separate_z, "lowres axis", lowres_axis)
        seg_old_spacing = resample_data_or_seg(segmentation_softmax, shape_original_after_cropping, is_seg=False,
                                               axis=lowres_axis, order=order, do_separate_z=do_separate_z, cval=0,
                                               order_z=interpolation_order_z)
        # seg_old_spacing = resize_softmax_output(segmentation_softmax, shape_original_after_cropping, order=order)
    else:
        if verbose: print("no resampling necessary")
        seg_old_spacing = segmentation_softmax

    if resampled_npz_fname is not None:
        np.savez_compressed(resampled_npz_fname, softmax=seg_old_spacing.astype(np.float16))
        # this is needed for ensembling if the nonlinearity is sigmoid
        if region_class_order is not None:
            properties_dict['regions_class_order'] = region_class_order
        save_pickle(properties_dict, resampled_npz_fname[:-4] + ".pkl")

    if region_class_order is None:
        seg_old_spacing = seg_old_spacing.argmax(0)
    else:
        seg_old_spacing_final = np.zeros(seg_old_spacing.shape[1:])
        for i, c in enumerate(region_class_order):
            seg_old_spacing_final[seg_old_spacing[i] > 0.5] = c
        seg_old_spacing = seg_old_spacing_final

    bbox = properties_dict.get('crop_bbox')

    if bbox is not None:
        seg_old_size = np.zeros(shape_original_before_cropping)
        for c in range(3):
            bbox[c][1] = np.min((bbox[c][0] + seg_old_spacing.shape[c], shape_original_before_cropping[c]))
        seg_old_size[bbox[0][0]:bbox[0][1],
        bbox[1][0]:bbox[1][1],
        bbox[2][0]:bbox[2][1]] = seg_old_spacing
    else:
        seg_old_size = seg_old_spacing

    if seg_postprogess_fn is not None:
        seg_old_size_postprocessed = seg_postprogess_fn(np.copy(seg_old_size), *seg_postprocess_args)
    else:
        seg_old_size_postprocessed = seg_old_size

    seg_resized_itk = sitk.GetImageFromArray(seg_old_size_postprocessed.astype(np.uint8))
    seg_resized_itk.SetSpacing(properties_dict['itk_spacing'])
    seg_resized_itk.SetOrigin(properties_dict['itk_origin'])
    seg_resized_itk.SetDirection(properties_dict['itk_direction'])
    sitk.WriteImage(seg_resized_itk, out_fname)

    if (non_postprocessed_fname is not None) and (seg_postprogess_fn is not None):
        seg_resized_itk = sitk.GetImageFromArray(seg_old_size.astype(np.uint8))
        seg_resized_itk.SetSpacing(properties_dict['itk_spacing'])
        seg_resized_itk.SetOrigin(properties_dict['itk_origin'])
        seg_resized_itk.SetDirection(properties_dict['itk_direction'])
        sitk.WriteImage(seg_resized_itk, non_postprocessed_fname)

def get_do_separate_z(spacing, anisotropy_threshold=3):
    do_separate_z = (np.max(spacing) / np.min(spacing)) > anisotropy_threshold
    return do_separate_z

def get_lowres_axis(new_spacing):
    axis = np.where(max(new_spacing) / np.array(new_spacing) == 1)[0]  # find which axis is anisotropic
    return axis	

def resample_data_or_seg(data, new_shape, is_seg, axis=None, order=3, do_separate_z=False, cval=0, order_z=0):
    """
    separate_z=True will resample with order 0 along z
    :param data:
    :param new_shape:
    :param is_seg:
    :param axis:
    :param order:
    :param do_separate_z:
    :param cval:
    :param order_z: only applies if do_separate_z is True
    :return:
    """
    assert len(data.shape) == 4, "data must be (c, x, y, z)"
    if is_seg:
        resize_fn = resize_segmentation
        kwargs = OrderedDict()
    else:
        resize_fn = resize
        kwargs = {'mode': 'edge', 'anti_aliasing': False}
    dtype_data = data.dtype
    data = data.astype(float)
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)
    if np.any(shape != new_shape):
        if do_separate_z:
            print("separate z, order in z is", order_z, "order inplane is", order)
            assert len(axis) == 1, "only one anisotropic axis supported"
            axis = axis[0]
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]

            reshaped_final_data = []
            for c in range(data.shape[0]):
                reshaped_data = []
                for slice_id in range(shape[axis]):
                    if axis == 0:
                        reshaped_data.append(resize_fn(data[c, slice_id], new_shape_2d, order, cval=cval, **kwargs))
                    elif axis == 1:
                        reshaped_data.append(resize_fn(data[c, :, slice_id], new_shape_2d, order, cval=cval, **kwargs))
                    else:
                        reshaped_data.append(resize_fn(data[c, :, :, slice_id], new_shape_2d, order, cval=cval,
                                                       **kwargs))
                reshaped_data = np.stack(reshaped_data, axis)
                if shape[axis] != new_shape[axis]:

                    # The following few lines are blatantly copied and modified from sklearn's resize()
                    rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                    orig_rows, orig_cols, orig_dim = reshaped_data.shape

                    row_scale = float(orig_rows) / rows
                    col_scale = float(orig_cols) / cols
                    dim_scale = float(orig_dim) / dim

                    map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                    map_rows = row_scale * (map_rows + 0.5) - 0.5
                    map_cols = col_scale * (map_cols + 0.5) - 0.5
                    map_dims = dim_scale * (map_dims + 0.5) - 0.5

                    coord_map = np.array([map_rows, map_cols, map_dims])
                    if not is_seg or order_z == 0:
                        reshaped_final_data.append(map_coordinates(reshaped_data, coord_map, order=order_z, cval=cval,
                                                                   mode='nearest')[None])
                    else:
                        unique_labels = np.unique(reshaped_data)
                        reshaped = np.zeros(new_shape, dtype=dtype_data)

                        for i, cl in enumerate(unique_labels):
                            reshaped_multihot = np.round(
                                map_coordinates((reshaped_data == cl).astype(float), coord_map, order=order_z,
                                                cval=cval, mode='nearest'))
                            reshaped[reshaped_multihot > 0.5] = cl
                        reshaped_final_data.append(reshaped[None])
                else:
                    reshaped_final_data.append(reshaped_data[None])
            reshaped_final_data = np.vstack(reshaped_final_data)
        else:
            print("no separate z, order", order)
            reshaped = []
            for c in range(data.shape[0]):
                reshaped.append(resize_fn(data[c], new_shape, order, cval=cval, **kwargs)[None])
            reshaped_final_data = np.vstack(reshaped)
        return reshaped_final_data.astype(dtype_data)
    else:
        print("no resampling necessary")
        return data

def mkdir(path):
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False
        
def CalculateDice(mask_gt, mask_pred):
 
    if (mask_gt.sum() == 0) and (mask_pred.sum() == 0):
        dice = 2.0 
    else:
        intersection = np.sum(mask_gt * mask_pred)
        union = np.sum(mask_gt) + np.sum(mask_pred)
        dice = (2. * intersection) / (union + 0.00001)   

    return dice

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

def normalization(image,window):
    image = image.astype(np.float32)
    image = (image-window[0])/window[1]
    image[image>1] = 1
    image[image<0] = 0
    return image

if __name__ == '__main__':
    inputarg=""
    outputarg=""

    opts, args = getopt.getopt(sys.argv[1:],"hi:o:",["in=","out="])

    for opt,arg in opts:
        if opt in ("-i", "--id"):
            idarg=arg
        elif opt in ("-n","--name"):
            namearg=arg
            
    print ('id:'+inputarg)
    print ('other param:'+",".join(args))
   
    with open('./log/' + args[0], "r") as f:
        data1 = f.read()
        s1 = re.split('\n', data1)
        print(s1[0])
    with open('./log/' + args[1], "r") as f:
        data2 = f.read()
        s2 = re.split('\n', data2)
        print(s2[0])
    with open('./log/' + args[2], "r") as f:
        data3 = f.read()
        s3 = re.split('/', data3)
        print(s3[2])
    with open('./log/' + args[3], "r") as f:
        data4 = f.read()
        s4 = re.split('\n', data4)
        print(s4[0])

    expected_num_modalities = load_pickle("{}/nnUNet/{}/Task{}_{}/nnUNetTrainerV2_noDeepSupervision__nnUNetPlansv2.1/plans.pkl".format(os.getenv('RESULTS_FOLDER'),s3[2],s4[0],s2[0]))['num_modalities']
    input_path = r"{}/nnUNet_raw_data/Eval{}_{}/imagesTs/".format(os.getenv('nnUNet_raw_data_base'),s1[0],s2[0])
    output_path = r"{}/result/Eval{}_{}/".format(os.getenv('nnUNet_raw_data_base'),s1[0],s2[0])
    mkdir(output_path)
    all_files = subfiles(input_path, suffix=".nii.gz", join=False, sort=True)
    count = len(all_files)
    
    for i in all_files:
        output_files = [os.path.join(output_path, i)]
        list_of_lists = [os.path.join(input_path, i)]
        print("number of cases that still need to be predicted:", count)
        torch.cuda.empty_cache()
        info = load_pickle("{}/nnUNet/{}/Task{}_{}/nnUNetTrainerV2_noDeepSupervision__nnUNetPlansv2.1/fold_0/model_best.model.pkl".format(os.getenv('RESULTS_FOLDER'),s3[2],s4[0],s2[0]))
        init = info['init']
        name = info['name']
        search_in = os.path.join(nnunet.__path__[0], "training", "network_training")
        tr = recursive_find_python_class([search_in], name, current_module="nnunet.training.network_training")
        
        trainer = tr(*init)
        trainer.process_plans(info['plans'])
        trainer.load_checkpoint("{}/nnUNet/{}/Task{}_{}/nnUNetTrainerV2_noDeepSupervision__nnUNetPlansv2.1/fold_0/model_best.model".format(os.getenv('RESULTS_FOLDER'),s3[2],s4[0],s2[0]), train=False)
        all_params = [torch.load("{}/nnUNet/{}/Task{}_{}/nnUNetTrainerV2_noDeepSupervision__nnUNetPlansv2.1/fold_0/model_best.model".format(os.getenv('RESULTS_FOLDER'),s3[2],s4[0],s2[0]), map_location=torch.device('cpu'))]
        force_separate_z = None
        interpolation_order = 1
        interpolation_order_z = 0
        
        preprocessing = preprocess_multithreaded(trainer, list_of_lists, output_files, 6,
                                                     None)
        for i,l in enumerate(list_of_lists):
            d1,d2,d3 = trainer.preprocess_patient([l])
        softmax = []
        for p in all_params:
                    trainer.load_checkpoint_ram(p, False)
                    softmax.append(trainer.predict_preprocessed_data_return_seg_and_softmax(d1, False, trainer.data_aug_params[
                        'mirror_axes'], True, step_size=0.5, use_gaussian=True, all_in_gpu=None,
                                                                                            mixed_precision=True)[1][None])
        softmax = np.vstack(softmax)
        softmax_mean = np.mean(softmax, 0)
        transpose_forward = trainer.plans.get('transpose_forward')
        if transpose_forward is not None:
            transpose_backward = trainer.plans.get('transpose_backward')
            softmax_mean = softmax_mean.transpose([0] + [i + 1 for i in transpose_backward])
        result = save_segmentation_nifti_from_softmax(softmax_mean, output_files[-1], d3, interpolation_order, None,
                                                    None, None,None, None, force_separate_z, interpolation_order_z)	
    
    nii_path = r"{}/nnUNet_raw_data/Eval{}_{}/imagesTs/".format(os.getenv('nnUNet_raw_data_base'),s1[0],s2[0])
    mask_path = r"{}/nnUNet_raw_data/Eval{}_{}/labelsTs/".format(os.getenv('nnUNet_raw_data_base'),s1[0],s2[0])
    nnunet_path = output_path
    output_dir = r"{}/compare/Eval{}_{}/".format(os.getenv('nnUNet_raw_data_base'),s1[0],s2[0])
    
    files = os.listdir(nii_path)
    
    mkdir(output_dir)

    v_name = output_dir+'/testdice_volume.csv'
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
        predict_path = nnunet_path + name + '_0000.nii.gz'
    
        predict = Itk2array(predict_path)
        predict_c = Itk2array(predict_path)
        mask_gt = Itk2array(mask_gt_path)
        ct_image = Itk2array(ct_path)
        
        predict = predict.astype(np.uint8)
        mask_gt = mask_gt.astype(np.uint8)        
            
        dice = dice_score(mask_gt,predict)
        dice_sum += dice
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
    vfile.close()