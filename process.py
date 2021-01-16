
import glob
import os

import cv2
import numpy as np
import scipy.io as sio
from scipy.ndimage import filters, measurements
from scipy.ndimage.morphology import (binary_dilation, binary_fill_holes,
                                      distance_transform_cdt,
                                      distance_transform_edt)
from skimage.morphology import remove_small_objects, watershed

import postproc.hover
import postproc.dist
import postproc.other

from config import Config

from misc.viz_utils import visualize_instances
from misc.utils import get_inst_centroid
from metrics.stats_utils import remap_label
import matplotlib.pyplot as plt

###################

# TODO: 
# * due to the need of running this multiple times, should make 
# * it less reliant on the training config file

## ! WARNING: 
## check the prediction channels, wrong ordering will break the code !
## the prediction channels ordering should match the ones produced in augs.py

cfg = Config()

# * flag for HoVer-Net only
# 1 - threshold, 2 - sobel based
energy_mode = 2 
marker_mode = 2 

pred_dir = cfg.inf_output_dir
proc_dir = pred_dir + '_proc2'
true_mask_dir = '/home/test/GhulamMurtaza/panNuke/Test/Labels/'

file_list = glob.glob('%s/*.mat' % (pred_dir))
file_list.sort() # ensure same order

if not os.path.isdir(proc_dir):
    os.makedirs(proc_dir)

for filename in file_list:
    filename = os.path.basename(filename)
    basename = filename.split('.')[0]
    print(pred_dir, basename, end=' ', flush=True)

    ##
    img = cv2.imread(cfg.inf_data_dir + basename + cfg.inf_imgs_ext)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    true_mask = np.load('%s/%s.npy'%(true_mask_dir,basename))
    true_mask_type = true_mask[:,:,1]

    pred = sio.loadmat('%s/%s.mat' % (pred_dir, basename))
    pred = np.squeeze(pred['result'])

    if hasattr(cfg, 'type_classification') and cfg.type_classification:
        pred_inst = pred[...,cfg.nr_types:]
        pred_type = pred[...,:cfg.nr_types]

        pred_inst = np.squeeze(pred_inst)
        pred_type = np.argmax(pred_type, axis=-1)

        if cfg.model_type == 'micronet':
            # dilate prediction of all type to match it with
            # the instance segmentation post-proc code
            kernel = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]], np.uint8)
            canvas = np.zeros_like(pred_type, dtype=np.int32)
            for type_id in range(1, cfg.nr_classes):
                type_map = (pred_type == type_id).astype('uint8')
                type_map = cv2.dilate(type_map, kernel, iterations=1)
                canvas[type_map > 0] = type_id
    else:
        pred_inst = pred

    # h_dis_raw = pred_inst[...,1]
    # v_dis_raw = pred_inst[...,2]
    # np.save('/home/test/GhulamMurtaza/Distance Maps/%s_horizontal.npy' % (basename),h_dis_raw)
    # np.save('/home/test/GhulamMurtaza/Distance Maps/%s_vertical.npy' % (basename),v_dis_raw)
    probab_maps = pred_inst[...,0]
    np.save('/home/test/GhulamMurtaza/Nuclei Probabiilty Maps/%s_porbbiltyMaps' % (basename),probab_maps)

    if cfg.model_type == 'np_hv':
        pred_inst = postproc.hover.proc_np_hv(pred_inst, 
                        marker_mode=marker_mode,
                        energy_mode=energy_mode, rgb=img)

    elif cfg.model_type == 'np_dist':
        pred_inst = postproc.hover.proc_np_dist(pred_inst)
    elif cfg.model_type == 'dist':
        pred_inst = postproc.dist.process(pred_inst)
    else:
        pred_inst = postproc.other.process(pred_inst, cfg.model_type)

    # ! will be extremely slow on WSI/TMA so it's advisable to comment this out
    # * remap once so that further processing faster (metrics calculation, etc.)
    pred_inst = remap_label(pred_inst, by_size=True)
    overlaid_output = visualize_instances(pred_inst, img)
    # plt.subplot(1,2,1)
    # plt.imshow(pred_inst)
    # plt.subplot(1,2,2)
    # plt.imshow(pred_type)
    # plt.imshow()
    # np.save('/home/test/GhulamMurtaza/pred_maps_class_instance/%s.npy' % (basename),pred_type)
    # np.save('/home/test/GhulamMurtaza/pred_maps_class_instance/%s.npy' % (basename),pred_inst)

    overlaid_output_type = visualize_instances(pred_type,img,color=[[255,0,0],[255,255,0],[0,0,225],[0,128,0],[100,122,56]])
    overlaid_output_true = visualize_instances(true_mask_type,img,color=[[255,0,0],[255,255,0],[0,0,225],[0,128,0],[100,122,56]])
    overlaid_output = cv2.cvtColor(overlaid_output, cv2.COLOR_BGR2RGB)
    overlaid_output_type = cv2.cvtColor(overlaid_output_type,cv2.COLOR_BGR2RGB)
    overlaid_output_true = cv2.cvtColor(overlaid_output_true,cv2.COLOR_BGR2RGB)

    cv2.imwrite('%s/%s.png' % (proc_dir, basename), overlaid_output)
    cv2.imwrite('%s/%s_type.png' % (proc_dir, basename), overlaid_output_type)
    cv2.imwrite('%s/%s_true_mask.png' % (proc_dir, basename), overlaid_output_true)


    # for instance segmentation only
    if cfg.type_classification:                   
        #### * Get class of each instance id, stored at index id-1
        pred_id_list = list(np.unique(pred_inst))[1:] # exclude background ID
        pred_inst_type = np.full(len(pred_id_list), 0, dtype=np.int32)
        for idx, inst_id in enumerate(pred_id_list):
            inst_type = pred_type[pred_inst == inst_id]
            type_list, type_pixels = np.unique(inst_type, return_counts=True)
            type_list = list(zip(type_list, type_pixels))
            type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
            inst_type = type_list[0][0]
            if inst_type == 0: # ! pick the 2nd most dominant if exist
                if len(type_list) > 1:
                    inst_type = type_list[1][0]
                else:
                    print('[Warn] Instance has `background` type' )
            pred_inst_type[idx] = inst_type
        pred_inst_centroid = get_inst_centroid(pred_inst)

    #     sio.savemat('%s/%s.mat' % (proc_dir, basename), 
    #                 {'inst_map'  :     pred_inst,
    #                     'type_map'  :     pred_type,
    #                     'inst_type' :     pred_inst_type[:, None], 
    #                     'inst_centroid' : pred_inst_centroid,
    #                 })
    # else:
    #     sio.savemat('%s/%s.mat' % (proc_dir, basename), 
    #                 {'inst_map'  : pred_inst})

    # ##
    # print('FINISH')
