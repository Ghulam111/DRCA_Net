import numpy as np
import os
import glob
import scipy.io as sio
from misc.utils import get_inst_centroid, rm_n_mkdir
from metrics.stats_utils import remap_label

ann_dir = '/home/test/GhulamMurtaza/panNuke/Test/Labels/' # * directory contains .npy
filepath_list = glob.glob('%s/*.npy' % ann_dir)

save_dir = 'GroundTruth/dump/' # directory to save summarized info about nuclei

rm_n_mkdir(save_dir)
for path in filepath_list:
    basename = os.path.basename(path).split('.')[0]

    true_map = np.load(path)
    true_inst = true_map[...,0]
    true_type = true_map[...,1]

    true_inst = remap_label(true_inst, by_size=True)
    true_inst_centroid = get_inst_centroid(true_inst)
    #### * Get class of each instance id, stored at index id-1
    # for ground truth instance blob
    true_id_list = list(np.unique(true_inst))[1:] # exclude background
    true_inst_type = np.full(len(true_id_list), -1, dtype=np.int32)
    for idx, inst_id in enumerate(true_id_list):
        inst_type = true_type[true_inst == inst_id]
        type_list, type_pixels = np.unique(inst_type, return_counts=True)
        inst_type = type_list[np.argmax(type_pixels)]
        if inst_type != 0: # there are artifact nuclei (background types)
            true_inst_type[idx] = inst_type

    sio.savemat('%s/%s.mat' % (save_dir, basename), 
                {
                'inst_type' :     true_inst_type[:, None], 
                'inst_centroid' : true_inst_centroid,
                })