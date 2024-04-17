import math
import pdb

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tools
import traceback
import sys
from collections import OrderedDict

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))
    
if __name__ == "__main__":
    # graph = "graph.ntu_rgb_d_hierarchy.Graph"
    graph = "graph.ntu_rgb_d.Graph"
    # graph_args = {'labeling_mode': 'spatial', 'CoM': 1}
    graph_args = {'labeling_mode': 'spatial'}
    # Model = import_class('model.ctrsgcn.Model')
    # Model = import_class('model.ctrgcn.Model')
    # Model = import_class('model.hdgcn.Model')
    Model = import_class('model.MMCL.Model')
    model = Model(num_class = 120, num_point=25, num_person = 2, graph = graph, graph_args = graph_args)
    model_dict = model.state_dict() 
    # weights_files = './ctrsgcn_ntu120_XSubJ.pt' 
    # weights_files = './ctrgcn_ntu120_XSubJ.pt'
    # weights_files = './hdgcn_joint_CoM_1.pt'
    weights_files = './mmcl.pt'
    weights = torch.load(weights_files) 
    match_dict = {k: v for k, v in weights.items() if k in model_dict}
    model_dict.update(match_dict)
    model.load_state_dict(model_dict)
    model.eval()

    root_path = './UM_output/'
    name_list = os.listdir(root_path)
    action_list = ['a3', 'a6', 'a7', 'a24', 'a25', 'a27']
    # 3. wave 22
    # 6. arm cross 95
    # 7. basketball shoot 62
    # 24. Sit to stand 8
    # 25. Stand to sit 7
    # 27. squat 79
    label_list = [22, 95, 62, 8, 7, 79]
    top_1 = 0
    top_5 = 0
    action_right_top_1 = [0, 0, 0, 0, 0, 0]
    action_right_top_5 = [0, 0, 0, 0, 0, 0]
    for i in range(len(name_list)):
        sample = name_list[i]
        action_idx = action_list.index(sample.split('_')[0])
        ske_path = root_path + sample
        data_numpy = np.load(ske_path, allow_pickle = True) # T V C(3)
        data_numpy = torch.from_numpy(data_numpy).permute(2, 0, 1).unsqueeze(3)
        data_numpy = torch.concat((data_numpy, torch.zeros(data_numpy.shape)), dim = 3)
        origin = data_numpy[:, 0, 1, :] # C M
        origin = origin.unsqueeze(1).unsqueeze(2) # C 1 1 M
        origin = np.repeat(np.array(origin), data_numpy.shape[1], axis = 1) # C T 1 M
        origin = np.repeat(origin, data_numpy.shape[2], axis = 2) # C T V M
        data_numpy = np.array(data_numpy) - origin
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, [0.95], 64) # 3 64 25 2
        skeleton_input = torch.from_numpy(data_numpy).float().unsqueeze(0) # 1 3 64 25 2

        # blip_input = torch.from_numpy(np.load('./feature/' + name + '.npy', allow_pickle = True))
        # score, refine_score = model(skeleton_input, blip_input.float(), 'train')
        score = model(skeleton_input)
        max_index1 = np.argmax(score.cpu().detach().numpy())
        score_idx = np.argsort(score.cpu().detach().numpy())
        if (max_index1 == label_list[action_idx]): 
            top_1 += 1
            action_right_top_1[action_idx] += 1
        if (label_list[action_idx] in score_idx[0][-5:]): 
            top_5 += 1
            action_right_top_5[action_idx] += 1
    print("All done!")