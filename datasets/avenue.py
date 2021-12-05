import os
import sys
sys.path.append(os.path.realpath('.'))

import json
import pickle as pkl
import numpy as np
from PIL import Image
import torch
from torch.utils import data
# from bitrap.structures.trajectory_ops import * 
# from datasets.JAAD_origin import JAAD
# from . import transforms as T
# from bitrap.utils.dataset_utils import bbox_to_goal_map
import copy
import glob
import time
import pdb
from load_avenue_data import Files_Load, Boxes, norm_train_max_min
from config_for_my_data import hyparams, loc, exp

import argparse
# from configs import cfg



class Avenue(data.Dataset):
    def __init__(self, cfg, split, train_file, test_file):
        self.split = split
        # self.root = cfg.DATASET.ROOT
        self.cfg = cfg

        location = Files_Load(train_file,test_file)

        if self.split == 'train' or self.split =='val':
            traindict = Boxes(  loc_files = location['files_train'], 
                                txt_names = location['txt_train'],
                                input_seq = hyparams['input_seq'],
                                pred_seq = hyparams['pred_seq'],
                                data_consecutive = exp['data_consecutive'], 
                                pad = 'pre', 
                                to_xywh = hyparams['to_xywh'],
                                testing = False
                                )
        else:                
            testdict = Boxes(   loc_files = location['files_test'], 
                                txt_names = location['txt_test'],
                                input_seq = hyparams['input_seq'],
                                pred_seq = hyparams['pred_seq'], 
                                data_consecutive = exp['data_consecutive'],
                                pad = 'pre',
                                to_xywh = hyparams['to_xywh'],
                                testing = True,
                                # window = hyparams['input_seq'] # Intention to offset by input=output seq for ablation study
                                )
            


        if self.split == 'train':
            np.random.seed(49)
            rand = np.random.permutation(len(traindict['x_ppl_box']))
            step = int(len(traindict['x_ppl_box'])*0.7)
            train = {}
            for key in traindict:
                train[key] = traindict[key][rand][:step]
            
            self.data = train
            self.xx = self.convert_normalize_bboxes(self.data['x_ppl_box'])
            self.yy = self.convert_normalize_bboxes(self.data['y_ppl_box'])
        
        elif self.split == 'val':
            np.random.seed(49)
            rand = np.random.permutation(len(traindict['x_ppl_box']))
            step = int(len(traindict['x_ppl_box'])*0.7)

            val = {}
            for key in traindict:
                val[key] = traindict[key][rand][step:] # to all for the correct batch

            self.data = val
            self.xx = self.convert_normalize_bboxes(self.data['x_ppl_box'])
            self.yy = self.convert_normalize_bboxes(self.data['y_ppl_box'])




        else:
            assert self.split == 'test'
            self.data = testdict
            self.xx = self.convert_normalize_bboxes(self.data['x_ppl_box'])
            self.yy = self.convert_normalize_bboxes(self.data['y_ppl_box'])
                        

    def __getitem__(self, index):
        # temp = self.xx[index]
        # print('befor float tensor count:{}, index:{}, {}'.format(self.count, index, temp))
        # obs_bbox = torch.FloatTensor(self.xx[index])
        # print('after float tensor count:{},index:{}, {}'.format(self.count,index, obs_bbox))
        obs_bbox = torch.FloatTensor(self.xx[index])
        # pred_bbox = torch.FloatTensor(np.expand_dims(self.yy[index], axis=0))
        pred_bbox = torch.FloatTensor(self.yy[index])
        id_x = torch.from_numpy(self.data['id_x'][index])
        id_y = torch.from_numpy(np.array(self.data['id_y'][index]))
        frame_x = torch.from_numpy(self.data['frame_x'][index])
        frame_y = torch.from_numpy(np.array(self.data['frame_y'][index]))
        text = self.data['video_file'][index]
        video_file = torch.from_numpy(np.array(int(text.split('.')[0])))
        # print('video file {}'.format(int(text.split('.')[0])))
        abnormal_ped_input = torch.from_numpy(np.array(self.data['abnormal_ped_input'][index]))
        abnormal_ped_pred = torch.from_numpy(np.array(self.data['abnormal_ped_pred'][index]))
        abnormal_gt_frame = torch.from_numpy(np.array(self.data['abnormal_gt_frame'][index]))


        # obs_bbox = torch.FloatTensor(self.data['obs_bbox'][index])
        # pred_bbox = torch.FloatTensor(self.data['pred_bbox'][index])
        # cur_image_file = self.data['obs_image'][index][-1]
        # pred_resolution = torch.FloatTensor(self.data['pred_resolution'][index])
        # flow_input = torch.FloatTensor(self.data['flow_input'][index])
        ret = {'input_x':obs_bbox, 'target_y':pred_bbox, 'video_file':video_file, 
                'abnormal_ped_pred':abnormal_ped_pred, 'abnormal_gt_frame':abnormal_gt_frame,
                'abnormal_ped_input':abnormal_ped_input, 'id_x':id_x, 
                'id_y':id_y, 'frame_x':frame_x,'frame_y':frame_y}
        # ret = {'input_x':obs_bbox, 'flow_input':flow_input, 
        #        'target_y':pred_bbox, 'cur_image_file':cur_image_file, 'pred_resolution':pred_resolution}
        # ret['timestep'] = int(cur_image_file.split('/')[-1].split('.')[0])
        return ret

    def __len__(self):
        return len(self.data[list(self.data.keys())[0]])
    
    def norm(self):
        xx = (self.data['x_ppl_box'] - self.min1)/(self.max1 - self.min1)
        yy = (self.data['y_ppl_box'] - self.min1)/(self.max1 - self.min1)
        return xx,yy

    def undo_norm(self, data):
        # If data comes in here it is not in a dictornay format
        data = data*(max1-min1) + min1
        return data
    
    def convert_normalize_bboxes(self, all_bboxes):
        '''input box type is x1y1x2y2 in original resolution'''
        _min = np.array(self.cfg.DATASET.MIN_BBOX)[None, :]
        _max = np.array(self.cfg.DATASET.MAX_BBOX)[None, :]
        # _min = np.array([0,0,0,0])[None, :]
        # # # _max = np.array([640, 360, 640, 360])[None,:]
        # # _max = np.array([856,480,856,480])[None,:]
        # _max = np.array([1920, 1080, 1920, 1080])[None,:]

        for i in range(len(all_bboxes)):
            if len(all_bboxes[i]) == 0:
                continue
            bbox = all_bboxes[i]
            bbox = (bbox - _min) / (_max - _min)
            # NOTE ltrb to cxcywh

            # W, H  = all_resolutions[i][0]
            #     bbox = (bbox - _min) / (_max - _min)
            # if self.cfg.DATASET.NORMALIZE == 'zero-one':
            # elif self.cfg.DATASET.NORMALIZE == 'plus-minus-one':
            #     # W, H  = all_resolutions[i][0]
            #     bbox = (2 * (bbox - _min) / (_max - _min)) - 1
            # elif self.cfg.DATASET.NORMALIZE == 'none':
            #     pass
            # else:
            #     raise ValueError(self.cfg.DATASET.NORMALIZE)

            all_bboxes[i] = bbox
        return all_bboxes


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    # parser.add_argument('--gpu', default='0', type=str)
    # parser.add_argument(
    #     "--config_file",
    #     default="",
    #     metavar="FILE",
    #     help="path to config file",
    #     type=str,
    # )
    # parser.add_argument(
    #     "opts",
    #     help="Modify config options using the command-line",
    #     default=None,
    #     nargs=argparse.REMAINDER,
    # )
    # args = parser.parse_args()
    
    # cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    train_file =  loc['data_load']['corridor']['train_file']
    test_file =  loc['data_load']['corridor']['test_file']

    avenue_train = Avenue('cfg','train', train_file, test_file)
    print('done train')
    avenue_val = Avenue('cfg','val', train_file, test_file)
    print('done val')
    avenue_test = Avenue('cfg','test', train_file, test_file)

    i = 0
    for train, val,test in zip(avenue_train, avenue_val, avenue_test):
        
        i +=1
        if i == 3:
            break
        print('*'*20)
    quit()
    

    # for data 
    print('done')