import argparse
import os
import sys
import cv2
import math
import numpy as np 
import random
import datetime
import cvbase as cvb
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F

from models import resnet_models
from models import FlowNet2
from models import DeepFill
from tools.frame_inpaint import DeepFillv1
from core.dataset import dataset
from core.utils import set_device, get_clear_state_dict, propagation


parser = argparse.ArgumentParser(description="deep-flow-guided")
parser.add_argument("-b", type=int, default=1)
parser.add_argument("-e", type=int, default=0)
parser.add_argument("-n", type=str, default='youtube-vos') 
parser.add_argument("-m", type=str, default='fixed') 
args = parser.parse_args()
class Object():
  pass

PRETRAINED_MODEL_flownet2 = './pretrained_models/FlowNet2_checkpoint.pth.tar'
PRETRAINED_MODEL_inpaint = './pretrained_models/imagenet_deepfill.pth'
PRETRAINED_MODEL_dfc ='./pretrained_models/resnet50_stage1.pth'
flow_args = Object() 
flow_args.fp16 = False
flow_args.rgb_max = 255.
# set random seed 
seed = 2020
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


DATA_NAME = args.n 
MASK_TYPE = args.m
K = 5
IMG_SIZE = (424, 240)
imgw, imgh = IMG_SIZE
FLOW_SIZE = (448, 256)
default_fps = 6

def main_worker(gpu, ngpus_per_node):
        comp_rflo.append(pred_rflo * masks_[idx] + tmp_rflo[K] * (1. - masks_[idx]))
      # flow_guided_propagation
      frames = propagation(deepfill, comp_flo, comp_rflo, gts_, masks_, os.path.join(save_path, info_['vnames'][0]))




if __name__ == '__main__':
  ngpus_per_node = torch.cuda.device_count()
  print('Using {} GPUs for testing {}_{}... '.format(ngpus_per_node, DATA_NAME, MASK_TYPE))
  processes = []
  mp.set_start_method('spawn', force=True)
  for rank in range(ngpus_per_node):
    p = mp.Process(target=main_worker, args=(rank, ngpus_per_node))
    p.start()
    processes.append(p)
  for p in processes:
    p.join()
  print('Finished testing for {}_{}'.format(DATA_NAME, MASK_TYPE))
