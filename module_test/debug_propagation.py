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
import glob


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
IMG_SIZE = (432, 240)
imgw, imgh = IMG_SIZE
FLOW_SIZE = (448, 256)
default_fps = 6


# deepfill: used for image inpainting on unseen part
deepfill = set_device(DeepFill.Generator())
model_weight = torch.load(PRETRAINED_MODEL_inpaint, map_location = lambda storage, loc: set_device(storage))
deepfill.load_state_dict(model_weight)
deepfill.eval()


if __name__ == '__main__':
  # loading flows
  flo_path = 'demo/flamingo/Flow_res/initial_res/'
  flo_list = glob.glob(os.path.join(flo_path, '*.flo'))
  flo_list.sort()
  rflo_list = glob.glob(os.path.join(flo_path, '*.rflo'))
  rflo_list.sort()
  comp_flo = []
  for i, n in enumerate(flo_list):
    flow = cvb.read_flow(n)
    origin_shape = flow.shape
    flow = cv2.resize(flow, (IMG_SIZE))
    flow[:, :, 0] = flow[:, :, 0].clip(-1. * origin_shape[1], origin_shape[1]) / origin_shape[1] * IMG_SIZE[0]
    flow[:, :, 1] = flow[:, :, 1].clip(-1. * origin_shape[0], origin_shape[0]) / origin_shape[0] * IMG_SIZE[1]
    flow = torch.from_numpy(flow).permute(2, 0, 1).contiguous().float().unsqueeze(0)
    comp_flo.append(flow)
  comp_rflo = []
  for i, n in enumerate(rflo_list):
    flow = cvb.read_flow(n)
    origin_shape = flow.shape
    flow = cv2.resize(flow, (IMG_SIZE))
    flow[:, :, 0] = flow[:, :, 0].clip(-1. * origin_shape[1], origin_shape[1]) / origin_shape[1] * IMG_SIZE[0]
    flow[:, :, 1] = flow[:, :, 1].clip(-1. * origin_shape[0], origin_shape[0]) / origin_shape[0] * IMG_SIZE[1]
    flow = torch.from_numpy(flow).permute(2, 0, 1).contiguous().float().unsqueeze(0)
    comp_rflo.append(flow)
  
  img_list = glob.glob(os.path.join('demo/flamingo/frames', '*.jpg'))
  img_list.sort()
  mask_list = glob.glob(os.path.join('demo/flamingo/masks', '*.png'))
  mask_list.sort()
  gts_ = []
  for i, n in enumerate(img_list):
    image_ = cv2.imread(n)[:,:,::-1]
    image_ = cv2.resize(np.array(image_), IMG_SIZE, cv2.INTER_CUBIC)
    gts_.append(torch.from_numpy(np.array(image_)).permute(2,0,1).contiguous().float().unsqueeze(0))
  masks_ = []
  for i, n in enumerate(mask_list):
    m = cv2.imread(n)[:,:,2]
    m = np.array(m>0).astype(np.uint8)
    m = cv2.resize(np.array(m), IMG_SIZE, cv2.INTER_NEAREST)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)), iterations=4).astype(np.float32)
    masks_.append(torch.from_numpy(np.array(m)).contiguous().float().unsqueeze(0))
  print(len(comp_flo), len(comp_rflo), len(gts_), len(masks_))
  print('begin propagation')
  propagation(deepfill, comp_flo, comp_rflo, gts_, masks_, 'flamingo')



