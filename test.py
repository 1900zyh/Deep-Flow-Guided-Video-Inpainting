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

from utils import region_fill as rf

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
PRETRAINED_MODEL_dfc ='./pretrained_models/resnet101_movie.pth'
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

def main_worker(gpu, ngpus_per_node):
  if ngpus_per_node > 1:
    torch.cuda.set_device(int(gpu))
  # model preparation
  # flownet: used for extracting optical flow
  Flownet = FlowNet2(flow_args, requires_grad=False)
  flownet2_ckpt = torch.load(PRETRAINED_MODEL_flownet2, map_location = lambda storage, loc: set_device(storage))
  Flownet.load_state_dict(get_clear_state_dict(flownet2_ckpt['state_dict']))
  set_device(Flownet)
  Flownet.eval()
  # dfc_resnet: used for completing optical_flow
  dfc_resnet = resnet_models.Flow_Branch(33,2)#_Multi(input_chanels=33, NoLabels=2)
  ckpt_dict = torch.load(PRETRAINED_MODEL_dfc, map_location = lambda storage, loc: set_device(storage))
  dfc_resnet.load_state_dict(get_clear_state_dict(ckpt_dict['model']))
  set_device(dfc_resnet)
  dfc_resnet.eval()
  # deepfill: used for image inpainting on unseen part
  deepfill = set_device(DeepFill.Generator())
  model_weight = torch.load(PRETRAINED_MODEL_inpaint, map_location = lambda storage, loc: set_device(storage))
  deepfill.load_state_dict(model_weight)
  deepfill.eval()

  # dataset 
  DTset = dataset(DATA_NAME, MASK_TYPE, flow_size=FLOW_SIZE, img_size=IMG_SIZE)
  step = math.ceil(len(DTset) / ngpus_per_node)
  DTset.set_subset(gpu*step, min(gpu*step+step, len(DTset)))
  Trainloader = torch.utils.data.DataLoader(DTset, batch_size=1, shuffle=False, num_workers=1)
  save_path = 'results/{}_{}'.format(DATA_NAME, MASK_TYPE)
  print('GPU-{}: finished building models, begin test for {} ...'.format(
    gpu, save_path))

  with torch.no_grad():
    for seq, (frames_, masks_, gts_, info_) in enumerate(Trainloader):
      length = len(frames_)
      masks_ = list(set_device(list(masks_)))
      # extracting flow
      print('[{}] GPU-{}: {}/{} {} for {} frames ...'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
        gpu, seq, len(Trainloader), info_['vnames'], length))
      flo = []
      rflo = []
      for idx in range(length):
        id1, id2, id3 = max(0, idx-1), idx, min(idx+1, length-1)
        if id2 < id3:
          f2, f3 = set_device([frames_[id2], frames_[id3]])
          f = Flownet(f2, f3)
          f = nn.Upsample(size=(imgh,imgw), mode='bilinear', align_corners=True)(f)
          f[0,0,...] = f[0,0,...].clamp(-1. * FLOW_SIZE[0], FLOW_SIZE[0]) / FLOW_SIZE[0] * IMG_SIZE[0]
          f[0,1,...] = f[0,1,...].clamp(-1. * FLOW_SIZE[1], FLOW_SIZE[1]) / FLOW_SIZE[1] * IMG_SIZE[1]
          flo.append(f)
        if id1 < id2:
          f1, f2 = set_device([frames_[id1], frames_[id2]])
          f = Flownet(f2, f1)
          f = nn.Upsample(size=(imgh,imgw), mode='bilinear', align_corners=True)(f)
          f[0,0,...] = f[0,0,...].clamp(-1. * FLOW_SIZE[0], FLOW_SIZE[0]) / FLOW_SIZE[0] * IMG_SIZE[0]
          f[0,1,...] = f[0,1,...].clamp(-1. * FLOW_SIZE[1], FLOW_SIZE[1]) / FLOW_SIZE[1] * IMG_SIZE[1]
          rflo.append(f)
      rflo.insert(0, rflo[0].clone()*0)
      # flow completion 
      comp_flo = []
      comp_rflo = []
      for idx in range(length):
        # flo
        tmp_flo = [flo[0]]*(max(0, K-idx)) + flo[max(0, idx-K):min(idx+K+1, length)] + [flo[-1]]*(max(0, K+idx+1-length+1))
        tmp_rflo = [rflo[0]]*(max(0, K-idx)) + rflo[max(0, idx-K):min(idx+K+1, length)] + [rflo[-1]]*(max(0, K+idx+1-length))
        tmp_mask = [masks_[0]]*(max(0, K-idx)) + masks_[max(0, idx-K):min(idx+K+1, length)] + [masks_[-1]]*(max(0, K+idx+1-length))
        # initial holes
        for fi in range(len(tmp_flo)):
          np_mask = tmp_mask[fi][0].permute(1,2,0).cpu().data.numpy()
          np_flo = tmp_flo[fi][0].permute(1,2,0).data.cpu().numpy() * (1.-np_mask)
          np_flo[:, :, 0] = np_flo[:, :, 0] + np_mask[:,:,0] * rf.regionfill(np_flo[:, :, 0], np_mask[:,:,0])
          np_flo[:, :, 1] = np_flo[:, :, 1] + np_mask[:,:,0] * rf.regionfill(np_flo[:, :, 1], np_mask[:,:,0])
          tmp_flo[fi] = set_device(torch.from_numpy(np_flo).permute(2, 0, 1).contiguous().float().unsqueeze(0))
          np_rflo = tmp_rflo[fi][0].permute(1,2,0).data.cpu().numpy() * (1.-np_mask)
          np_rflo[:, :, 0] = np_rflo[:, :, 0] + np_mask[:,:,0] * rf.regionfill(np_rflo[:, :, 0], np_mask[:,:,0])
          np_rflo[:, :, 1] = np_rflo[:, :, 1] + np_mask[:,:,0] * rf.regionfill(np_rflo[:, :, 1], np_mask[:,:,0])
          tmp_rflo[fi] = set_device(torch.from_numpy(np_rflo).permute(2, 0, 1).contiguous().float().unsqueeze(0))
        mask_flo = [torch.cat([f,m], dim=1) for f,m in zip(tmp_flo, tmp_mask)]
        mask_rflo =[torch.cat([f,m], dim=1) for f,m in zip(tmp_rflo, tmp_mask)]
        # flo 
        mask_flo = torch.cat(mask_flo, dim=1)
        pred_flo = dfc_resnet(mask_flo)
        comp_flo.append(pred_flo * masks_[idx] + tmp_flo[K] * (1. - masks_[idx]))
        # rflo
        mask_rflo = torch.cat(mask_rflo, dim=1)
        pred_rflo = dfc_resnet(mask_rflo)
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
