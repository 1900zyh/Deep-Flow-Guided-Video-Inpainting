import argparse
import os
import sys
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
from tools.frame_inpaint import DeepFillv1
from core.dataset import dataset


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
FLOW_SIZE = (448, 256)
th_warp=40
default_fps = 6


def to_img(x):
  tmp = (x[0,:,0,:,:].cpu().data.numpy().transpose((1,2,0))+1)/2
  tmp = np.clip(tmp,0,1)*255.
  return tmp.astype(np.uint8)


# set parameter to gpu or cpu
def set_device(args):
  if torch.cuda.is_available():
    if isinstance(args, list):
      return (item.cuda() for item in args)
    else:
      return args.cuda()
  return args

def get_clear_state_dict(old_state_dict):
  from collections import OrderedDict
  new_state_dict = OrderedDict()
  for k,v in old_state_dict.items():
    name = k 
    if k.startswith('module.'):
      name = k[7:]
    new_state_dict[name] = v
  return new_state_dict


def propagation(flo, rflo, images, masks):
  while masked_frame_num > 0:
    # forward
    results = [np.zeros(image.shape + (2,), dtype=image.dtype) for _ in range(frames_num)]
    time_stamp = [-np.ones(image.shape[:2] + (2,), dtype=int) for _ in range(frames_num)]
    label = (label > 0).astype(np.uint8)
    image[label > 0, :] = 0
    results[0][..., 0] = image
    time_stamp[0][label == 0, 0] = 0
    for th in range(1, frames_num):
      flow1 = flo[th]
      flow2 = flo[th+1]
      temp1 = flo.get_warp_label(flow1, flow2, results[th - 1][..., 0], th=th_warp)
      temp2 = flo.get_warp_label(flow1, flow2, time_stamp[th - 1], th=th_warp, value=-1)[..., 0]
      results[th][..., 0] = temp1
      time_stamp[th][..., 0] = temp2
      results[th][label == 0, :, 0] = image[label == 0, :]
      time_stamp[th][label == 0, 0] = th

    # backward
    results[frames_num - 1][..., 1] = image
    time_stamp[frames_num - 1][label == 0, 1] = frames_num - 1
    for th in range(frames_num - 2, -1, -1):
      flow1 = flo[th]
      flow2 = flo[th+1]
      temp1 = flo.get_warp_label(flow1, flow2, results[th + 1][..., 1], th=th_warp)
      temp2 = flo.get_warp_label(flow1, flow2, time_stamp[th + 1], value=-1, th=th_warp,)[..., 1]
      results[th][..., 1] = temp1
      time_stamp[th][..., 1] = temp2
      results[th][label == 0, :, 1] = image[label == 0, :]
      time_stamp[th][label == 0, 1] = th

    # merge
    for th in range(0, frames_num - 1):
      v1 = (time_stamp[th][..., 0] == -1)
      v2 = (time_stamp[th][..., 1] == -1)
      hole_v = (v1 & v2)
      result = results[th][..., 0].copy()
      result[v1, :] = results[th][v1, :, 1].copy()

      v3 = ((v1 == 0) & (v2 == 0))
      dist = time_stamp[th][..., 1] - time_stamp[th][..., 0]
      dist[dist < 1] = 1

      w2 = (th - time_stamp[th][..., 0]) / dist
      w2 = (w2 > 0.5).astype(np.float)

      result[v3, :] = (results[th][..., 1] * w2[..., np.newaxis] +
                        results[th][..., 0] * (1 - w2)[..., np.newaxis])[v3, :]

      result_pool[th] = result.copy()
      tmp_mask = np.zeros_like(result)
      tmp_mask[hole_v, :] = 255
      label_pool[th] = tmp_mask.copy()
      tmp_label_seq[th] = np.sum(tmp_mask)

      sys.stdout.write('\n')
      frame_inpaint_seq[tmp_label_seq == 0] = 0
      masked_frame_num = np.sum((frame_inpaint_seq > 0).astype(np.int))
      print(masked_frame_num)
      iter_num += 1

    # inpaint unseen part
    with torch.no_grad():
      tmp_inpaint_res = frame_inapint_model.forward(result_pool[id], label_pool[id])
    label_pool[id] = label_pool[id] * 0.
    result_pool[id] = tmp_inpaint_res
    

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
  dfc_resnet = resnet_models.Flow_Branch_Multi(input_chanels=33, NoLabels=2)
  ckpt_dict = torch.load(PRETRAINED_MODEL_dfc, map_location = lambda storage, loc: set_device(storage))
  dfc_resnet.load_state_dict(get_clear_state_dict(ckpt_dict['model']))
  set_device(dfc_resnet)
  dfc_resnet.eval()
  # deepfill: used for image inpainting on unseen part
  deepfill_model = DeepFillv1(pretrained_model=PRETRAINED_MODEL_inpaint, image_shape=FLOW_SIZE)

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
      # extracting flow
      print('[{}] {}/{}: {} for {} frames ...'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
        seq, len(Trainloader), info_['vnames'], length))
      flo = []
      rflo = []
      for idx in range(length):
        id1, id2, id3 = max(0, idx-1), idx, min(idx+1, length-1)
        if id2 < id3:
          f2, f3 = set_device([frames_[id2], frames_[id3]])
          f = Flownet(f2, f3)
          f = nn.Upsample(size=list(IMG_SIZE)[::-1], mode='bilinear', align_corners=True)(f)
          f[0,0,...] = f[0,0,...].clamp(-1. * FLOW_SIZE[1], FLOW_SIZE[1]) / FLOW_SIZE[1] * IMG_SIZE[1]
          f[0,1,...] = f[0,1,...].clamp(-1. * FLOW_SIZE[0], FLOW_SIZE[0]) / FLOW_SIZE[0] * IMG_SIZE[0]
          flo.append(f)
        if id1 < id2:
          f1, f2 = set_device([frames_[id1], frames_[id2]])
          f = Flownet(f1, f2)
          f = nn.Upsample(size=list(IMG_SIZE)[::-1], mode='bilinear', align_corners=True)(f)
          f[0,0,...] = f[0,0,...].clamp(-1. * FLOW_SIZE[1], FLOW_SIZE[1]) / FLOW_SIZE[1] * IMG_SIZE[1]
          f[0,1,...] = f[0,1,...].clamp(-1. * FLOW_SIZE[0], FLOW_SIZE[0]) / FLOW_SIZE[0] * IMG_SIZE[0]
          rflo.append(f)
      flo.append(flo[-1]*0)
      rflo.insert(0, rflo[0]*0)
      # flow completion 
      comp_flo = []
      comp_rflo = []
      for idx in range(length):
        # flo
        tmp_flo = [flo[0]]*(max(0, K-idx)) + flo[max(0, idx-5):min(idx+5, length)] + [flo[-1]]*(max(0, K+idx-length))
        tmp_rflo = [rflo[0]]*(max(0, len(K-idx))) + rflo[max(0, idx-5):min(idx+5, length)] + [rflo[-1]]*(max(0, K+idx-length))
        tmp_mask = [masks_[0]]*(max(0, K-idx)) + masks_[max(0, idx-5):min(idx+5, length)] + [masks_[-1]]*(max(0, K+idx-length))
        mask_flo = [torch.cat([f,m], dim=1) for f,m in zip(tmp_flo, tmp_mask)]
        mask_rflo = [torch.cat([f,m], dim=1) for f,m in zip(tmp_rflo, tmp_mask)]
        # flo 
        mask_flo = torch.cat(mask_flo, dim=1)
        pred_flo = dfc_resnet(mask_flo)
        comp_flo.append(pred_flo * masks_[idx] + pred_flo * (1. - masks_[idx]))
        # rflo
        mask_rflo = torch.cat(mask_rflo, dim=1)
        pred_rflo = dfc_resnet(mask_rflo)
        comp_flo.append(pred_rflo * masks_[idx] + pred_rflo * (1. - masks_[idx]))
      # # flow_guided_propagation
      # propagation(args, frame_inapint_model=deepfill_model)




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
