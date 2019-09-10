from __future__ import division
import torch
from torch.utils import data

# general libs
import cv2
import io
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import tqdm
import os
import random
import argparse
import glob
import json

from scipy import ndimage, signal
import pdb
import zipfile


class ZipReader(object):
  file_dict = dict()
  def __init__(self):
    super(ZipReader, self).__init__()

  @staticmethod
  def build_file_dict(path):
    file_dict = ZipReader.file_dict
    if path in file_dict:
      return file_dict[path]
    else:
      file_handle = zipfile.ZipFile(path, 'r')
      file_dict[path] = file_handle
      return file_dict[path]

  @staticmethod
  def imread(path, image_name):
    zfile = ZipReader.build_file_dict(path)
    data = zfile.read(image_name)
    im = Image.open(io.BytesIO(data))
    return im

class dataset(data.Dataset):
  def __init__(self, data_name, mask_type, flow_size=(448, 256), img_size=(424, 240)):
    with open(os.path.join('../flist', data_name, 'test.json'), 'r') as f:
      self.video_dict = json.load(f)
    self.videos = list(self.video_dict.keys())
    with open(os.path.join('../flist', data_name, 'mask.json'), 'r') as f:
      self.mask_dict = json.load(f)
    self.masks = list(self.mask_dict.keys())
    self.flow_size = flow_size
    self.img_size = img_size
    self.mask_type = mask_type
    self.data_name = data_name

  def __len__(self):
    return len(self.videos)
  
  def set_subset(self, start, end):
    self.videos = self.videos[start:end]


  def __getitem__(self, index):
    info = {}
    video = self.videos[index]
    frame_names = self.video_dict[video]
    info['vnames'] = video
    info['fnames'] = frame_names
    
    gts = []
    inps = []
    masks = []
    
    for f, name in enumerate(frame_names):
      image_ = ZipReader.imread('../datazip/{}/JPEGImages/{}.zip'.format(self.data_name, video), name)
      image_ = cv2.resize(np.array(image_), self.img_size, cv2.INTER_CUBIC)
      gts.append(torch.from_numpy(np.array(image_)).permute(2,0,1).contiguous().float())

      mask_ = self._get_masks(self.img_size, index, video, f)
      mask_ = cv2.resize(mask_, self.img_size, cv2.INTER_NEAREST)
      masks.append(torch.from_numpy(mask_).float().unsqueeze(0))
      mask_ = torch.from_numpy(cv2.resize(mask_, self.flow_size, cv2.INTER_NEAREST)).float()
      image_ = torch.from_numpy(cv2.resize(image_, self.flow_size, cv2.INTER_CUBIC)).permute(2,0,1).contiguous().float()
      inps.append(image_)# * (1.-mask_))

    return inps, masks, gts, info


  def _get_masks(self, size, index, video, i):
    h, w = size
    if self.mask_type == 'fixed':
      m = np.zeros((h,w), np.uint8)
      m[h//2-h//8:h//2+h//8, w//2-w//8:w//2+w//8] = 1
      return m
    elif self.mask_type == 'object':
      m_name = self.mask_dict[video][i]
      m = ZipReader.imread('../datazip/{}/Annotations/{}.zip'.format(self.data_name, video), m_name).convert('L')
      m = np.array(m)
      m = np.array(m>0).astype(np.uint8)
      m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)), iterations=4).astype(np.float32)
      return m
    else:
      raise NotImplementedError(f"Mask type {self.mask_type} not exists")

