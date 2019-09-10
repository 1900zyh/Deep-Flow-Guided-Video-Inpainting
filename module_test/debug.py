import cv2
import os
import sys
import math
import time
import json
import glob
import argparse
import urllib.request
from PIL import Image, ImageFilter
import importlib
from numpy import random
import numpy as np


def read_frame_from_videos(vname):
  frames = []
  vidcap = cv2.VideoCapture(vname)
  success, image = vidcap.read()
  count = 0
  while success:
    frames.append(image)
    success,image = vidcap.read()
    count += 1
  print(len(frames))
  return frames


if __name__ == '__main__':
  read_frame_from_videos('results/davis_object/flamingo/comp.avi')

      
