
import cv2
import numpy as np
import torch
from collections import OrderedDict
from utils.flow import get_warp_label


IMG_SIZE = (424, 240)
w, h = IMG_SIZE
th_warp=40


def to_img(img):
  img = img.squeeze().cpu().data.numpy()
  if len(img.shape) == 3:
    img = img.transpose(1,2,0)
  return img 

def propagation(deepfill, flo, rflo, images_, masks_):
  masked_frame_num = len(masks_)
  frames_num = len(masks_)
  iter_num = 0
  frame_inpaint_seq = np.ones(frames_num-1)
  result_pool = [np.zeros((h,w,3), dtype=np.uint8) for _ in range(frames_num)]
  label_pool = [np.zeros((h,w), dtype=np.uint8) for _ in range(frames_num)]

  while masked_frame_num > 0:
    # forward
    results = [np.zeros((h,w,3,2), dtype=np.uint8) for _ in range(frames_num)]
    time_stamp = [-np.ones((h,w,2), dtype=int) for _ in range(frames_num)]
    if iter_num != 0:
      label = label_pool[0]
      image = result_pool[0]
    else:
      label = to_img(masks_[0])
      image = to_img(images_[0])
    label = (label > 0).astype(np.uint8)
    image[label > 0, :] = 0
    results[0][..., 0] = image
    time_stamp[0][label == 0, 0] = 0
    for th in range(1, frames_num):
      if iter_num == 0:
        image = to_img(images_[th])
        label = to_img(masks_[th])
      else:
        image = result_pool[th]
        label = label_pool[th]
      flow1 = flo[th-1][0].permute(1,2,0).data.cpu().numpy()
      flow2 = flo[th][0].permute(1,2,0).data.cpu().numpy()
      label = (label>0).astype(np.uint8)
      image[(label>0), :] = 0
      temp1 = get_warp_label(flow1, flow2, results[th - 1][..., 0], th=th_warp)
      temp2 = get_warp_label(flow1, flow2, time_stamp[th - 1], th=th_warp, value=-1)[..., 0]
      results[th][..., 0] = temp1
      time_stamp[th][..., 0] = temp2
      results[th][label == 0, :, 0] = image[label == 0, :]
      time_stamp[th][label == 0, 0] = th

    # backward
    results[frames_num - 1][..., 1] = image
    time_stamp[frames_num - 1][label == 0, 1] = frames_num - 1
    for th in range(frames_num - 2, -1, -1):
      flow1 = flo[th][0].permute(1,2,0).data.cpu().numpy()
      flow2 = flo[th+1][0].permute(1,2,0).data.cpu().numpy()
      temp1 = get_warp_label(flow1, flow2, results[th + 1][..., 1], th=th_warp)
      temp2 = get_warp_label(flow1, flow2, time_stamp[th + 1], value=-1, th=th_warp,)[..., 1]
      results[th][..., 1] = temp1
      time_stamp[th][..., 1] = temp2
      results[th][label == 0, :, 1] = image[label == 0, :]
      time_stamp[th][label == 0, 1] = th

    # merge
    tmp_label_seq = np.zeros(frames_num-1)
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

    frame_inpaint_seq[tmp_label_seq == 0] = 0
    masked_frame_num = np.sum((frame_inpaint_seq > 0).astype(np.int))
    print(masked_frame_num)
    iter_num += 1

    # inpaint unseen part
    with torch.no_grad():
      tmp_inpaint_res = deepfill.forward(result_pool[id], label_pool[id])
    label_pool[id] = label_pool[id] * 0.
    result_pool[id] = tmp_inpaint_res
    return result_pool
    

# set parameter to gpu or cpu
def set_device(args):
  if torch.cuda.is_available():
    if isinstance(args, list):
      return (item.cuda() for item in args)
    else:
      return args.cuda()
  return args

def get_clear_state_dict(old_state_dict):
  new_state_dict = OrderedDict()
  for k,v in old_state_dict.items():
    name = k 
    if k.startswith('module.'):
      name = k[7:]
    new_state_dict[name] = v
  return new_state_dict

def data_preprocess(img, mask):
  img = img / 127.5 - 1
  mask = (mask > 0).astype(np.int)
  if len(mask.shape) == 3:
    mask = mask[:, :, 0:1]
  else:
    mask = np.expand_dims(mask, axis=2)
  small_mask = cv2.resize(mask, (IMG_SIZE[1]//8, IMG_SIZE[0]//8), interpolation=cv2.INTER_NEAREST)
  img = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()
  mask = torch.from_numpy(mask).permute(2, 0, 1).contiguous().float()
  small_mask = torch.from_numpy(small_mask).permute(2, 0, 1).contiguous().float()
  return img*(1-mask), mask, small_mask

  # mask = mask.data.numpy()[0]
  # res = res.cpu().data.numpy()[0]
  # res_complete = res * mask + img * (1. - mask)
  # res_complete = (res_complete + 1) * 127.5
  # res_complete = res_complete.transpose(1, 2, 0)
  # return res_complete
