
import cv2
import os
import numpy as np
import torch
from collections import OrderedDict
from utils.flow import get_warp_label


IMG_SIZE = (424, 240)
w, h = IMG_SIZE
th_warp=40
DEFAULT_FPS = 6 

def to_img(img):
  img = img.squeeze().cpu().data.numpy().copy()
  if len(img.shape) == 3:
    img = img.transpose(1,2,0)
  return img 

def propagation(deepfill, flo, rflo, images_, masks_, save_path):
  # replicate for the last frame
  #flo.append(flo[-1].clone())
  #rflo.append(rflo[-1].clone())
  #images_.append(images_[-1].clone())
  #masks_.append(masks_[-1].clone())
  # propagate 
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
    image[(label > 0), :] = 0
    results[0][..., 0] = image
    time_stamp[0][label == 0, 0] = 0
    for th in range(1, frames_num):
      if iter_num != 0:
        image = result_pool[th]
        label = label_pool[th]
      else:
        image = to_img(images_[th])
        label = to_img(masks_[th])
      flow1 = flo[th-1][0].permute(1,2,0).data.cpu().numpy()
      flow2 = rflo[th][0].permute(1,2,0).data.cpu().numpy()
      label = (label>0).astype(np.uint8)
      image[(label>0), :] = 0
      temp1 = get_warp_label(flow1, flow2, results[th - 1][..., 0], th=th_warp)
      temp2 = get_warp_label(flow1, flow2, time_stamp[th - 1], th=th_warp, value=-1)[..., 0]
      results[th][..., 0] = temp1
      time_stamp[th][..., 0] = temp2
      results[th][label == 0, :, 0] = image[label == 0, :]
      time_stamp[th][label == 0, 0] = th

    # backward
    if iter_num != 0:
      image = result_pool[-1]
      label = label_pool[-1]
    else:
      image = to_img(images_[frames_num-1])
      label = to_img(masks_[frames_num-1])
    label = (label > 0).astype(np.uint8)
    image[(label > 0), :] = 0
    results[frames_num - 1][..., 1] = image
    time_stamp[frames_num - 1][label == 0, 1] = frames_num - 1
    for th in range(frames_num - 2, -1, -1):
      if iter_num != 0:
        image = result_pool[th]
        label = label_pool[th]
      else:
        image = to_img(images_[th])
        label = to_img(masks_[th])
      flow1 = rflo[th+1][0].permute(1,2,0).data.cpu().numpy()
      flow2 = flo[th][0].permute(1,2,0).data.cpu().numpy()
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
      label_pool[th] = tmp_mask.copy()[:,:,0]
      tmp_label_seq[th] = np.sum(tmp_mask)

    frame_inpaint_seq[tmp_label_seq == 0] = 0
    masked_frame_num = np.sum((frame_inpaint_seq > 0).astype(np.int))
    iter_num += 1

    if masked_frame_num > 0:
      key_frame_ids = get_key_ids(frame_inpaint_seq)
      for idx in key_frame_ids:
        with torch.no_grad():
          img, mask, small_mask  = data_preprocess(result_pool[idx], label_pool[idx])
          img, mask, small_mask = set_device([img, mask, small_mask])
          _, tmp_inpaint_res, _ = deepfill(img, mask, small_mask)
        label_pool[idx] = label_pool[idx] * 0.
        result_pool[idx] = ((tmp_inpaint_res+1.0)*127.5).squeeze().cpu().data.numpy().transpose(1,2,0)
    tmp_label_seq = np.zeros(frames_num - 1)
    for th in range(0, frames_num - 1):
      tmp_label_seq[th] = np.sum(label_pool[th])
    frame_inpaint_seq[tmp_label_seq == 0] = 0
    masked_frame_num = np.sum((frame_inpaint_seq > 0).astype(np.int))
  os.makedirs(save_path, exist_ok=True)
  comp_writer = cv2.VideoWriter(os.path.join(save_path, 'comp.avi'),
    cv2.VideoWriter_fourcc(*"MJPG"), DEFAULT_FPS, IMG_SIZE)
  pred_writer = cv2.VideoWriter(os.path.join(save_path, 'pred.avi'),
    cv2.VideoWriter_fourcc(*"MJPG"), DEFAULT_FPS, IMG_SIZE)
  mask_writer = cv2.VideoWriter(os.path.join(save_path, 'mask.avi'),
    cv2.VideoWriter_fourcc(*"MJPG"), DEFAULT_FPS, IMG_SIZE)
  orig_writer = cv2.VideoWriter(os.path.join(save_path, 'orig.avi'),
    cv2.VideoWriter_fourcc(*"MJPG"), DEFAULT_FPS, IMG_SIZE)
  for idx in range(frames_num-1):
    orig = np.array(cv2.cvtColor(to_img(images_[idx]), cv2.COLOR_RGB2BGR)).astype(np.uint8)
    m = np.expand_dims(to_img(masks_[idx]), axis=2).astype(np.uint8)
    pred = cv2.cvtColor(result_pool[idx], cv2.COLOR_RGB2BGR).astype(np.uint8)
    orig_writer.write(orig)
    comp_writer.write(m*pred+(1-m)*orig)
    pred_writer.write(pred)
    mask_writer.write((1-m)*orig+m*255)
  comp_writer.release()
  pred_writer.release()
  mask_writer.release()
  orig_writer.release()
  return result_pool
    

def get_key_ids(seq):
  st_pointer = 0
  end_pointer = len(seq) - 1
  st_status = False
  end_status = False
  key_id_list = []

  for i in range((len(seq)+1) // 2):
    if st_pointer > end_pointer:
      break
    if not st_status and seq[st_pointer] > 0:
      key_id_list.append(st_pointer)
      st_status = not st_status
    elif st_status and seq[st_pointer] <= 0:
      key_id_list.append(st_pointer-1)
      st_status = not st_status
    if not end_status and seq[end_pointer] > 0:
      key_id_list.append(end_pointer)
      end_status = not end_status
    elif end_status and seq[end_pointer] <= 0:
      key_id_list.append(end_pointer+1)
      end_status = not end_status

    st_pointer += 1
    end_pointer -= 1
  return sorted(list(set(key_id_list)))


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
    mask = mask[:,:,0:1]
  else:
    mask = np.expand_dims(mask, axis=2)
  small_mask = cv2.resize(mask, (IMG_SIZE[0]//8, IMG_SIZE[1]//8), interpolation=cv2.INTER_NEAREST)
  img = torch.from_numpy(img).permute(2,0,1).contiguous().float().unsqueeze(0)
  mask = torch.from_numpy(mask).permute(2,0,1).contiguous().float().unsqueeze(0)
  small_mask = torch.from_numpy(small_mask).contiguous().float().unsqueeze(0).unsqueeze(0)
  return img*(1-mask), mask, small_mask

