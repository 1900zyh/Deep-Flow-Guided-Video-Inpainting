
import cv2
import os
import numpy as np
import torch
import random
from collections import OrderedDict
from utils.flow import get_warp_label
from PIL import Image, ImageOps, ImageDraw,ImageFilter


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



def get_video_masks_by_moving_random_stroke(
    video_len, imageWidth=424, imageHeight=240, nStroke=3,
    nVertexBound=[5, 20], maxHeadSpeed=15, maxHeadAcceleration=(15, 3.14),
    brushWidthBound=(30, 50), boarderGap=50, nMovePointRatio=0.5, maxPiontMove=10,
    maxLineAcceleration=(5,0.5), maxInitSpeed=10
):
    '''
    Get video masks by random strokes which move randomly between each
    frame, including the whole stroke and its control points
    Parameters
    ----------
        imageWidth: Image width
        imageHeight: Image height
        nStroke: Number of drawed lines
        nVertexBound: Lower/upper bound of number of control points for each line
        maxHeadSpeed: Max head speed when creating control points
        maxHeadAcceleration: Max acceleration applying on the current head point (
            a head point and its velosity decides the next point)
        brushWidthBound (min, max): Bound of width for each stroke
        boarderGap: The minimum gap between image boarder and drawed lines
        nMovePointRatio: The ratio of control points to move for next frames
        maxPiontMove: The magnitude of movement for control points for next frames
        maxLineAcceleration: The magnitude of acceleration for the whole line
    Examples
    ----------
        object_like_setting = {
            "nVertexBound": [5, 20],
            "maxHeadSpeed": 15,
            "maxHeadAcceleration": (15, 3.14),
            "brushWidthBound": (30, 50),
            "nMovePointRatio": 0.5,
            "maxPiontMove": 10,
            "maxLineAcceleration": (5, 0.5),
            "boarderGap": 20,
            "maxInitSpeed": 10,
        }
        rand_curve_setting = {
            "nVertexBound": [10, 30],
            "maxHeadSpeed": 20,
            "maxHeadAcceleration": (15, 0.5),
            "brushWidthBound": (3, 10),
            "nMovePointRatio": 0.5,
            "maxPiontMove": 3,
            "maxLineAcceleration": (5, 0.5),
            "boarderGap": 20,
            "maxInitSpeed": 6
        }
        get_video_masks_by_moving_random_stroke(video_len=5, nStroke=3, **object_like_setting)
    '''
    assert(video_len >= 1)

    # Initilize a set of control points to draw the first mask
    mask = Image.new(mode='1', size=(imageWidth, imageHeight), color=0)
    control_points_set = []
    for _ in range(nStroke):
      brushWidth = np.random.randint(brushWidthBound[0], brushWidthBound[1])
      Xs, Ys, velocity = get_random_stroke_control_points(
        imageWidth=imageWidth, imageHeight=imageHeight,
        nVertexBound=nVertexBound, maxHeadSpeed=maxHeadSpeed,
        maxHeadAcceleration=maxHeadAcceleration, boarderGap=boarderGap,
        maxInitSpeed=maxInitSpeed)
      control_points_set.append((Xs, Ys, velocity, brushWidth))
      draw_mask_by_control_points(mask, Xs, Ys, brushWidth, fill=255)

    # Generate the following masks by randomly move strokes and their control points
    masks = [mask]
    for _ in range(video_len - 1):
      mask = Image.new(mode='1', size=(imageWidth, imageHeight), color=0)
      for j in range(len(control_points_set)):
        Xs, Ys, velocity, brushWidth = control_points_set[j]
        new_Xs, new_Ys, velocity = random_move_control_points(
          Xs, Ys, imageWidth, imageHeight, velocity, nMovePointRatio, maxPiontMove,
          maxLineAcceleration, boarderGap, maxInitSpeed)
        control_points_set[j] = (new_Xs, new_Ys, velocity, brushWidth)
      for Xs, Ys, velocity, brushWidth in control_points_set:
        draw_mask_by_control_points(mask, Xs, Ys, brushWidth, fill=255)
      masks.append(mask)
    return masks


def random_accelerate(velocity, maxAcceleration, dist='uniform'):
    speed, angle = velocity
    d_speed, d_angle = maxAcceleration

    if dist == 'uniform':
        speed += np.random.uniform(-d_speed, d_speed)
        angle += np.random.uniform(-d_angle, d_angle)
    elif dist == 'guassian':
        speed += np.random.normal(0, d_speed / 2)
        angle += np.random.normal(0, d_angle / 2)
    else:
        raise NotImplementedError(f'Distribution type {dist} is not supported.')

    return (speed, angle)


def random_move_control_points(Xs, Ys, imageWidth, imageHeight, lineVelocity, nMovePointRatio, maxPiontMove, maxLineAcceleration, boarderGap=15, maxInitSpeed=10):
    new_Xs = Xs.copy()
    new_Ys = Ys.copy()

    # move the whole line and accelerate
    speed, angle = lineVelocity
    new_velocity = False
    new_Xs += int(speed * np.cos(angle))
    new_Ys += int(speed * np.sin(angle))
    lineVelocity = random_accelerate(lineVelocity, maxLineAcceleration, dist='guassian')

    # choose points to move
    chosen = np.arange(len(Xs))
    np.random.shuffle(chosen)
    chosen = chosen[:int(len(Xs) * nMovePointRatio)]
    for i in chosen:
        new_Xs[i] += np.random.randint(-maxPiontMove, maxPiontMove)
        new_Ys[i] += np.random.randint(-maxPiontMove, maxPiontMove)
        if not new_velocity and ((new_Xs[i] > imageWidth) or (new_Xs[i] < 0) or (new_Ys[i]>imageHeight) or (new_Ys[i]<0)):
          new_velocity = True
        new_Xs[i] = np.clip(new_Xs[i], boarderGap, imageWidth - boarderGap)
        new_Ys[i] = np.clip(new_Ys[i], boarderGap, imageHeight - boarderGap)
    if new_velocity:
      lineVelocity = get_random_velocity(maxInitSpeed, dist='guassian')
    return new_Xs, new_Ys, lineVelocity


def get_random_stroke_control_points(
    imageWidth, imageHeight,
    nVertexBound=(10, 30), maxHeadSpeed=10, maxHeadAcceleration=(5, 0.5), boarderGap=20,
    maxInitSpeed=10
):
    '''
    Implementation the free-form training masks generating algorithm
    proposed by JIAHUI YU et al. in "Free-Form Image Inpainting with Gated Convolution"
    '''
    startX = np.random.randint(imageWidth)
    startY = np.random.randint(imageHeight)
    Xs = [startX]
    Ys = [startY]

    numVertex = np.random.randint(nVertexBound[0], nVertexBound[1])

    angle = np.random.uniform(0, 2 * np.pi)
    speed = np.random.uniform(0, maxHeadSpeed)

    for i in range(numVertex):
        speed, angle = random_accelerate((speed, angle), maxHeadAcceleration)
        speed = np.clip(speed, 0, maxHeadSpeed)

        nextX = startX + speed * np.sin(angle)
        nextY = startY + speed * np.cos(angle)

        if boarderGap is not None:
            nextX = np.clip(nextX, boarderGap, imageWidth - boarderGap)
            nextY = np.clip(nextY, boarderGap, imageHeight - boarderGap)

        startX, startY = nextX, nextY
        Xs.append(nextX)
        Ys.append(nextY)

    velocity = get_random_velocity(maxInitSpeed, dist='guassian')

    return np.array(Xs), np.array(Ys), velocity


def get_random_velocity(max_speed, dist='uniform'):
    if dist == 'uniform':
        speed = np.random.uniform(max_speed)
    elif dist == 'guassian':
        speed = np.abs(np.random.normal(0, max_speed / 2))
    else:
        raise NotImplementedError(f'Distribution type {dist} is not supported.')

    angle = np.random.uniform(0, 2 * np.pi)
    return (speed, angle)


def draw_mask_by_control_points(mask, Xs, Ys, brushWidth, fill=255):
    radius = brushWidth // 2 - 1
    for i in range(1, len(Xs)):
        draw = ImageDraw.Draw(mask)
        startX, startY = Xs[i - 1], Ys[i - 1]
        nextX, nextY = Xs[i], Ys[i]
        draw.line((startX, startY) + (nextX, nextY), fill=fill, width=brushWidth)
    for x, y in zip(Xs, Ys):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)
    return mask


# modified from https://github.com/naoto0804/pytorch-inpainting-with-partial-conv/blob/master/generate_data.py
def get_random_walk_mask(imageWidth=320, imageHeight=180, length=None):
    action_list = [[0, 1], [0, -1], [1, 0], [-1, 0]]
    canvas = np.zeros((imageHeight, imageWidth)).astype("i")
    if length is None:
        length = imageWidth * imageHeight
    x = random.randint(0, imageHeight - 1)
    y = random.randint(0, imageWidth - 1)
    x_list = []
    y_list = []
    for i in range(length):
        r = random.randint(0, len(action_list) - 1)
        x = np.clip(x + action_list[r][0], a_min=0, a_max=imageHeight - 1)
        y = np.clip(y + action_list[r][1], a_min=0, a_max=imageWidth - 1)
        x_list.append(x)
        y_list.append(y)
    canvas[np.array(x_list), np.array(y_list)] = 1
    return Image.fromarray(canvas * 255).convert('1')


def get_masked_ratio(mask):
    """
    Calculate the masked ratio.
    mask: Expected a binary PIL image, where 0 and 1 represent
          masked(invalid) and valid pixel values.
    """
    hist = mask.histogram()
    return hist[0] / np.prod(mask.size)
