import torch
import numpy as np

def iou_eps():
    return 0.000001

def iou_PI():
    return 3.141592

def bbox_wh_iou(wh1, wh2):  #wh1 anchor (2，) wh2 gwh (?, 2)
    #gwh 进行转置 (2, ?)
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    # w2，h2 shape (?,)
    w2, h2 = wh2[0], wh2[1]

    # inter_area (?, )
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)

    # union_area (?, )
    union_area = w1 * h1 + w2 * h2 - inter_area
    #(?, )
    return inter_area / (union_area + iou_eps())


def bbox_iou_n(box1, box2):
    b1_x1, b1_x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
    b1_y1, b1_y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
    b2_x1, b2_x2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2
    b2_y1, b2_y2 = box2[..., 1] - box2[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2

    # 相交处面积
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # 并区域
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = w1 * h1 + w2 * h2 - inter

    i_iou = inter / (union + iou_eps())

    return i_iou  # IoU交并比


# 坐标转换
def to_xxyy(box1, box2):
    b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
    b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
    b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    return b1_x1, b1_x2, b1_y1, b1_y2, b2_x1, b2_x2, b2_y1, b2_y2


# 最小包围框
def c_box(box1, box2):
    b1_min_x, b1_max_x, b1_min_y, b1_max_y, b2_min_x, b2_max_x, b2_min_y, b2_max_y = to_xxyy(box1, box2)
    c_top = torch.min(b1_min_y, b2_min_y)
    c_bot = torch.max(b1_max_y, b2_max_y)
    c_left = torch.min(b1_min_x, b2_min_x)
    c_right = torch.max(b1_max_x, b2_max_x)

    return c_top, c_bot, c_left, c_right


# 相交面积
def inter_box(b1_min_x, b1_max_x, b1_min_y, b1_max_y, b2_min_x, b2_max_x, b2_min_y, b2_max_y):
    # Intersection area
    inter = (torch.min(b1_max_x, b2_max_x) - torch.max(b1_min_x, b2_min_x)).clamp(0) * \
    (torch.min(b1_max_y, b2_max_y) - torch.max(b1_min_y, b2_min_y)).clamp(0)
    return inter


# 相并面积
def union_box(b1_min_x, b1_max_x, b1_min_y, b1_max_y, b2_min_x, b2_max_x, b2_min_y, b2_max_y, inter):
    # Union Area
    w1, h1 = b1_max_x - b1_min_x, b1_max_y - b1_min_y
    w2, h2 = b2_max_x - b2_min_x, b1_max_y - b2_min_y
    union = w1 * h1 + w2 * h2 - inter
    return union


# 计算iou
def iou(box1, box2):
    b1_min_x, b1_max_x, b1_min_y, b1_max_y, b2_min_x, b2_max_x, b2_min_y, b2_max_y = to_xxyy(box1, box2)
    inter = inter_box(b1_min_x, b1_max_x, b1_min_y, b1_max_y, b2_min_x, b2_max_x, b2_min_y, b2_max_y)
    union = union_box(b1_min_x, b1_max_x, b1_min_y, b1_max_y, b2_min_x, b2_max_x, b2_min_y, b2_max_y, inter)
    iou = inter / (union + iou_eps())

    return iou


# 计算giou
def giou(box1, box2):
    c_top, c_bot, c_left, c_right = c_box(box1, box2)

    w = c_right - c_left
    h = c_bot - c_top
    c = w * h

    i_iou = iou(box1, box2)

    b1_min_x, b1_max_x, b1_min_y, b1_max_y, b2_min_x, b2_max_x, b2_min_y, b2_max_y = to_xxyy(box1, box2)
    inter = inter_box(b1_min_x, b1_max_x, b1_min_y, b1_max_y, b2_min_x, b2_max_x, b2_min_y, b2_max_y)
    union = union_box(b1_min_x, b1_max_x, b1_min_y, b1_max_y, b2_min_x, b2_max_x, b2_min_y, b2_max_y, inter)

    giou_term = (c - union) / c

    return i_iou - giou_term


# 计算diou
def diou(box1, box2):
    c_top, c_bot, c_left, c_right = c_box(box1, box2)

    w = c_right - c_left
    h = c_bot - c_top
    c = w * w + h * h

    i_iou = iou(box1, box2)

    d = (box1[:, 0] - box2[:, 0]) * (box1[:, 0] - box2[:, 0]) + (box1[:, 1] - box2[:, 1]) * (box1[:, 1] - box2[:, 1])

    diou_term = torch.pow(d / c, 0.6)

    return i_iou - diou_term


# 计算ciou
def ciou(box1, box2):
    c_top, c_bot, c_left, c_right = c_box(box1, box2)

    w = c_right - c_left
    h = c_bot - c_top
    c = w * w + h * h

    i_iou = iou(box1, box2)

    u = (box1[:, 0] - box2[:, 0]) * (box1[:, 0] - box2[:, 0]) + (box1[:, 1] - box2[:, 1]) * (box1[:, 1] - box2[:, 1])
    d = u / c

    ar_gt = box2[:, 2] / box2[:, 3]
    ar_pred = box1[:, 2] / (box1[:, 3]+ iou_eps())
    ar_loss = 4 / (iou_PI() * iou_PI()) * (torch.atan(ar_gt) - torch.atan(ar_pred)) * (torch.atan(ar_gt) - torch.atan(ar_pred))

    alpha = ar_loss / (1 - i_iou + ar_loss)

    ciou_term = d + alpha * ar_loss

    return i_iou - ciou_term


# 中心坐标转xyxy坐标
def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


# xyxy坐标转中心坐标
def xyxy2xywh(x):
    y = x.new(x.shape)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y


# 中心坐标转xyxy坐标
def xywh2xyxy_np(x):
    y = np.zeros(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


# xyxy坐标转中心坐标
def xyxy2xywh_np(x):
    y = np.zeros(x.shape)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y


def bbox_iou_crop(box1, box2):

    b1_x1, b1_x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
    b1_y1, b1_y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
    b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
    b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # 相交处面积
    inter = np.clip((np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)), 0., float('inf')) * np.clip(
        (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)), 0., float('inf'))

    # 并区域
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = w1 * h1 + w2 * h2 - inter

    i_iou = inter / (union + iou_eps())

    return i_iou  # IoU交并比
