import os
from yacs.config import CfgNode as CN


_C = CN()

_C.LOG_DIR =  '/workspace/YOLOPX/runs/'
_C.GPUS = (0,)     # 显卡数 = len(GPUS)
# _C.GPUS = (0, 1, 2, 3)     # 显卡数 = len(GPUS)
# _C.GPUS = ("cpu")     # 显卡数 = len(GPUS)
_C.WORKERS = 0      # 指数据装载时cpu所使用的线程数，默认为8（注意，一般默使用8的话，会报错~~。原因是爆系统内存）
_C.PIN_MEMORY = True
_C.PRINT_FREQ = 5
_C.AUTO_RESUME =False       # Resume from the last training interrupt
_C.NEED_AUTOANCHOR = False      # Re-select the prior anchor(k-means)    When training from scratch (epoch=0), set it to be ture!
_C.DEBUG = False
_C.num_seg_class = 2

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True



# common params for NETWORK
_C.MODEL = CN(new_allowed=True)
_C.MODEL.NAME = "YOLOPv1" # YOLOPv1, YOLOPX
_C.MODEL.NAME = "YOLOPX" # YOLOPv1, YOLOPX
_C.MODEL.STRU_WITHSHARE = False     #add share_block to segbranch
_C.MODEL.HEADS_NAME = ['']

# MODIFY
if _C.MODEL.NAME == "YOLOPX":
    _C.MODEL.PRETRAINED = '/workspace/YOLOPX/weights/epoch-195_yolopx.pth' # default ""
if _C.MODEL.NAME == "YOLOPv1":
    # _C.MODEL.PRETRAINED = '/workspace/YOLOPX/weights/End-to-end_yolopv1.pth' # default ""
    _C.MODEL.PRETRAINED = '' # default ""


#
_C.MODEL.PRETRAINED_DET = ""
_C.MODEL.IMAGE_SIZE = [640, 640]  # width * height, ex: 192 * 256
_C.MODEL.EXTRA = CN(new_allowed=True)


# loss params
_C.LOSS = CN(new_allowed=True)
_C.LOSS.LOSS_NAME = ''
_C.LOSS.MULTI_HEAD_LAMBDA = None
_C.LOSS.FL_GAMMA = 2.0  # focal loss gamma
_C.LOSS.CLS_POS_WEIGHT = 1.0  # classification loss positive weights
_C.LOSS.OBJ_POS_WEIGHT = 1.0  # object loss positive weights
_C.LOSS.SEG_POS_WEIGHT = 1.0  # segmentation loss positive weights
_C.LOSS.BOX_GAIN = 0.05  # box loss gain
_C.LOSS.CLS_GAIN = 0.5  # classification loss gain
_C.LOSS.OBJ_GAIN = 1.0  # object loss gain
_C.LOSS.DA_SEG_GAIN = 0.3  # driving area segmentation loss gain
_C.LOSS.LL_SEG_GAIN = 0.4  # lane line segmentation loss gain
_C.LOSS.LL_IOU_GAIN = 0.4 # lane line iou loss gain


# DATASET related params
_C.DATASET = CN(new_allowed=True)
# 79863
# _C.DATASET.DATAROOT = '/workspace/bdd100k_79863/images/'       # the path of images folder
# _C.DATASET.LABELROOT = '/workspace/bdd100k_79863/labels/det_20/'      # the path of det_annotations folder
# _C.DATASET.MASKROOT = '/workspace/bdd100k_79863/labels/drivable/masks/'                # the path of da_seg_annotations folder
# _C.DATASET.LANEROOT = '/workspace/bdd100k_79863/labels/lane/masks/'               # the path of ll_seg_annotations folder

# yolop
_C.DATASET.DATAROOT = '/workspace/YOLOPX/v1/bdd100k/images'       # the path of images folder
_C.DATASET.LABELROOT = '/workspace/bdd100k/yolop_train/bdd_det/'      # the path of det_annotations folder
_C.DATASET.MASKROOT = '/workspace/bdd100k/yolop_train/bdd_seg_gt/'                # the path of da_seg_annotations folder
_C.DATASET.LANEROOT = '/workspace/bdd100k/yolop_train/bdd_lane_gt/'    


_C.DATASET.DATASET = 'BddDataset_refactor'
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.TEST_SET = 'val'
_C.DATASET.DATA_FORMAT = 'jpg'
_C.DATASET.SELECT_DATA = False
_C.DATASET.ORG_IMG_SIZE = [720, 1280]

# training data augmentation
_C.DATASET.FLIP = True
_C.DATASET.SCALE_FACTOR = 0.25
_C.DATASET.ROT_FACTOR = 10
_C.DATASET.TRANSLATE = 0.1
_C.DATASET.SHEAR = 0.0
_C.DATASET.COLOR_RGB = False
_C.DATASET.HSV_H = 0.015  # image HSV-Hue augmentation (fraction)
_C.DATASET.HSV_S = 0.7  # image HSV-Saturation augmentation (fraction)
_C.DATASET.HSV_V = 0.4  # image HSV-Value augmentation (fraction)
# TODO: more augmet params to add

# train
_C.TRAIN = CN(new_allowed=True)
_C.TRAIN.LR0 = 0.001  # initial learning rate (SGD=1E-2, Adam=1E-3)
_C.TRAIN.LRF = 0.1  # final OneCycleLR learning rate (lr0 * lrf)
_C.TRAIN.WARMUP_EPOCHS = 3.0
_C.TRAIN.WARMUP_BIASE_LR = 0.1
_C.TRAIN.WARMUP_MOMENTUM = 0.8

_C.TRAIN.OPTIMIZER = 'adamw'
_C.TRAIN.MOMENTUM = 0.937
_C.TRAIN.WD = 0.0005
_C.TRAIN.NESTEROV = True
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0



_C.TRAIN.VAL_FREQ = 5 # for val and save weight
_C.TRAIN.BATCH_SIZE_PER_GPU = 4 #default 32
_C.TRAIN.SHUFFLE = True

_C.TRAIN.IOU_THRESHOLD = 0.2
_C.TRAIN.ANCHOR_THRESHOLD = 4.0

_C.mosaic_rate = 0.0  # 1.0
_C.mixup_rate = 0.15

# if training 3 tasks end-to-end, set all parameters as True
# Alternating optimization
_C.TRAIN.SEG_ONLY = False           # Only train two segmentation branchs
_C.TRAIN.DET_ONLY = False           # Only train detection branch
_C.TRAIN.ENC_SEG_ONLY = False       # Only train encoder and two segmentation branchs
_C.TRAIN.ENC_DET_ONLY = False       # Only train encoder and detection branch

# Single task 
_C.TRAIN.DRIVABLE_ONLY = False      # Only train da_segmentation task
_C.TRAIN.LANE_ONLY = False          # Only train ll_segmentation task
_C.TRAIN.DET_ONLY = False          # Only train detection task

_C.TRAIN.PLOT = False                # 

# testing
_C.TEST = CN(new_allowed=True)
_C.TEST.BATCH_SIZE_PER_GPU = 8 # default 32
_C.TEST.MODEL_FILE = ''
_C.TEST.SAVE_JSON = False
_C.TEST.SAVE_TXT = False
_C.TEST.PLOTS = True # validation ploting figures
_C.TEST.NMS_CONF_THRESHOLD  = 0.001
_C.TEST.NMS_IOU_THRESHOLD  = 0.6



# conflict with other params
_C.USE_GRADNORM = True  # use grad norm to adjust learning rate
_C.DATASET.NUMBER_IMAGE = 100
_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 2 # default 200
_C.MODEL.CONFIG = "/workspace/YOLOPX/v1/lib/config/yolopx.yaml"

def update_config_xy(cfg, args):
    cfg.defrost()
    # cfg.merge_from_file(args.cfg)

    if args.modelDir:
        cfg.OUTPUT_DIR = args.modelDir

    if args.logDir:
        cfg.LOG_DIR = args.logDir

    cfg.freeze()
