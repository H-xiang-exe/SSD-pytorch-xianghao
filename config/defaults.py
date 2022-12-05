from yacs.config import CfgNode as CN

_C = CN()

# ----------------------------------------------------------------------------------- #
# MODEL
# ----------------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.NUM_CLASSES = 21
_C.MODEL.THRESHOLD = 0.5
_C.MODEL.NEG_POS_RATIO = 3
_C.MODEL.CENTER_VARIANCE = 0.1
_C.MODEL.SIZE_VARIANCE = 0.2
# ----------------------------------------------------------------------------------- #
# PRIORS
# ----------------------------------------------------------------------------------- #
_C.MODEL.PRIORS = CN()
_C.MODEL.PRIORS.FEATURE_MAPS = [38, 19, 10, 5, 3, 1]
_C.MODEL.PRIORS.STRIDES = [8, 16, 32, 64, 100, 300]
_C.MODEL.PRIORS.MIN_SIZES = [30, 60, 111, 162, 213, 264]
_C.MODEL.PRIORS.MAX_SIZES = [60, 111, 162, 213, 264, 315]
_C.MODEL.PRIORS.ASPECT_RATIOS = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

# ----------------------------------------------------------------------------------- #
# INPUT
# ----------------------------------------------------------------------------------- #
_C.INPUT = CN()
# Image size
_C.INPUT.IMAGE_SIZE = 300
# Values to be used for image normalization, RGB layout
_C.INPUT.PIXEL_MEAN = (104, 117, 123)  # 顺序是RGB

# ----------------------------------------------------------------------------------- #
# DATASETS
# ----------------------------------------------------------------------------------- #
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()

# ----------------------------------------------------------------------------------- #
# DATALOADER
# ----------------------------------------------------------------------------------- #
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 2
_C.DATALOADER.PIN_MEMORY = True

# ----------------------------------------------------------------------------------- #
# SOLVER
# ----------------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_ITER = 12000
_C.SOLVER.LR_STEPS = [80000, 100000]
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.BATCH_SIZE = 8
_C.SOLVER.LR = 1e-3
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 5e-4

# TEST options
_C.TEST = CN()
_C.TEST.NMS_THRESHOLD = 0.45
_C.TEST.CONFIDENCE_THRESHOLD=0.01