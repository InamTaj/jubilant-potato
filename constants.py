from datetime import datetime
import multiprocessing
import torch
import torchvision.transforms as transforms


LR = 0.0001
BATCHSIZE = 16
EPOCHS = 100

WIDTH = 448
HEIGHT = 448
CHANNELS = 3

CPU_COUNT = multiprocessing.cpu_count()
# Optimal to use fewer workers than CPU_COUNT
NUM_WORKERS = 1 if CPU_COUNT < 2 else 4

# check for multiple GPUs
MULTI_GPU = True if torch.cuda.device_count() > 1 else False


TIMESTAMP = datetime.now().strftime('%d.%m.%Y_%H.%M.%S')

DIR_PREFIX = './'

DATASET_DIR = DIR_PREFIX + 'datasets/chexnet/'

DATA_DIR = DATASET_DIR + 'images'
TRAIN_IMAGES_LIST = DATASET_DIR + 'labels/train_list.txt'
VALDN_IMAGES_LIST = DATASET_DIR + 'labels/val_list.txt'
TEST_IMAGES_LIST = DATASET_DIR + 'labels/test_list.txt'

MODEL_DIR = DIR_PREFIX + 'model/'
MODEL_CHKPTS_DIR = MODEL_DIR + 'checkpoints/'
MODEL_LOGS_DIR   = MODEL_DIR  + 'losses_{}.log'.format(TIMESTAMP)

# should be a complete path,
# PRETRAINED_MODEL_PATH = None
PRETRAINED_MODEL_PATH = MODEL_CHKPTS_DIR + 'checkpoint30.04.2020_15.14.19_epoch13_loss0.1476.pth.tar'

CLASS_NAMES = [
    'infiltration',
    'nodule',
    'consolidation',
    'fibrosis',  # aka - Fibrotic Scarring
    'pleural_thickening',
]

DISEASE_TYPES = [
    'atelectasis',
    'cardiomegaly',
    'effusion',
    'infiltration',
    'mass',
    'nodule',
    'pneumonia',
    'pneumothorax',
    'consolidation',
    'edema',
    'emphysema',
    'fibrosis',   # aka - Fibrotic Scarring
    'pleural_thickening',
    'hernia'
]

CLASSES = len(DISEASE_TYPES)


IMAGENET_RGB_MEAN_TORCH = [0.485, 0.456, 0.406]
IMAGENET_RGB_SD_TORCH = [0.229, 0.224, 0.225]

# Normalise by imagenet mean/sd
transform_normalizer = transforms.Normalize(IMAGENET_RGB_MEAN_TORCH,
                                            IMAGENET_RGB_SD_TORCH)
