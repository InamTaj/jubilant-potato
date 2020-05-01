import os
import sys
import time
import logging

from commons import ChestXrayDataSet, no_augmentation_dataset, get_symbol, init_symbol, \
    train_epoch, valid_epoch, save_checkpoint

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from constants import BATCHSIZE, EPOCHS, LR,  WIDTH, CPU_COUNT, MULTI_GPU, NUM_WORKERS, CLASSES, MODEL_CHKPTS_DIR, \
    MODEL_LOGS_DIR, DATA_DIR, TRAIN_IMAGES_LIST, VALDN_IMAGES_LIST, transform_normalizer, PRETRAINED_MODEL_PATH

from utils import get_gpu_name

############# SYSTEM INFO AND GPU SETUP
print("OS: ", sys.platform)
print("Python: ", sys.version)
print("PyTorch: ", torch.__version__)
print("Numpy: ", np.__version__)
print("GPU: ", get_gpu_name())
GPU_COUNT = len(get_gpu_name())
print("CPUs: ", CPU_COUNT)
print("GPUs: ", GPU_COUNT)

# Manually scale to multi-gpu
assert torch.cuda.is_available()
_DEVICE = torch.device('cuda:0')
# enables cudnn's auto-tuner
torch.backends.cudnn.benchmark = True
if MULTI_GPU:
    LR *= GPU_COUNT
    BATCHSIZE *= GPU_COUNT
###################################### MODEL PATHS CONFIGS


# ----------------------------
if not os.path.exists(MODEL_CHKPTS_DIR):
    os.makedirs(MODEL_CHKPTS_DIR)

######################## LOGGING CONFIGS
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler(MODEL_LOGS_DIR)
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)

# add the file handler to the logger
logger.addHandler(handler)
########################


def main_train():

    # Dataset for training
    train_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                                     image_list_file=TRAIN_IMAGES_LIST,
                                     transform=transforms.Compose([
                                         transforms.RandomResizedCrop(size=WIDTH),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),  # need to convert image to tensor!
                                         transform_normalizer,   # from constants
                                     ]))

    # Dataset for Validation and Testing
    valid_dataset = no_augmentation_dataset(DATA_DIR, VALDN_IMAGES_LIST, transform_normalizer)


    # DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCHSIZE,
                              shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    # Using a bigger batch-size (than BATCHSIZE) for below worsens performance
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCHSIZE,
                              shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Load symbol
    chexnet_sym = get_symbol(CLASSES, MULTI_GPU)

    # Load optimiser, loss
    # Scheduler for LRPlateau is not used
    optimizer, criterion, scheduler = init_symbol(chexnet_sym)

    # Load pre-trained weights and optimizer if available
    if PRETRAINED_MODEL_PATH != None:
        try:
            # loads checkpoint file
            chkpt = torch.load(PRETRAINED_MODEL_PATH)
            # loads model weights
            chexnet_sym.load_state_dict(chkpt['model_state_dict'])
            # loads optimizer
            optimizer.load_state_dict(chkpt['optimizer_state_dict'])
            # notify
            print('=> Loaded pre-trained model successfully from: {0}'.format(PRETRAINED_MODEL_PATH))
        except Exception as ex:
            print(ex)
    else:
        print('=> Could not find pre-trained weights at: {0}'.format(PRETRAINED_MODEL_PATH))
        print('=> Starting Training from scratch!')

    ############################################################### TRAINING LOOP
    best_loss_val = 10000  # initial value set to a big number

    for j in range(EPOCHS):
        stime = time.time()
        epoch_num = j + 1  # to cater for 0 index in logs
        current_lr = scheduler.get_lr()
        print('---+---+--- Epoch #{0} of {1} | LR: {2} ---+---+---'.format(epoch_num, EPOCHS, current_lr))
        train_epoch(chexnet_sym, train_loader, optimizer, criterion, epoch_num, logger)
        loss_val = valid_epoch(chexnet_sym, valid_loader, criterion)

        is_best = bool(loss_val < best_loss_val)
        best_loss_val = min(loss_val, best_loss_val)
        save_checkpoint(epoch_num, chexnet_sym, optimizer, loss_val, is_best, current_lr, logger)
        # decay Learning Rate
        # Note that step should be called after validate()
        scheduler.step(best_loss_val)

        print("Epoch time: {0:.0f} seconds".format(time.time() - stime))

    print('Done')


if __name__ == '__main__':
    main_train()
