import multiprocessing
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
# # import torch.nn.functional as F
# # import torch.nn.init as init
# # from torch.autograd import Variable
# # from sklearn.metrics import roc_auc_score, accuracy_score
# from sklearn.model_selection import train_test_split
from PIL import Image

from utils import compute_roc_auc, get_cuda_version, get_cudnn_version, get_gpu_name

############# CONSTANTS
MULTI_GPU = False

WIDTH = 1024
HEIGHT = 1024
CHANNELS = 3
LR = 0.0001
EPOCHS = 2
BATCHSIZE = 4

IMAGENET_RGB_MEAN_TORCH = [0.485, 0.456, 0.406]
IMAGENET_RGB_SD_TORCH = [0.229, 0.224, 0.225]

CLASS_NAMES = [
    'infiltration',
    'nodule',
    'consolidation',
    'fibrosis',  # aka - Fibrotic Scarring
    'pleural_thickening',
]

CLASSES = len(CLASS_NAMES)

DIR_PREFIX = './'

DATA_DIR = DIR_PREFIX + 'images'
TRAIN_IMAGES_LIST = DIR_PREFIX + 'labels/train_list.txt'
VALDN_IMAGES_LIST = DIR_PREFIX + 'labels/val_list.txt'
TEST_IMAGES_LIST = DIR_PREFIX + 'labels/test_list.txt'
MODEL_DIR = DIR_PREFIX + 'model/'
MODEL_CHKPTS_DIR = MODEL_DIR + 'checkpoints/'

if not os.path.exists(MODEL_CHKPTS_DIR):
    os.makedirs(MODEL_CHKPTS_DIR)


########################


def show_setup_details():
    print("OS: ", sys.platform)
    print("Python: ", sys.version)
    print("PyTorch: ", torch.__version__)
    print("Numpy: ", np.__version__)
    print("GPU: ", get_gpu_name())
    print(get_cuda_version())
    print("CuDNN Version ", get_cudnn_version())

    CPU_COUNT = multiprocessing.cpu_count()
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


class ChestXrayDataSet(Dataset):
    def __init__(self, data_dir, image_list_file, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        image_names = []
        labels = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name = items[0]
                label = items[1:]
                label = [int(i) for i in label]
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)


def no_augmentation_dataset(data_dir, image_list_file, normalize):
    return ChestXrayDataSet(data_dir=data_dir,
                            image_list_file=image_list_file,
                            transform=transforms.Compose([
                                transforms.Resize(WIDTH),
                                transforms.ToTensor(),  # need to convert image to tensor!
                                normalize,
                            ]))


def get_symbol(out_features=CLASSES, multi_gpu=MULTI_GPU):
    _DEVICE = torch.device('cuda:0')
    model = models.densenet.densenet121(pretrained=True)
    # Replace classifier (FC-1000) with (FC-14)
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, out_features),
        nn.Sigmoid())
    if multi_gpu:
        model = nn.DataParallel(model)
    # CUDA
    model.to(_DEVICE)
    return model


def init_symbol(sym, lr=LR):
    # BCE Loss since classes not mutually exclusive + Sigmoid FC-layer
    cri = nn.BCELoss()
    opt = optim.Adam(sym.parameters(), lr=lr, betas=(0.9, 0.999))
    sch = ReduceLROnPlateau(opt, factor=0.1, patience=5, mode='min')
    return opt, cri, sch


def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    # print("Training epoch")
    loss_val = 0
    for i, (data, target) in enumerate(dataloader):
        # Get samples (both async)
        data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        # Forwards
        output = model(data)
        # Loss
        loss = criterion(output, target)
        # Back-prop
        optimizer.zero_grad()
        # Log the loss (before .backward())
        loss_val += loss.item()
        loss.backward()
        optimizer.step()
    print("Training loss: {0:.4f}".format(loss_val / i))


def valid_epoch(model, dataloader, criterion, phase='valid', cl=CLASSES):
    model.eval()
    if phase == 'testing':
        print("Testing epoch")
    else:
        # print("Validating epoch")
        pass

    # Don't save gradients
    with torch.no_grad():
        if phase == 'testing':
            # pre-allocate predictions
            len_pred = len(dataloader) * (dataloader.batch_size)
            num_lab = len(dataloader.dataset.labels[0])
            out_pred = torch.cuda.FloatTensor(len_pred, num_lab).fill_(0)
        loss_val = 0
        for i, (data, target) in enumerate(dataloader):
            # Get samples
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            # Forwards
            output = model(data)
            # Loss
            loss = criterion(output, target)
            # Log the loss
            loss_val += loss.item()
            # Log for AUC
            if phase == 'testing':
                out_pred[output.size(0) * i:output.size(0) * (1 + i)] = output.data
        # Fina loss
        loss_mean = loss_val / i

    if phase == 'testing':
        out_gt = dataloader.dataset.labels
        out_pred = out_pred.cpu().numpy()[:len(out_gt)]  # Trim padding
        print("Test-Dataset loss: {0:.4f}".format(loss_mean))
        print("Test-Dataset AUC: {0:.4f}".format(compute_roc_auc(out_gt, out_pred, cl)))
    else:
        print("Validation loss: {0:.4f}".format(loss_mean))
    return loss_mean


def save_checkpoint(epoch, model, optimizer, loss, is_best):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print("=> Saving a new best")
        PATH = MODEL_CHKPTS_DIR + 'epoch{0}_loss{1:.0f}_checkpoint.pth.tar'.format(epoch, loss)
        # torch.save(state, PATH)  # save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, PATH)
    else:
        print("=> Validation Accuracy did not improve")


def main():
    show_setup_details()

    # Normalise by imagenet mean/sd
    normalize = transforms.Normalize(IMAGENET_RGB_MEAN_TORCH,
                                     IMAGENET_RGB_SD_TORCH)

    # Dataset for training
    train_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                                     image_list_file=TRAIN_IMAGES_LIST,
                                     transform=transforms.Compose([
                                         transforms.RandomResizedCrop(size=WIDTH),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),  # need to convert image to tensor!
                                         normalize,
                                     ]))

    # Dataset for Validation and Testing
    valid_dataset = no_augmentation_dataset(DATA_DIR, VALDN_IMAGES_LIST, normalize)
    test_dataset = no_augmentation_dataset(DATA_DIR, TEST_IMAGES_LIST, normalize)

    # Optimal to use fewer workers than CPU_COUNT
    # DataLoaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCHSIZE,
                              shuffle=True, num_workers=1, pin_memory=True)
    # Using a bigger batch-size (than BATCHSIZE) for below worsens performance
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCHSIZE,
                              shuffle=False, num_workers=1, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCHSIZE,
                             shuffle=False, num_workers=1, pin_memory=True)

    # Load symbol
    chexnet_sym = get_symbol()

    # Load optimiser, loss
    # Scheduler for LRPlateau is not used
    optimizer, criterion, scheduler = init_symbol(chexnet_sym)

    ############################################################### TRAINING LOOP
    best_loss_val = 10000  # intial value set to a big number

    for j in range(EPOCHS):
        stime = time.time()
        epoch_num = j + 1  # to cater for 0 index in logs
        print('---+---+--- Epoch #{} ---+---+---'.format(epoch_num))
        train_epoch(chexnet_sym, train_loader, optimizer, criterion)
        loss_val = valid_epoch(chexnet_sym, valid_loader, criterion)

        is_best = bool(loss_val < best_loss_val)
        best_loss_val = min(loss_val, best_loss_val)
        save_checkpoint(epoch_num, chexnet_sym, optimizer, loss_val, is_best)

        print("Epoch time: {0:.0f} seconds".format(time.time() - stime))

    ###########################################################################

    ###################################################### TESTING & EVALUATION

    test_loss = valid_epoch(chexnet_sym, test_loader, criterion, 'testing')

    ###########################################################################

    ### LOAD SAVED CHECKPOINT MODEL AND TEST IT
    # chexnet_sym_test = get_symbol()
    # chkpt = torch.load(MODEL_CHKPTS_DIR + 'epoch1_loss0.4352886577447255_checkpoint' + '.pth.tar')
    # chexnet_sym_test.load_state_dict(chkpt['model_state_dict'])
    #
    # optimizer_test, criterion_test, scheduler_test = init_symbol(chexnet_sym_test)
    # optimizer_test.load_state_dict(chkpt['optimizer_state_dict'])
    #
    # test_loss = valid_epoch(chexnet_sym_test, test_loader, criterion, 'testing')
    ###########################

    print('Done')


if __name__ == '__main__':
    main()
