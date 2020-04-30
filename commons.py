import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchsummary import summary

from PIL import Image

from constants import LR, WIDTH, HEIGHT, CHANNELS, MULTI_GPU, CLASSES, DISEASE_TYPES, TIMESTAMP, MODEL_CHKPTS_DIR
from utils import compute_roc_auc, get_classification_report


SHOW_MODEL = False

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

def init_symbol(sym, lr=LR):
    # BCE Loss since classes not mutually exclusive + Sigmoid FC-layer
    cri = nn.BCELoss()
    opt = optim.Adam(sym.parameters(), lr=lr, betas=(0.9, 0.999))
    sch = ReduceLROnPlateau(opt, factor=0.1, patience=5, mode='min')
    return opt, cri, sch


def get_symbol(out_features=CLASSES, multi_gpu=MULTI_GPU):
    _DEVICE = torch.device('cuda:0')
    model = models.densenet.densenet121(pretrained=True)
    # Replace classifier (FC-1000) with (FC-14)
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, out_features),
        nn.Sigmoid())
    if multi_gpu:
        print('==> expanding to multiple GPUs !!')
        model = nn.DataParallel(model)
    # CUDA
    model.to(_DEVICE)

    if SHOW_MODEL == True:
        summary(model, input_data=(CHANNELS, HEIGHT, WIDTH))

    return model


def train_epoch(model, dataloader, optimizer, criterion, epoch_num, logger):
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

    logger.info('epoch:{0},train:{1:.4f}'.format(epoch_num, (loss_val / i)))
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
        get_classification_report(out_gt, out_pred, DISEASE_TYPES)
    else:
        print("Validation loss: {0:.4f}".format(loss_mean))
    return loss_mean


def save_checkpoint(epoch, model, optimizer, loss, is_best, logger):
    """Save checkpoint if a new best is achieved"""
    if not is_best:
        print("=> Validation Accuracy did not improve")

    else:
        print("=> Saving a new best")

        PATH = MODEL_CHKPTS_DIR + 'checkpoint{0}_epoch{1}_loss{2:.4f}.pth.tar'.format(TIMESTAMP, epoch, loss)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, PATH)

    logger.info('epoch:{0},val:{1:.4f}'.format(epoch, loss))
