import glob
import os
import subprocess
import sys

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

# import pathlib
# import subprocess
# import tarfile
# import pickle
# import multiprocessing
# import pandas as pd
# from PIL import Image
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import train_test_split

# if sys.version_info.major == 2:
#     # Backward compatibility with python 2.
#     from six.moves import urllib
#     urlretrieve = urllib.request.urlretrieve
# else:
#     from urllib.request import urlretrieve


def get_gpu_name():
    try:
        out_str = subprocess.run(["nvidia-smi", "--query-gpu=gpu_name", "--format=csv"], stdout=subprocess.PIPE).stdout
        out_list = out_str.decode("utf-8").split('\n')
        out_list = out_list[1:-1]
        return out_list
    except Exception as e:
        print(e)

def get_cuda_version():
    """Get CUDA version"""
    if sys.platform == 'win32':
        raise NotImplementedError("Implement this!")
        # This breaks on linux:
        #cuda=!ls "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        #path = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\" + str(cuda[0]) +"\\version.txt"
    elif sys.platform == 'linux' or sys.platform == 'darwin':
        path = '/usr/local/cuda/version.txt'
    else:
        raise ValueError("Not in Windows, Linux or Mac")
    if os.path.isfile(path):
        with open(path, 'r') as f:
            data = f.read().replace('\n','')
        return data
    else:
        return "No CUDA in this machine"

def get_cudnn_version():
    """Get CUDNN version"""
    if sys.platform == 'win32':
        raise NotImplementedError("Implement this!")
        # This breaks on linux:
        #cuda=!ls "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        #candidates = ["C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\" + str(cuda[0]) +"\\include\\cudnn.h"]
    elif sys.platform == 'linux':
        candidates = ['/usr/include/x86_64-linux-gnu/cudnn_v[0-99].h',
                      '/usr/local/cuda/include/cudnn.h',
                      '/usr/include/cudnn.h']
    elif sys.platform == 'darwin':
        candidates = ['/usr/local/cuda/include/cudnn.h',
                      '/usr/include/cudnn.h']
    else:
        raise ValueError("Not in Windows, Linux or Mac")
    for c in candidates:
        file = glob.glob(c)
        if file: break
    if file:
        with open(file[0], 'r') as f:
            version = ''
            for line in f:
                if "#define CUDNN_MAJOR" in line:
                    version = line.split()[-1]
                if "#define CUDNN_MINOR" in line:
                    version += '.' + line.split()[-1]
                if "#define CUDNN_PATCHLEVEL" in line:
                    version += '.' + line.split()[-1]
        if version:
            return version
        else:
            return "Cannot find CUDNN version"
    else:
        return "No CUDNN in this machine"

def compute_roc_auc(data_gt, data_pd, classes, full=True):
    AUROCs = []
    data_gt_np = np.asarray(data_gt)
    data_pd_np = np.rint(np.asarray(data_pd))

    for i in range(classes):
      if len(np.unique(data_gt_np[:, i])) == 1: # bug in roc_auc_score
          score = accuracy_score(data_gt_np[:, i], data_pd_np[:, i])
          AUROCs.append( score )
      else:
        AUROCs.append(roc_auc_score(data_gt_np[:, i], data_pd_np[:, i]))

    print("Full AUC", AUROCs)
    AUROCs = np.mean(AUROCs)
    return AUROCs
