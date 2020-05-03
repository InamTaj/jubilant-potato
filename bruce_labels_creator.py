#!/usr/local/bin/python3

import os
import pandas as pd

################## CONSTANTS
CHEXNET_DIR = r'./datasets/chexnet/'
IMAGES_DIR = CHEXNET_DIR + 'images'
LABELS_DIR = CHEXNET_DIR + 'labels'
AVAILABLE_LABELS_DIR = CHEXNET_DIR + 'bruce_labels'
TRAIN_LIST = LABELS_DIR + '/train_list.txt'
VAL_LIST = LABELS_DIR + '/val_list.txt'
TEST_LIST = LABELS_DIR + '/test_list.txt'
AVAILABLE_TRAIN_LIST = AVAILABLE_LABELS_DIR + '/train.csv'
AVAILABLE_VAL_LIST = AVAILABLE_LABELS_DIR + '/val.csv'
AVAILABLE_TEST_LIST = AVAILABLE_LABELS_DIR + '/test.csv'

if not os.path.exists(AVAILABLE_LABELS_DIR):
    os.makedirs(AVAILABLE_LABELS_DIR)
###########################

files = os.listdir(IMAGES_DIR)

FILES_LOOKUP = {}

for f_name in files:
    FILES_LOOKUP[f_name] = f_name


###########################

def process_labels(INPUT_LIST, OUTPUT_FILE):
    FILTERED_LIST = []

    with open(INPUT_LIST) as file:
        LABELS = [line.rstrip() for line in file]

    # filter labels
    for each in LABELS:
        splitted = each.split(' ')
        img_name = splitted[0]

        try:
            if FILES_LOOKUP[img_name] != None:
                FILTERED_LIST.append(splitted)
        except KeyError:
            pass

    # finally write to file
    col_names = ['Image Index', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
    dF = pd.DataFrame(FILTERED_LIST, columns=col_names)
    print(dF.head())
    dF.to_csv(OUTPUT_FILE, sep=',', header=col_names, index=False)


def main():
    process_labels(TRAIN_LIST, AVAILABLE_TRAIN_LIST)
    process_labels(VAL_LIST, AVAILABLE_VAL_LIST)
    process_labels(TEST_LIST, AVAILABLE_TEST_LIST)
    print('Done')


if __name__ == '__main__':
    main()

