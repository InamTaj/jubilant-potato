#!/usr/local/bin/python3

import os, glob
import tqdm

################## CONSTANTS
CHEXNET_DIR = r'./datasets/chexnet/'
IMAGES_DIR = CHEXNET_DIR + 'images'
LABELS_DIR = CHEXNET_DIR + 'full_labels'
AVAILABLE_LABELS_DIR = CHEXNET_DIR + 'available_labels'
TRAIN_LIST = LABELS_DIR + '/train_list.txt'
VAL_LIST = LABELS_DIR + '/val_list.txt'
TEST_LIST = LABELS_DIR + '/test_list.txt'
AVAILABLE_TRAIN_LIST = AVAILABLE_LABELS_DIR + '/train_list.txt'
AVAILABLE_VAL_LIST = AVAILABLE_LABELS_DIR + '/val_list.txt'
AVAILABLE_TEST_LIST = AVAILABLE_LABELS_DIR + '/test_list.txt'

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
        img_name = each.split(' ')[0]

        try:
            if FILES_LOOKUP[img_name] != None:
                FILTERED_LIST.append(each)
        except KeyError:
            pass

    # finally write to file
    if len(FILTERED_LIST) > 0:
        file = open(OUTPUT_FILE, 'w+')

        for line in FILTERED_LIST:
            to_write = line + '\n'
            file.write(to_write)


def main():
    process_labels(TRAIN_LIST, AVAILABLE_TRAIN_LIST)
    process_labels(VAL_LIST, AVAILABLE_VAL_LIST)
    process_labels(TEST_LIST, AVAILABLE_TEST_LIST)
    print('Done')


if __name__ == '__main__':
    main()

