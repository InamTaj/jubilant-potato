import torch
from torch.utils.data import DataLoader

from commons import no_augmentation_dataset, get_symbol, init_symbol, valid_epoch
from constants import BATCHSIZE, MULTI_GPU, NUM_WORKERS, CLASSES, DATA_DIR, TEST_IMAGES_LIST, transform_normalizer, \
    MODEL_CHKPTS_DIR, PRETRAINED_MODEL_PATH

def main_test():
    ###################################################### TESTING & EVALUATION
    test_dataset = no_augmentation_dataset(DATA_DIR, TEST_IMAGES_LIST, transform_normalizer)

    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCHSIZE,
                             shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    chexnet_sym = get_symbol(CLASSES, MULTI_GPU)

    if PRETRAINED_MODEL_PATH != None:
        # Load weights of saved model
        chkpt = torch.load(PRETRAINED_MODEL_PATH)
        chexnet_sym.load_state_dict(chkpt['model_state_dict'])

        # Load criterion
        # Scheduler for LRPlateau is not used
        _, criterion, _ = init_symbol(chexnet_sym)


        test_loss = valid_epoch(chexnet_sym, test_loader, criterion, 'testing')
        print('=> Test Loss at end:', test_loss)
        print('Done')
    else:
        print('=> ERROR: No PRE-TRAINED model  at {0}'.format(PRETRAINED_MODEL_PATH))


if __name__ == '__main__':
    main_test()
