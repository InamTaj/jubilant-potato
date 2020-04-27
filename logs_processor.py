# venv/bin/python

import pandas as pd
import matplotlib.pyplot as plt

def main():
    filename = 'losses_27.04.2020_04.45.07'
    theFrame = pd.read_csv(filename + '.log', names=['epochs', 'losses'])

    EPOCHS = set()
    TRAIN_LOSS = []
    VAL_LOSS   = []

    for index, row in theFrame.iterrows():
        epoch = row['epochs'].split(':')[1]
        losses = row['losses'].split(':')
        loss_type = losses[0]
        loss_val = losses[1]

        EPOCHS.add(epoch)

        if loss_type == 'train':
            TRAIN_LOSS.append(float(loss_val))
        if loss_type == 'val':
            VAL_LOSS.append(float(loss_val))

    EPOCHS = list(EPOCHS)
    EPOCHS.sort()

    # sanity check
    if(len(EPOCHS) != len(TRAIN_LOSS) or len(EPOCHS) !=  len(VAL_LOSS)):
        print('Inconsistent # of entries for losses')
        exit(1)

    DICT = {
        'epochs': EPOCHS,
        'training_loss': TRAIN_LOSS,
        'validation_loss': VAL_LOSS,
    }

    dictFrame = pd.DataFrame(DICT)
    dictFrame.to_csv(filename + '.csv')

    ax = plt.gca()
    dictFrame.plot(kind='line', x='epochs', y='training_loss', ax=ax)
    dictFrame.plot(kind='line', x='epochs', y='validation_loss', ax=ax)
    plt.show()


if __name__ == '__main__':
    main()
