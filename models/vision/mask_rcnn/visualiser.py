import numpy as np
import matplotlib.pyplot as plt

def mask_imshow(image, mask):
    '''
    pytorch の image変換
    '''
    npimg = image.numpy()
    npmask = mask.numpy()
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(2 * 10,10))
    ax[0].imshow(np.transpose(npimg, (1, 2, 0)), cmap=plt.cm.bone)
    ax[1].imshow(np.transpose(npimg, (1, 2, 0)), cmap=plt.cm.bone)
    ax[1].imshow(np.squeeze(npmask).T, alpha=0.3, cmap="Reds")
    # as opencv loads in BGR format by default, we want to show it in RGB.
    plt.show()