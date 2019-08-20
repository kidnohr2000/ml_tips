import numpy as np
import matplotlib.pyplot as plt
import cv2

def mask_imshow(image, target):
    npimg = image.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    npbbox = np.squeeze(target['boxes'].numpy())
    npmask = target['masks'].numpy()
    npmask = np.squeeze(npmask)
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(2 * 10,10))
    ax[0].imshow(npimg, cmap=plt.cm.bone)
    print((npbbox[0], npbbox[1]), (npbbox[2], npbbox[3]))
    cv2.rectangle(npmask, (npbbox[0], npbbox[1]), (npbbox[2], npbbox[3]), (255,0,0), 5)
    ax[1].imshow(npimg, cmap=plt.cm.bone)
    ax[1].imshow(npmask, alpha=0.3, cmap="Reds")
    # as opencv loads in BGR format by default, we want to show it in RGB.
    plt.show()