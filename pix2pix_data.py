import numpy as np
import cv2
import matplotlib.pyplot as plt

if __name__ == "__main__":
    npzfile = np.load('SOCOFing/data.npz')
    x, y = npzfile['x'], npzfile['y']
    x = np.squeeze(x) * 255
    y = np.squeeze(y) * 255

    z = np.zeros(shape=(x.shape[0], x.shape[1], 2 * x.shape[2]), dtype='uint8')
    z[:, :, 0:x.shape[1]] = x
    z[:, :, x.shape[1]:2 * x.shape[1]] = y



    for i in range(z.shape[0]):
        rgb_img = np.concatenate([np.expand_dims(z[i], axis=-1), np.expand_dims(z[i], axis=-1), np.expand_dims(z[i], axis=-1)], axis=-1)
        cv2.imwrite('pix2pixdataset/{}.png'.format(i), rgb_img)
