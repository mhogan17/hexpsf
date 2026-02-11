import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.feature import blob_dog
import tifffile

def correct_image(raw_image, gain_image, offset_image, variance_image=None, return_variance=False):
    corrected = (raw_image - offset_image) / gain_image

    if return_variance:
        if variance_image is not None:
            corrected_var = variance_image / (gain_image ** 2)
        else:
            corrected_var = np.ones_like(corrected)  # dummy if not provided
        return corrected, corrected_var

    return corrected

def highpass(frame):
    A = np.fft.fft2(frame)
    A1 = np.fft.fftshift(A)
    M, N = np.shape(A)
    R = 4
    X = np.linspace(0, N-1, N)
    Y = np.linspace(0, M-1, M)
    X, Y = np.meshgrid(X, Y)
    Cx = N/2
    Cy = M/2

    Hi = 1 - np.exp(-((X - Cx) ** 2 + (Y - Cy) ** 2) / (2 * R) ** 2)
    K = A1 * Hi
    K1 = np.fft.ifftshift(K)
    B = np.fft.ifft2(K1)
    out = B - np.min(np.min(B))

    return np.abs(out)


def uniform_small(frame):
    f1 = frame[0:-1, 0:-1]
    f2 = frame[0:-1, 1::]
    f3 = frame[1::, 0:-1]
    f4 = frame[1::, 1::]
    return f1 + f2 + f3 + f4


def uniform_large(frame):
    f1 = frame[0:-4, 0:-4]
    f2 = frame[0:-4, 2:-2]
    f3 = frame[0:-4, 4::]
    f4 = frame[2:-2, 0:-4]
    f5 = frame[2:-2, 2:-2]
    f6 = frame[2:-2, 4::]
    f7 = frame[4::, 0:-4]
    f8 = frame[4::, 2:-2]
    f9 = frame[4::, 4::]

    return f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9

class HexPSFFinder:
    def __init__(self, dirname, n=100):
        self.dirname = dirname
        self.n = n
        self.gain = np.array(tifffile.imread('PIXSTAT/gain.tiff'))
        self.offset = np.array(tifffile.imread('PIXSTAT/offset.tiff'))
        self.variance = np.array(tifffile.imread('PIXSTAT/variance.tiff'))

        self.images = []
        self.fnames = []

        count = 0
        for fname in sorted(os.listdir(dirname)):
            if count >= n:
                break
            if fname == 'ROIs.csv' or fname == 'particles.csv':
                continue
            image = np.array(tifffile.imread(dirname + fname))
            image = correct_image(image, self.gain, self.offset)[30:-30, 30:-30]
            self.images.append(image)
            self.fnames.append(fname)
            count += 1

        x = []
        y = []
        idx = []
        a = 15
        for i in range(n):
            print(f"\rFinding Blobs... Progress: {round(i / n * 100, 3)}%", flush=True, end='')

            image = self.images[i]
            image = highpass(image)[50:-50, 30:-30]
            image = uniform_small(image)
            image = uniform_large(image)
            blobs_dog = blob_dog(image[a:-a, a:-a], min_sigma=7, max_sigma=14, threshold_rel=0.05, overlap=0.1)
            blobs_dog[:, 2] = blobs_dog[:, 2] * np.sqrt(2)
            y.append(blobs_dog[:, 0])
            x.append(blobs_dog[:, 1])
            idx.append(np.zeros(blobs_dog.shape[0]) + i)

            # Plotting
            # x0 = blobs_dog[:, 1]
            # y0 = blobs_dog[:, 0]
            # r0 = blobs_dog[:, 2]
            # fig, axes = plt.subplots(2, 1)
            # fig.set_size_inches(10, 20)
            # im = axes[0].imshow(image, cmap='gray')
            # plt.colorbar(im, ax=axes[0])
            # for xc, yc, rc in zip(x0, y0, r0):
            #     circ = plt.Circle((xc+a, yc+a), rc, fill=False, color='red')
            #     axes[0].add_patch(circ)
            #
            # axes[1].hist(image.flatten())
            # plt.show()

        x = np.concatenate(x)
        y = np.concatenate(y)
        idx = np.concatenate(idx)
        ROIlist = np.hstack((idx, x, y))
        ROIlist = np.array(ROIlist.reshape((3, x.shape[0])), dtype='float')
        ROIlist = ROIlist.T

        np.savetxt(dirname + 'ROIs.csv', ROIlist, delimiter=',')


finder = HexPSFFinder(dirname='hexpsf_all_cal/', n=63)
with open('hexpsf_all/ROIs.csv', 'r') as file:
    ROIs = file.read()
    file.close()

ROIs = ROIs.split('\n')
print(len(ROIs))