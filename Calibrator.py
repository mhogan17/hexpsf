from BlobDetectionSpherical import find_blobs, correct_image
import os
import tifffile
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from HexPSF import HexPSF
import pandas as pd
import time
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


def mle_poisson_fit_corrections(model_params, measured_psf, model_func,
                                initial_guess=(0, 0, 0, 0, 0, 0),
                                bounds=None):
    measured_psf = measured_psf.astype(np.float64)
    # measured_psf = np.clip(measured_psf, a_min=1e-12, a_max=None)
    x0, y0, zp, wl, N, B = model_params

    def nll_poisson(params):
        x_off, y_off, theta_off, z0, z1, z2 = params
        model_psf = model_func(x0=x0, y0=y0, zp=zp, wl=wl, N=N, B=B,
                               corrections=[x_off, y_off, theta_off, z0, z1, z2])
        # model_psf = np.clip(model_psf, a_min=1e-12, a_max=None)
        return np.sum(model_psf - measured_psf * np.log(model_psf))

    result = minimize(nll_poisson, x0=initial_guess, bounds=bounds, method="Nelder-Mead")
    return result.x



def mle_poisson_fit_rhombus(measured_psf, model_func, initial_guess=(15, 15, 10, np.pi/6, 10000, 100), bounds=None):
    measured_psf = measured_psf.astype(np.float64)
    # measured_psf = np.clip(measured_psf, a_min=1e-12, a_max=None)

    def nll_poisson(params):
        x0, y0, a, alpha, N, B = params
        model = model_func(x0, y0, a, alpha, N, B)
        return np.sum(model - measured_psf * np.log(model))

    result = minimize(nll_poisson, x0=initial_guess, bounds=bounds, method='Nelder-Mead')
    return result.x


def mle_poisson_fit_params(corrections, measured_psf, model_func,
                           initial_guess=(0, 0, 0, 0.55, 1000000, 1000), bounds=None):
    measured_psf = measured_psf.astype(np.float64)
    # measured_psf = np.clip(measured_psf, a_min=1e-12, a_max=None)

    def nll_poisson(params):
        x0, y0, zp, wl, N, B = params
        model_psf = model_func(x0=x0, y0=y0, zp=zp, wl=wl, N=N, B=B, corrections=corrections)
        # model_psf = np.clip(model_psf, a_min=1e-12, a_max=None)
        return np.sum(model_psf - measured_psf * np.log(model_psf))

    result = minimize(nll_poisson, x0=initial_guess, bounds=bounds, method="Nelder-Mead")
    return result.x


class Calibrator:
    def __init__(self, dirname):
        self.dirname = dirname
        self.images = []
        self.img_names = []
        self.gain = np.array(tifffile.imread('PIXSTAT/gain.tiff'))
        self.offset = np.array(tifffile.imread('PIXSTAT/offset.tiff'))
        self.variance = np.array(tifffile.imread('PIXSTAT/variance.tiff'))
        self.corrections = []
        self.params = []
        self.ROIs = []
        self.psf = HexPSF()
        h1 = 2.24
        h2 = 5.09
        self.psf.set_plate([0, h1, h2, 0, h1, h2])

        bead_ROIs = False
        for image in sorted(os.listdir(self.dirname)):
            if image == 'ROIs.csv':
                bead_ROIs = True
                continue
            if image == 'particles.csv':
                bead_particles = True
                continue

            self.images.append(correct_image(tifffile.imread(os.path.join(self.dirname, image)),
                                             self.gain, self.offset))
            self.img_names.append(image)

        if not bead_ROIs:
            find_blobs(self.dirname, int(len(os.listdir(dirname))))

        with open(self.dirname + 'ROIs.csv') as f:
            self.ROIs = f.read()
            f.close()
        self.ROIs = self.ROIs.split('\n')

        # self.data = np.zeros((len(self.ROIs), 8))
        # self.data[0] = np.array(['image', 'x0', 'y0', 'zp (microns)', 'wl (microns)', 'N (count)', 'B (count)', 'R2'])

    def hex_rhombus(self, x_0, y_0, a, alpha, N, B):
        x = np.linspace(0, 29, 30)
        y = np.linspace(0, 29, 30)
        Z = np.zeros((30, 30))

        x1 = x_0 + np.sqrt(3) / 2 * a * np.cos(alpha)
        x2 = x_0 - a / 2 * np.sin(alpha)
        x3 = x_0 - np.sqrt(3) / 2 * a * np.cos(alpha)
        x4 = x_0 + a / 2 * np.sin(alpha)
        xs = [x1, x2, x3, x4]
        # print(xs)
        y1 = y_0 + np.sqrt(3) / 2 * a * np.sin(alpha)
        y2 = y_0 + a / 2 * np.cos(alpha)
        y3 = y_0 - np.sqrt(3) / 2 * a * np.sin(alpha)
        y4 = y_0 - a / 2 * np.cos(alpha)
        ys = [y1, y2, y3, y4]
        # print(ys)
        coords = []
        for i in range(4):
            coords.append((xs[i], ys[i]))

        polygon = Polygon(coords)
        for i in range(len(x)):
            for j in range(len(y)):
                point = Point(x[i], y[j])
                Z[i, j] += N / (np.sqrt(3) * a ** 2) if (polygon.contains(point)) else 0
                Z[i, j] += B

        return Z
        # plt.imshow(Z)
        # plt.show()

    def fit_rhombus(self, i):
        row = self.ROIs[i].split(',')
        idx = int(float(row[0]))
        x = int(float(row[1]))
        y = int(float(row[2]))
        image = self.images[idx][y + 15:y + 45, x + 15:x + 45]
        width = 30
        height = 30

        B = (np.mean(image[0:, 0]) + np.mean(image[0:, width - 1]) + np.mean(image[0, 1:width - 2]) + + np.mean(
            image[height - 1, 1:width - 2])) / 4

        image = image - np.ones_like(image) * B
        img_min = np.min(image)
        image = image - np.ones_like(image) * (img_min - 1)


        B = B / 30 ** 2
        N = np.sum(image)

        result = mle_poisson_fit_rhombus(image, self.hex_rhombus)
        print(result)

        model = self.hex_rhombus(result[0], result[1], result[2], result[3], result[4], result[5])

        fig, axes = plt.subplots(2, 1)
        axes[0].imshow(image, cmap='gray')
        axes[1].imshow(model, cmap='gray')
        plt.show(block=False)
        plt.pause(1)
        plt.close()




    def calibrate(self):
        n = len(self.ROIs)
        for i in range(n):
            print("Calibrating... " + str(100*round(i/n, 2)) + '%', flush=True, end='')
            self.fit_rhombus(i)

        # self.data = pd.DataFrame(self.data)
        # self.data.to_csv(self.dirname + 'particles.csv')


# corrections = [ 0.00271894,  0.0043681,   0.00011607,  0.00118429, -0.00354776, -0.00162042]
# plt.imshow(psf.intensity(x0=0.05,y0=-0.01,zp=-0.5,wl=0.683,N=50000,B=0.01, corrections=corrections))
# plt.show()
cal = Calibrator('hexpsf_cal/')
cal.calibrate()