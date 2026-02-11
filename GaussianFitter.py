import numpy as np
import matplotlib.pyplot as plt
import tifffile
from PIL import Image
from scipy.optimize import minimize
from scipy import special
import os
import pandas as pd
from BlobDetectionSpherical import correct_image, highpass, uniform_small, uniform_large
import matplotlib.patches as patches


def mle_poisson_fit_psf(measured_psf, model_func,
                        initial_guess=(2, 2, 1, 100000, 1000),
                        bounds=None):

    measured_psf = measured_psf.astype(np.float64)
    measured_psf = np.clip(measured_psf, a_min=1e-12, a_max=None)

    def nll_poisson(params):
        x0, y0, s, N, B = params
        model_psf = model_func(x0=x0, y0=y0, s=s, N=N, B=B)
        # Poisson negative log likelihood
        return np.sum(model_psf - measured_psf * np.log(model_psf))

    result = minimize(nll_poisson, x0=initial_guess, bounds=bounds, method='Nelder-Mead')
    return result.x


class GaussianFitter:
    def __init__(self, m, pixel_size):
        self.M = m
        self.pixel_size = pixel_size
        self.K = 0
        self.params = [0, 0, 0, 0, 0]
        self.x = np.linspace(0,0,0)
        self.y = np.linspace(0,0,0)
        self.dx = 0
        self.dy = 0
        return

    def gauss_model(self, x0, y0, s, N, B):
        return 1 / (self.M ** 2) * \
               (N / (2 * np.pi * s ** 2) * np.exp((-(self.x - x0) ** 2 - (self.y - y0) ** 2) / (2 * s ** 2)) + B)

    def I_x0_num(self, a, b, c, d):
        return ((np.exp((2 * a * self.params[0] - self.params[0] ** 2 - a ** 2) / (2 * self.params[2] ** 2))
                      - np.exp((2 * b * self.params[0] - self.params[0] ** 2 - b ** 2) / (2 * self.params[2] ** 2)))
                     * (special.erf(np.sqrt(2)*(self.params[1] - d) / (2 * self.params[2])) -
                special.erf(np.sqrt(2)*(self.params[1] - c) / (2 * self.params[2])))) ** 2

    def I_x0_den(self, a, b, c, d):
        return (2 * np.pi * self.M ** 2 * self.params[2] ** 2 * \
               (special.erf(np.sqrt(2)*(self.params[1] - d) / (2 * self.params[2])) -
                special.erf(np.sqrt(2)*(self.params[1] - c) / (2 * self.params[2]))) * \
               (special.erf(np.sqrt(2)*(self.params[0] - b) / (2 * self.params[2])) -
                special.erf(np.sqrt(2)*(self.params[0] - a) / (2 * self.params[2]))) + \
               self.params[4] / self.params[3] * (1 / self.K))

    def fit(self, dirname):
        tifs = []
        gain = tifffile.imread('PIXSTAT/gain.tiff')
        offset = tifffile.imread('PIXSTAT/offset.tiff')
        for fname in sorted(os.listdir(dirname)):
            if fname == 'ROIs.csv' or fname == 'particles.csv' or fname=='readme.txt':
                continue
            tif = np.array(correct_image(tifffile.imread(dirname + fname), gain_image=gain, offset_image=offset))[30:-30, 30:-30]
            tifs.append(tif)

        with open(dirname + 'ROIs.csv') as f:
            ROIs = f.read()
            f.close()
        #
        ROIs = ROIs.split('\n')[1:-1]
        data = [['image', 'x0 (microns)', 'y0 (microns)', 'σ (microns)', 'N (count)', 'B (count)', 'R^2',
                 'CRLB_x0 (microns)']]

        for row in ROIs:
            row = row.split(',')
            idx = int(float(row[0]))
            x = int(float(row[1]))
            y = int(float(row[2]))
            width = 30
            height = 30

            self.x = np.linspace(0, width - 1, width) * self.pixel_size / self.M
            self.y = np.linspace(0, height - 1, height)[:, None] * self.pixel_size / self.M
            self.dx = self.x[1] - self.x[0]
            self.dy = self.y[1] - self.y[0]
            self.K = width * height
            image = uniform_large(uniform_small(highpass(tifs[idx])[50:-50, 30:-30]))

            # fig = plt.figure()
            # fig.set_size_inches(20, 10)
            # ax = fig.add_subplot(1, 3, 1)
            # ax.imshow(image)
            # circ = plt.Circle((x+15, y+15), 15, fill=False, color='red')
            # ax.add_patch(circ)

            image = image[y:y+30, x:x+30]


            # B = (np.mean(image[0:, 0]) + np.mean(image[0:, width-1]) + np.mean(image[0, 1:width-2]) + + np.mean(
            #     image[height-1, 1:width-2])) / 4

            # image = image - np.ones_like(image) * B

            result = mle_poisson_fit_psf(image, self.gauss_model, initial_guess=(2,2,1,10000000,10000),
                                              bounds=((0,None), (0,None), (0,None), (0,None), (0,None)))

            # print(result)
            self.params = result

            model = self.gauss_model(self.params[0], self.params[1], self.params[2], self.params[3], self.params[4])


            # Determine R^2, just for fun
            mean = np.mean(np.clip(image, a_min=0, a_max=None))
            s_res = (image - model) ** 2
            s_tot = (image - mean) ** 2
            R2 = 1 - np.sum(s_res) / np.sum(s_tot)

            # ax = fig.add_subplot(1, 3, 2)
            # ax.imshow(image)
            # ax = fig.add_subplot(1,3,3)
            # ax.imshow(model)
            # plt.show(block=False)
            # plt.pause(2)
            # plt.close(fig)

            I_x0 = 0
            for i in self.x:
                for j in self.y:
                    I_x0 += self.I_x0_num(i, i+self.dx, j, j+self.dy) / self.I_x0_den(i, i+self.dx, j, j+self.dy)

            I_x0 *= self.params[3]
            CRLB_x0 = np.sqrt(1/I_x0)

            data_row = [int(idx), float((x + (self.params[0] * self.M / self.pixel_size) - width/2)),
                        float((y + (self.params[1] * self.M / self.pixel_size) - height/2)),
                        float(self.M / self.pixel_size * self.params[2]),
                        float(self.params[3]), float(self.params[4]), float(R2), float(self.M / self.pixel_size * CRLB_x0[0])]
            print(data_row)
            data.append(data_row)

        data = pd.DataFrame(data)
        data.to_csv(dirname + 'particles.csv')

# fitter = GaussianFitter(m=67.7, pixel_size=6.5)
# fitter.fit('11_07_25/vutara/small_beads/')

# fitter = GaussianFitter(m=175.4, pixel_size=16)
# fitter.fit('new beads/')

# fitter = GaussianFitter(m=175.4, pixel_size=16)
# fitter.fit('new beads/')

fitter = GaussianFitter(m=67.7, pixel_size=6.5)
fitter.fit('hexpsf_all/')