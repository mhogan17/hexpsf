import numpy as np
import matplotlib.pyplot as plt
import tifffile
from PIL import Image
from scipy.optimize import minimize
from scipy import special
import os
import pandas as pd
from HexPSF import HexPSF
from BlobDetectionSpherical import correct_image, highpass, uniform_small, uniform_large
import time


def mle_poisson_fit_psf(measured_psf, model_func,
                        initial_guess=(2, 2, 0, 0.550, 100000, 1000),
                        bounds=None):

    measured_psf = measured_psf.astype(np.float64)
    measured_psf = np.clip(measured_psf, a_min=1e-12, a_max=None)

    def nll_poisson(params):
        x0, y0, zp, wl, N, B = params
        model_psf = model_func(x0=x0, y0=y0, zp=zp, wl=wl, N=N, B=B)
        # Poisson negative log likelihood
        return np.sum(model_psf - measured_psf * np.log(model_psf))

    result = minimize(nll_poisson, x0=initial_guess, bounds=bounds, method='Nelder-Mead')
    return result.x

class HexPSFFitter:
    def __init__(self):
        self.psf = HexPSF()
        h1 = 2.24
        h2 = 5.09
        self.psf.set_plate([0, h1, h2, 0, h1, h2])
        self.params = [0, 0, 0, 0, 0, 0]
        self.x = np.linspace(0,0,0)
        self.y = np.linspace(0,0,0)
        self.dx = 0
        self.dy = 0
        self.data = []
        return

    def HexPSF_model(self, x0, y0, zp, wl, N, B):
        corrections = [0,0,-0.25]
        z_modes = [0]
        return self.psf.intensity(x0, y0, zp, wl, N, B, corrections, z_modes)


    def fit(self, dirname):
        tifs = []
        gain = tifffile.imread('PIXSTAT/gain.tiff')
        offset = tifffile.imread('PIXSTAT/offset.tiff')
        for fname in sorted(os.listdir(dirname)):
            if fname == 'ROIs.csv' or fname == 'particles.csv' or fname == 'readme.txt':
                continue
            tif = np.array(correct_image(tifffile.imread(dirname + fname), gain_image=gain, offset_image=offset))
            tifs.append(tif)

        with open(dirname + 'ROIs.csv') as f:
            ROIs = f.read()
            f.close()
        n=len(ROIs)
        ROIs = ROIs.split('\n')[1:-1]
        self.data = [['image', 'x0 (microns)', 'y0 (microns)', 'zp (microns)', 'wl (microns)','N (count)', 'B (count)', 'R^2']]

        times=[]
        for row in ROIs:
            start = time.time()

            row = row.split(',')
            idx = int(float(row[0]))
            x = int(float(row[1]))
            y = int(float(row[2]))
            width = 30
            height = 30

            self.x = np.linspace(0, width - 1, width) * self.psf.optics.PixelSize / self.psf.optics.M
            self.y = np.linspace(0, height - 1, height)[:, None] * self.psf.optics.PixelSize / self.psf.optics.M

            self.K = width * height

            image = tifs[idx]

            fig = plt.figure()
            fig.set_size_inches(12, 8)
            ax = fig.add_subplot(1, 3, 1)
            ax.imshow(image)
            circ = plt.Circle((x+30, y+30), 15, fill=False, color='red')
            ax.add_patch(circ)

            width = 30
            height = 30
            image = image[y+15:y + 45, x+15:x + 45]

            B = (np.mean(image[0:, 0]) + np.mean(image[0:, width - 1]) + np.mean(image[0, 1:width - 2]) + np.mean(image[height - 1, 1:width - 2])) / 4

            image = image - np.ones_like(image) * B
            img_min = np.min(image)
            image = image - np.ones_like(image) * (img_min - 1)

            N = np.sum(image)

            params = []
            R2s = []
            i=0
            for wl in [0.513, 0.579, 0.683]:
                params.append(mle_poisson_fit_psf(image, self.HexPSF_model, initial_guess=(0, 0, 0, wl, N, B),
                                                  bounds=((-1,1), (-1,1), (-0.5,0.5), (0.5, 0.7), (0, None), (0,None))))
                x0 = params[i][0]
                y0 = params[i][1]
                zp = params[i][2]
                wl = params[i][3]
                N = params[i][4]
                B = params[i][5]

                model = self.HexPSF_model(x0,y0,zp,wl,N, B)

                mean = np.mean(image)
                s_res = (image - model) ** 2
                s_tot = (image - mean) ** 2
                R2s.append(1 - np.sum(s_res) / np.sum(s_tot))
                i+=1

            if R2s[0] > R2s[1] and R2s[0] > R2s[2]:
                self.params = params[0]
                R2 = R2s[0]
            elif R2s[1] > R2s[2]:
                self.params = params[1]
                R2 = R2s[1]
            else:
                self.params = params[2]
                R2 = R2s[2]

            x0 = self.params[0]
            y0 = self.params[1]
            zp = self.params[2]
            wl = self.params[3]
            N = self.params[4]
            B = self.params[5]

            model = self.HexPSF_model(x0, y0, zp, wl, N, B)

            ax = fig.add_subplot(1, 3, 2)
            ax.imshow(image)
            ax = fig.add_subplot(1,3,3)
            ax.imshow(model)
            plt.show()

            data_row = [int(idx), float((x + (self.params[0] * self.psf.optics.M / self.psf.optics.PixelSize) - width / 2)),
                        float((y + (self.params[1] * self.psf.optics.M / self.psf.optics.PixelSize) - height / 2)),
                        float(self.params[2]),
                        float(self.params[3]), float(self.params[4]), float(self.params[5]), float(R2)]
            print(data_row)
            self.data.append(data_row)
            end = time.time()
            times.append(end - start)

            i += 1
            print(f"\rFitting HexPSFs... Progress: {round(i / n * 100, 3)}%, "
                  f"Estimated time remaining... {round((np.mean(times)) * (n-i), 3)} s", flush=True, end='')

        data = pd.DataFrame(self.data)
        data.to_csv(dirname + 'particles.csv')


            # iters = 0
            # while iters < 5:
            #     iters += 1
            #     self.corrections = mle_poisson_fit_corrections(self.params, image, psf.intensity,
            #                                                initial_guess=self.corrections,
            #                                                    bounds=((-1,1), (-1,1), (0, 2 * np.pi), (-1,1),(-1,1),(-1,1)))
            # x_off = self.corrections[0]
            # y_off = self.corrections[1]
            # theta_off = self.corrections[2]
            # z1 = self.corrections[3]
            # z2 = self.corrections[4]
            # z3 = self.corrections[5]
            # print(self.corrections)
            #
            # model = psf.intensity(x0=x0, y0=y0, zp=zp, wl=wl, N=N, B=B, corrections=self.corrections)
            #
            # mean = np.mean(image)
            # s_res = np.sum((image - model) ** 2)
            # s_tot = np.sum((image - mean) ** 2)
            # R2 = 1 - s_res / s_tot
            #
            # data_row = np.array([idx, float(x0), float(y0), float(zp), float(wl), float(N), float(B), float(R2)])
            # print(data_row)
            #
            # fig, axes = plt.subplots(2, 1)
            # axes[0].imshow(image)
            # axes[1].imshow(model)
            # plt.show(block=False)
            # plt.pause(5)
            # plt.close()

        # return data_row

fitter = HexPSFFitter()
fitter.fit('hexpsf_all_cal/')
# df = pd.read_csv('hexpsf_all_cal/particles.csv')
# zp_col = df['3'][1:]
# wl_col = df['4'][1:]
# R2_col = df['7'][1:]
# wl = []
# zp = []
# R2 = []
# for i in range(len(wl_col)):
#     i=i+1
#     wl.append(float(wl_col[i]))
#     zp.append(float(zp_col[i]))
#     R2.append(float(R2_col[i]))
#
#
# fig, ax = plt.subplots(1,3)
#
# ax[0].hist(wl, bins=40)
# ax[1].hist(zp, bins=40)
# ax[2].hist(R2, bins=40)
#
# plt.show()