from BlobDetectionSpherical import find_blobs, correct_image
import os
import tifffile
import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import center_of_mass, gaussian_filter
import matplotlib.pyplot as plt
from HexPSF import HexPSF
from skimage.feature import blob_dog
import pandas as pd
import time
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


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


def mle_poisson_fit_corrections(known_params, measured_psf, model_func, initial_guess=(0, 0, -0.2), bounds=((-0.5, 0.5), (-0.5, 0.5), (-np.pi, np.pi))):
    x, y, z, wl, N, B = known_params
    measured_psf = measured_psf.astype(np.float64)
    measured_psf = np.clip(measured_psf, a_min=1e-12, a_max=None)

    def nll_poisson(params):
        x_off, y_off, theta_off = params
        model_psf = model_func(x0=x, y0=y, zp=z, wl=wl, N=N, B=B, corrections=[x_off, y_off, theta_off], z_modes=[0,0,0,0,0,0])
        # Poisson negative log likelihood
        return np.sum(model_psf - measured_psf * np.log(model_psf))

    result = minimize(nll_poisson, x0=initial_guess, bounds=bounds, method='Nelder-Mead')
    return result.x



class Calibrator:
    def __init__(self, dirname):
        self.dirname = dirname
        self.images = []
        self.img_names = []
        self.gain = np.array(tifffile.imread('PIXSTAT/gain.tiff'))
        self.offset = np.array(tifffile.imread('PIXSTAT/offset.tiff'))
        self.variance = np.array(tifffile.imread('PIXSTAT/variance.tiff'))
        self.ROIs = []
        self.psf = HexPSF()
        h1 = 2.24
        h2 = 5.09
        self.psf.set_plate([0, h1, h2, 0, h1, h2])
        bead_ROIs = False
        bead_particles = False

        for image in sorted(os.listdir(self.dirname)):
            if image == 'ROIs.csv':
                bead_ROIs = True
                continue
            if image == 'particles.csv':
                bead_particles = True
                break

            self.images.append(correct_image(tifffile.imread(os.path.join(self.dirname, image)),
                                             self.gain, self.offset))
            self.img_names.append(image)

        if not bead_ROIs:
            find_blobs(self.dirname, int(len(os.listdir(dirname))))

        with open(self.dirname + 'ROIs.csv') as f:
            self.ROIs = f.read()
            f.close()
        self.ROIs = self.ROIs.split('\n')[:-1]

    def HexPSF_model(self, x0, y0, zp, wl, N, B):
        corrections = [0, 0, -0.2]
        z_modes = [0]
        return self.psf.intensity(x0, y0, zp, wl, N, B, corrections, z_modes)

    def calibrate(self):
        data = [['image', 'x0 (microns)', 'y0 (microns)', 'zp (microns)', 'wl (microns)', 'N (count)', 'B (count)', 'R^2']]
        i=0
        n=len(self.ROIs)
        for roi in self.ROIs:
            roi = roi.split(',')
            idx = int(float(roi[0]))
            x = int(float(roi[1]))
            y = int(float(roi[2]))
            image = self.images[idx][y + 15: y + 45, x + 15: x + 45]
            z = -(float(self.img_names[idx].split('_')[2][4:6]) - 10) * .05
            probe = int(self.img_names[idx].split('_')[4][-1])
            if probe == 0:
                wl = 0.683
            elif probe == 1:
                wl = 0.579
            elif probe == 2:
                wl = 0.513
            else:
                print("Unknown Probe")
            N = np.sum(image)
            B = (np.mean(image[0:, 0]) + np.mean(image[0:, 29]) + np.mean(image[0, 1:28]) + np.mean(
                image[29, 1:28])) / 4

            params = mle_poisson_fit_psf(image, self.HexPSF_model, initial_guess=(0, 0, z, wl, N, B), bounds=((-1,1), (-1,1), (-0.75,0.75), (0.5, 0.7), (0, None), (0,None)))
            x0 = params[0]
            y0 = params[1]
            zp = params[2]
            wl = params[3]
            N = params[4]
            B = params[5]

            model = self.HexPSF_model(x0, y0, zp, wl, N, B)

            mean = np.mean(image)
            s_res = (image - model) ** 2
            s_tot = (image - mean) ** 2
            R2 = 1 - np.sum(s_res) / np.sum(s_tot)

            data_row = [idx, x0, y0, zp, wl, N, B, R2]
            print(data_row)
            data.append(data_row)

            fig = plt.figure()
            ax = fig.add_subplot(1, 2, 1)
            ax.imshow(image)
            ax = fig.add_subplot(1,2,2)
            ax.imshow(model)
            plt.show(block=False)
            plt.pause(1)
            plt.close()

            # print(f"\rCalibrating HexPSFs... Progress: {round(i / n * 100, 3)}%", flush=True)
            i+=1

        data = pd.DataFrame(data)
        data.to_csv(self.dirname + 'particles.csv')










# plt.imshow(psf.intensity(x0=0.05,y0=-0.01,zp=-0.5,wl=0.683,N=50000,B=0.01, corrections=corrections))
# plt.show()
cal = Calibrator('hexpsf_all_cal/')
cal.calibrate()