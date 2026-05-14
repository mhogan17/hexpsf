import tifffile
import numpy as np
import matplotlib.pyplot as plt
from HexPSF import HexPSF
from scipy.optimize import minimize
np.set_printoptions(suppress=True)

def mle_poisson_fit(measured_psf, model_func,
                           initial_guess=(0,0,0,0,0), bounds=None):
    measured_psf = measured_psf.astype(np.float64)

    def nll_poisson(params):
        x, y, dt, N, B = params
        model_psf = model_func(x0=x, y0=y, dt=dt, N=N, B=B)
        return np.sum(model_psf - measured_psf * np.log(model_psf))

    result = minimize(nll_poisson, x0=initial_guess, bounds=bounds, method="Nelder-Mead")
    return result.x

# red = tifffile.imread("hex_beads/img000_000_000011_0000000000_0000000002_1.tif")[318:348, 312:342]
# yellow = tifffile.imread("hex_beads/img000_000_000011_0000000000_0000000001_1.tif")[318:348, 312:342]
# green = tifffile.imread("hex_beads/img000_000_000011_0000000000_0000000000_1.tif")[318:348, 312:342]

# red = tifffile.imread("hex_beads/img000_000_000000_0000000000_0000000002_1.tif")[318:348, 312:342]
# yellow = tifffile.imread("hex_beads/img000_000_000000_0000000000_0000000001_1.tif")[318:348, 312:342]
# green = tifffile.imread("hex_beads/img000_000_000000_0000000000_0000000000_1.tif")[318:348, 312:342]

red = tifffile.imread("hex_beads/img000_000_000020_0000000000_0000000002_1.tif")[318:348, 312:342]
yellow = tifffile.imread("hex_beads/img000_000_000020_0000000000_0000000001_1.tif")[318:348, 312:342]
green = tifffile.imread("hex_beads/img000_000_000020_0000000000_0000000000_1.tif")[318:348, 312:342]




def model(x0, y0, dt, N, B):
    psf = HexPSF()
    h1 = 2.24
    h2 = 5.09
    psf.set_plate([0, h1, h2, 0, h1, h2])
    psf.optics.delta_t = dt
    psf.optics.delta_z = 1.5
    corrections = [0, 0, -0.18]
    z_modes = [0]
    all = []
    all.append(psf.intensity(x0, y0, 0.05, 0.683, N, B, corrections, z_modes))
    all.append(psf.intensity(x0, y0, 0.05, 0.579, N, B, corrections, z_modes))
    all.append(psf.intensity(x0, y0, 0.05, 0.513, N, B, corrections, z_modes))
    all = np.array(all)
    return all

all = []
all.append(red)
all.append(yellow)
all.append(green)
all = np.array(all)
#
params = mle_poisson_fit(all, model, initial_guess=(0, 0, 1.5, 10000, 100), bounds=((None, None), (None, None),
                                                                                  (None, None), (0, None), (0, None)))
print(np.array(params))
x = params[0]
y = params[1]
dt = params[2]
# dz = params[3]
N = params[3]
B = params[4]


fig = plt.figure()
ax = fig.add_subplot(321)
ax.imshow(red)
ax = fig.add_subplot(322)
ax.imshow(model(x, y, dt, N, B)[0])

ax = fig.add_subplot(323)
ax.imshow(yellow)
ax = fig.add_subplot(324)
ax.imshow(model(x, y, dt, N, B)[1])

ax = fig.add_subplot(325)
ax.imshow(green)
ax = fig.add_subplot(326)
ax.imshow(model(x, y, dt, N, B)[2])

plt.show()


# psf = HexPSF()
# h1 = 2.24
# h2 = 5.09
# psf.set_plate([0, h1, h2, 0, h1, h2])
# psf.optics.zd_star = 1
# psf.optics.zd = 1

# for dt in np.arange(-1, 1, 0.1):
#     psf.optics.delta_t = dt
#
#     fig = plt.figure()
#
#     ax = fig.add_subplot(321)
#     ax.imshow(red)
#     ax = fig.add_subplot(322)
#     ax.imshow(psf.intensity(0, 0, 0.1, 0.683, 10000, 0, corrections=[0, 0, -0.18], z_modes=[]))
#
#     ax = fig.add_subplot(323)
#     ax.imshow(yellow)
#     ax = fig.add_subplot(324)
#     ax.imshow(psf.intensity(0, 0, 0.1, 0.579, 10000, 0, corrections=[0, 0, -0.18], z_modes=[]))
#
#     ax = fig.add_subplot(325)
#     ax.imshow(green)
#     ax = fig.add_subplot(326)
#     ax.imshow(psf.intensity(0, 0, 0.1, 0.513, 10000, 0, corrections=[0, 0, -0.18], z_modes=[]))
#
#     plt.show()
