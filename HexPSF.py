import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import PupilPlane
import OpticsParams
import HexPP
from scipy.ndimage import gaussian_filter


class HexPSF:
    def __init__(self):
        self.optics = OpticsParams.OpticsParams()
        self.Pupil_Plane = PupilPlane.PupilPlane()
        self.HexPP = HexPP.HexPP(x_off=0, y_off=0, theta_off=0, actuator_heights=[0,0,0,0,0,0])

        x = np.linspace(self.optics.PixelSize * -self.optics.K / 2,
                        self.optics.PixelSize * self.optics.K / 2,
                        self.optics.K) / self.optics.M

        X, Y = np.meshgrid(x, x)
        num_el_rho = np.size(self.Pupil_Plane.rho)

        self.x = np.tile(X.astype(np.float32)[..., np.newaxis], (1, 1, num_el_rho))
        self.y = np.tile(Y.astype(np.float32)[..., np.newaxis], (1, 1, num_el_rho))

        self.x0 = np.zeros_like(self.x)
        self.y0 = np.zeros_like(self.x)
        self.zp = np.zeros_like(self.x)
        self.wl = np.zeros_like(self.x)

    def GibsonLaniPSFModel(self, x, y, x0, y0, zp, wl, corrections):
        # Wavenumber
        k = 2 * np.pi / wl
        i = 1j
        NA = self.optics.NA
        n_p = self.optics.n_p
        n_s = self.optics.n_s
        n_i = self.optics.n_i
        n_a = self.optics.n_a
        delta_t = self.optics.delta_t
        rho = self.Pupil_Plane.rho
        theta = self.Pupil_Plane.theta
        B_star = self.Pupil_Plane.B_star



        # Axial distance terms
        term1 = delta_t * n_i * np.sqrt(1 - (NA ** 2 * rho ** 2) / n_i ** 2)
        term2 = zp * n_s * np.sqrt(1 - (NA ** 2 * rho ** 2) / n_s ** 2)

        # Lateral distance term
        r = np.sqrt((x0 - x) ** 2 + (y0 - y) ** 2)
        angle_diff = theta - np.arctan2(x0 - x, y0 - y)
        term3 = (NA * rho) * r * np.cos(angle_diff)

        # XPP term
        self.HexPP.x_off = corrections[0]
        self.HexPP.y_off = corrections[1]
        self.HexPP.theta_off = corrections[2]

        hex_pp = self.HexPP.get_HexPP(rho, theta)
        term4 = (n_p - n_a) * hex_pp

        # Zernike correction term
        term5 = np.zeros_like(term4)
        count = 1
        for z in corrections[3:]:
            term5 += z * self.zernike_polynomials(count, rho, theta)
            count += 1

        # Combine phase terms
        phase = -i * k * (term1 + term2 + term3 + term4 + term5)

        # Final prefactor
        prefactor = k * (NA ** 2 * rho)

        # Complete integrand
        integrand = np.exp(phase) * prefactor

        # Multiply with Bstar and sum over 3rd dimension (axis=2)
        out = np.sum(integrand * B_star, axis=2)

        return out

    def set_plate(self, plate):
        self.HexPP.actuator_heights = plate

    def intensity(self, x0, y0, zp, wl, N, B, corrections):
        # Return the intensity profile of the PSF
        self.x0 = np.array(x0, dtype=np.float32).flatten()
        self.y0 = np.array(y0, dtype=np.float32).flatten()
        self.zp = np.array(zp, dtype=np.float32).flatten()
        self.wl = np.array(wl, dtype=np.float32).flatten()

        integral = self.GibsonLaniPSFModel(self.x, self.y, self.x0, self.y0, self.zp, self.wl, corrections)

        model = np.real(integral * np.conj(integral))
        model_sum = np.sum(model)
        model = N * (model / model_sum) + B * np.ones_like(model)
        return gaussian_filter(model, sigma=1)

    def show(self):
        plate = self.HexPP.actuator_heights
        fig, axs = plt.subplots(1, 4)
        axs[0].set_aspect('equal')  # Ensure correct aspect ratio

        hexagon = RegularPolygon(
            (0.5, 0.5),  # Center coordinates
            numVertices=6,  # Number of vertices (for hexagon)
            radius=0.5,
            facecolor='yellow',
            edgecolor='black'
        )
        axs[0].add_patch(hexagon)
        axs[0].plot([0.072, 0.93], [0.25, 0.75], color='red', linewidth=1, linestyle='dotted')
        axs[0].plot([0.072, 0.93], [0.75, 0.25], color='red', linewidth=1, linestyle='dotted')
        axs[0].plot([0.5, 0.5], [0, 1], color='red', linewidth=1, linestyle='dotted')


        axs[0].text(0.2, 0.5, round(plate[0], 2), ha='center', va='center', fontsize=10)
        axs[0].text(0.33, 0.23, round(plate[1], 2), ha='center', va='center', fontsize=10)
        axs[0].text(0.67, 0.23, round(plate[2], 2), ha='center', va='center', fontsize=10)
        axs[0].text(0.8, 0.5, round(plate[3], 2), ha='center', va='center', fontsize=10)
        axs[0].text(0.67, 0.75, round(plate[4], 2), ha='center', va='center', fontsize=10)
        axs[0].text(0.33, 0.75, round(plate[5], 2), ha='center', va='center', fontsize=10)

        for i in range(0, 4):
            axs[i].set_xticks([])
            axs[i].set_yticks([])

        axs[1].imshow(self.intensity(x0=0, y0=0, zp=0, wl=0.513, N=100000, B=1000, corrections=[0, 0, 0, 0]))
        axs[2].imshow(self.intensity(x0=0, y0=0, zp=0, wl=0.579, N=100000, B=1000, corrections=[0, 0, 0, 0]))
        axs[3].imshow(self.intensity(x0=0, y0=0, zp=0, wl=0.683, N=100000, B=1000, corrections=[0, 0, 0, 0]))
        # axs[4].imshow(self.intensity(x0=0, y0=0, zp=0, wl=0.751, corrections=[0, 0, 0, 0]))

        axs[0].text(0.5, 1.15, "Phase Plate", ha='center', va='center', fontsize=12)
        axs[1].text(10, -4, "$green$", ha='center', va='center', fontsize=12)
        axs[2].text(10, -4, "$yellow$", ha='center', va='center', fontsize=12)
        axs[3].text(10, -4, "$red$", ha='center', va='center', fontsize=12)
        # axs[4].text(10, -4, "Dark Red", ha='center', va='center', fontsize=12)

        plt.autoscale(enable=True)
        plt.show()

    def zernike_polynomials(self, j, rho, theta):
        phase = np.zeros_like(rho)
        for i in range(len(theta)):
            if j == 0:
                phase[i] += 1
            elif j == 1:
                phase[i] += 2 * rho[i] * np.sin(theta[i])
            elif j==2:
                phase[i] += 2 * rho[i] * np.cos(theta[i])
            elif j==3:
                phase[i] += np.sqrt(6) * rho[i] ** 2 * np.sin(2 * theta[i])
            elif j==4:
                phase[i] += np.sqrt(3) * (2 * rho[i] ** 2 - 1)
            elif j==5:
                phase[i] += np.sqrt(6) * rho[i] ** 2 * np.cos(2 * theta[i])

        return phase

# psf = HexPSF()

# h1 = 2.24
# h2 = 5.09
# psf.set_plate([0, h1, h2, 0, h1, h2])
#
# corrections = [ 0.00271894,  0.0043681,   0.00011607,  0.00118429, -0.00354776, -0.00162042]
# corrections=[0,0,0,0]
#
# plt.imshow(psf.intensity(x0=0.05,y0=-0.01,zp=-0.5,wl=0.513,N=50000,B=0.01, corrections=corrections))
# plt.show()
# psf.show()
# for i in range(0, 10):
#     plt.imshow(psf.intensity(0,0,0,0.683, corrections=[0, 0, i * np.pi/6, 0]))
#     plt.show()
# psf.show()
#
#
# for z in [i / 10 for i in range(0, 10)]:
#     print(z)
#     fig, axs = plt.subplots(2, 3)
#     axs[0][0].imshow(psf.intensity(x0=0, y0=0, zp=z, wl=0.513, corrections=[0,0,0,0]))
#     axs[0][1].imshow(psf.intensity(x0=0, y0=0, zp=z, wl=0.579, corrections=[0,0,0,0]))
#     axs[0][2].imshow(psf.intensity(x0=0, y0=0, zp=z, wl=0.683, corrections=[0,0,0,0]))
#     axs[1][0].imshow(psf.intensity(x0=0, y0=0, zp=-z, wl=0.513, corrections=[0,0,0,0]))
#     axs[1][1].imshow(psf.intensity(x0=0, y0=0, zp=-z, wl=0.579, corrections=[0,0,0,0]))
#     axs[1][2].imshow(psf.intensity(x0=0, y0=0, zp=-z, wl=0.683, corrections=[0,0,0,0]))
#     for i in range(0, 3):
#         for j in range(0, 2):
#             axs[j][i].set_xticks([])
#             axs[j][i].set_yticks([])
#     plt.show()

    # [0, 4.989022173205141, 5.065769998975962, 1.256802167271463, 4.989022173205141, 5.065769998975962]
    # [0, 0.13943141436376483, 0.8488720724813718, 5.39898253200691, 0.13943141436376483, 0.8488720724813718]
    # [0, 5.353390779185647, 1.5896664598886472, 0, 5.353390779185647, 1.5896664598886472]
    # [0, 3.1434701347896854, 5.466305658121761, 0, 3.1434701347896854, 5.466305658121761]
    # [0, 1.5752852919312834, 5.105200002905643, 0, 1.5752852919312834, 5.105200002905643]
    # [0, 3.7939180963605956, 1.5705311300853158, 0, 3.7939180963605956, 1.5705311300853158]
    # [0, 5.0705097304386, 2.229193871243189, 0, 5.0705097304386, 2.229193871243189]
    # [0, 2.8423410662996216, 4.45478806125667, 0, 2.8423410662996216, 4.45478806125667]
    # [0, 3.8319155860217498, 0.979478867432857, 0, 3.8319155860217498, 0.979478867432857]