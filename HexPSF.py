import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from matplotlib.patches import RegularPolygon
from sympy.printing.pretty.pretty_symbology import line_width

import PupilPlane
import OpticsParams
import HexPP
from scipy.ndimage import gaussian_filter

def zernike_polynomials(j, rho, theta):
    phase = np.zeros_like(rho)
    for i in range(len(theta)):
        if j == 0:
            phase[i] += 1
        elif j == 1:
            phase[i] += 2 * rho[i] * np.sin(theta[i])
        elif j == 2:
            phase[i] += 2 * rho[i] * np.cos(theta[i])
        elif j == 3:
            phase[i] += np.sqrt(6) * rho[i] ** 2 * np.sin(2 * theta[i])
        elif j == 4:
            phase[i] += np.sqrt(3) * (2 * rho[i] ** 2 - 1)
        elif j == 5:
            phase[i] += np.sqrt(6) * rho[i] ** 2 * np.cos(2 * theta[i])
    return phase


def show_CRBs(psf):
    zs = np.array([(i - 10) / 20 for i in range(0, 21)])
    CRBx = [[], [], [], [], [], []]
    CRBy = [[], [], [], [], [], []]
    CRBz = [[], [], [], [], [], []]

    for z in zs:
        CRBx[0].append(1 / np.sqrt(psf.I_00(0, 0, z, 0.513, 2500, 100)))
        CRBx[1].append(1 / np.sqrt(psf.I_00(0, 0, z, 0.579, 2500, 100)))
        CRBx[2].append(1 / np.sqrt(psf.I_00(0, 0, z, 0.683, 2500, 100)))

        CRBx[3].append(1 / np.sqrt(psf.I_00(0, 0, z, 0.513, 2500, 400)))
        CRBx[4].append(1 / np.sqrt(psf.I_00(0, 0, z, 0.579, 2500, 400)))
        CRBx[5].append(1 / np.sqrt(psf.I_00(0, 0, z, 0.683, 2500, 400)))

        CRBy[0].append(1 / np.sqrt(psf.I_11(0, 0, z, 0.513, 2500, 100)))
        CRBy[1].append(1 / np.sqrt(psf.I_11(0, 0, z, 0.579, 2500, 100)))
        CRBy[2].append(1 / np.sqrt(psf.I_11(0, 0, z, 0.683, 2500, 100)))

        CRBy[3].append(1 / np.sqrt(psf.I_11(0, 0, z, 0.513, 2500, 400)))
        CRBy[4].append(1 / np.sqrt(psf.I_11(0, 0, z, 0.579, 2500, 400)))
        CRBy[5].append(1 / np.sqrt(psf.I_11(0, 0, z, 0.683, 2500, 400)))

        CRBz[0].append(1 / np.sqrt(psf.I_22(0, 0, z, 0.513, 2500, 100)))
        CRBz[1].append(1 / np.sqrt(psf.I_22(0, 0, z, 0.579, 2500, 100)))
        CRBz[2].append(1 / np.sqrt(psf.I_22(0, 0, z, 0.683, 2500, 100)))

        CRBz[3].append(1 / np.sqrt(psf.I_22(0, 0, z, 0.513, 2500, 400)))
        CRBz[4].append(1 / np.sqrt(psf.I_22(0, 0, z, 0.579, 2500, 400)))
        CRBz[5].append(1 / np.sqrt(psf.I_22(0, 0, z, 0.683, 2500, 400)))

    fig = plt.figure()
    ax0 = fig.add_subplot(131)
    ax0.plot(zs, CRBx[0], color='green')
    ax0.plot(zs, CRBx[1], color='orange')
    ax0.plot(zs, CRBx[2], color='red')
    ax0.plot(zs, CRBx[3], color='green', linestyle='dashed')
    ax0.plot(zs, CRBx[4], color='orange', linestyle='dashed')
    ax0.plot(zs, CRBx[5], color='red', linestyle='dashed')
    ax0.hlines(0.01, -0.5, 0.5, color='black', linestyles='dotted', linewidth=3.0)
    ax0.set_ylim(0.005, 0.13)
    ax0.set_ylabel("$CRB_{x_0}\ (\mu m)$")
    ax0.set_xlabel("$z_{p}\ (\mu m)$")
    ax0.minorticks_on()
    ax0.yaxis.set_minor_locator(tck.AutoMinorLocator(2))
    ax0.grid()
    ax0.grid(which='minor', linestyle=':', linewidth='0.5')


    ax1 = fig.add_subplot(132)
    ax1.plot(zs, CRBy[0], color='green')
    ax1.plot(zs, CRBy[1], color='orange')
    ax1.plot(zs, CRBy[2], color='red')
    ax1.plot(zs, CRBy[3], color='green', linestyle='dashed')
    ax1.plot(zs, CRBy[4], color='orange', linestyle='dashed')
    ax1.plot(zs, CRBy[5], color='red', linestyle='dashed')
    ax1.hlines(0.01, -0.5, 0.5, color='black', linestyles='dotted', linewidth=3.0)
    ax1.set_ylim(0.005, 0.13)
    ax1.set_ylabel("$CRB_{y_0}\ (\mu m)$")
    ax1.set_xlabel("$z_{p}\ (\mu m)$")
    ax1.minorticks_on()
    ax1.yaxis.set_minor_locator(tck.AutoMinorLocator(2))
    ax1.grid()
    ax1.grid(which='minor', linestyle=':', linewidth='0.5')

    ax2 = fig.add_subplot(133)
    ax2.plot(zs, CRBz[0], color='green', label='$\lambda = 513\ nm,\ B=100$')
    ax2.plot(zs, CRBz[1], color='orange', label='$\lambda = 579\ nm,\ B=100$')
    ax2.plot(zs, CRBz[2], color='red', label='$\lambda = 683\ nm,\ B=100$')
    ax2.plot(zs, CRBz[3], color='green', linestyle='dashed', label='$\lambda = 513\ nm,\ B=400$')
    ax2.plot(zs, CRBz[4], color='orange', linestyle='dashed', label='$\lambda = 579\ nm,\ B=400$')
    ax2.plot(zs, CRBz[5], color='red', linestyle='dashed', label='$\lambda = 683\ nm,\ B=400$')
    ax2.hlines(0.01, -0.5, 0.5, color='black', linestyles='dotted', linewidth=3.0, label='$CRB\\approx 10$ nm without Hex-PP ')
    # ax2.set_yscale('log')
    ax2.set_ylim(0.005, 0.13)
    ax2.set_ylabel("$CRB_{z_p}\ (\mu m)$")
    ax2.set_xlabel("$z_{p}\ (\mu m)$")
    ax2.minorticks_on()
    ax2.yaxis.set_minor_locator(tck.AutoMinorLocator(2))
    ax2.grid()
    ax2.grid(which='minor', linestyle=':', linewidth='0.5')
    ax2.legend(loc='upper right', bbox_to_anchor=(-1.65, 1), shadow=True, fancybox=True)


    plt.tight_layout()
    plt.show()

def add_noise(psf):
    return np.random.poisson(psf)

def zernike_polynomials(j, rho, theta):
    phase = np.zeros_like(rho)
    for i in range(len(theta)):
        if j == 0:
            phase[i] += 1
        elif j == 1:
            phase[i] += 2 * rho[i] * np.sin(theta[i])
        elif j == 2:
            phase[i] += 2 * rho[i] * np.cos(theta[i])
        elif j == 3:
            phase[i] += np.sqrt(6) * rho[i] ** 2 * np.sin(2 * theta[i])
        elif j == 4:
            phase[i] += np.sqrt(3) * (2 * rho[i] ** 2 - 1)
        elif j == 5:
            phase[i] += np.sqrt(6) * rho[i] ** 2 * np.cos(2 * theta[i])
    return phase


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

    def set_corrections(self, corrections):
        self.HexPP.x_off = corrections[0]
        self.HexPP.y_off = corrections[1]
        self.HexPP.theta_off = corrections[2]

    def GibsonLaniPSFModel(self, x0, y0, zp, wl):
        x = self.x
        y = self.y
        k = 2 * np.pi / wl
        i = 1j
        NA = self.optics.NA
        n_p = self.optics.n_p
        n_s = self.optics.n_s
        n_i = self.optics.n_i
        n_a = self.optics.n_a
        zd = self.optics.zd
        zd_star = self.optics.zd_star
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
        hex_pp = self.HexPP.get_HexPP(rho, theta)
        term4 = (n_p - n_a) * hex_pp

        # Camera defocus term
        term5 = 1 / 2 * rho * (zd_star - zd) / zd_star * zd

        # Zernike correction term
        term6 = self.Pupil_Plane.Zernike

        # Combine phase terms
        phase = -i * k * (term1 + term2 + term3 + term4 + term5 + term6)

        # Complete integrand
        integrand = k * np.exp(phase)

        # Multiply with weight (Bstar) and integrate over unit disk (axis=2) to get wavefront (see Cools and Kim paper)
        integral = np.sum(integrand * B_star, axis=2)
        intensity = np.real(integral * np.conj(integral))

        # Return normalized intensity
        out = intensity / np.sum(intensity)
        return out

    def I_00(self, x0, y0, zp, wl, N, B):
        x = self.x
        y = self.y
        self.x0 = np.array(x0, dtype=np.float32).flatten()
        self.y0 = np.array(y0, dtype=np.float32).flatten()
        self.zp = np.array(zp, dtype=np.float32).flatten()
        self.wl = np.array(wl, dtype=np.float32).flatten()

        model = self.GibsonLaniPSFModel(self.x0, self.y0, self.zp, self.wl)
        denominator = N * model + B * np.ones_like(model)

        k = 2 * np.pi / wl
        i = 1j
        NA = self.optics.NA
        M = self.optics.M
        n_p = self.optics.n_p
        n_s = self.optics.n_s
        n_i = self.optics.n_i
        n_a = self.optics.n_a
        zd = self.optics.zd
        zd_star = self.optics.zd_star
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
        self.HexPP.x_off = self.corrections[0]
        self.HexPP.y_off = self.corrections[1]
        self.HexPP.theta_off = self.corrections[2]

        hex_pp = self.HexPP.get_HexPP(rho, theta)
        term4 = (n_p - n_a) * hex_pp

        # Camera defocus term
        term5 = 1 / 2 * rho * (zd_star - zd) / zd_star * zd

        # Zernike correction term
        term6 = self.Pupil_Plane.Zernike

        # Combine phase terms
        phase = -i * k * (term1 + term2 + term3 + term4 + term5 + term6)

        # Complete integrand
        # integrand = k * rho * np.exp(phase)
        # # Multiply with Bstar and integrate over unit disk (axis=2)
        # integral = np.sum(integrand * B_star, axis=2)
        # intensity = np.real(integral * np.conj(integral))
        #
        # A = 1 / np.sum(intensity)
        dPhiGL_dx0 = NA * rho / r * (x0 * np.cos(angle_diff) + (y-y0) * np.sin(angle_diff) - x * np.cos(angle_diff))

        integrand = k * np.exp(phase) * i * k * dPhiGL_dx0
        integral = np.sum(integrand * B_star, axis=2)
        intensity = np.real(integral * np.conj(integral))
        model = intensity / np.sum(intensity)
        numerator = (N * model) ** 2

        # plt.imshow(numerator / denominator)
        # plt.show()

        return np.sum(numerator / denominator)

    def I_11(self, x0, y0, zp, wl, N, B):
        x = self.x
        y = self.y
        self.x0 = np.array(x0, dtype=np.float32).flatten()
        self.y0 = np.array(y0, dtype=np.float32).flatten()
        self.zp = np.array(zp, dtype=np.float32).flatten()
        self.wl = np.array(wl, dtype=np.float32).flatten()

        model = self.GibsonLaniPSFModel(self.x0, self.y0, self.zp, self.wl)
        denominator = N * model + B * np.ones_like(model)

        k = 2 * np.pi / wl
        i = 1j
        NA = self.optics.NA
        M = self.optics.M
        n_p = self.optics.n_p
        n_s = self.optics.n_s
        n_i = self.optics.n_i
        n_a = self.optics.n_a
        zd = self.optics.zd
        zd_star = self.optics.zd_star
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
        self.HexPP.x_off = self.corrections[0]
        self.HexPP.y_off = self.corrections[1]
        self.HexPP.theta_off = self.corrections[2]

        hex_pp = self.HexPP.get_HexPP(rho, theta)
        term4 = (n_p - n_a) * hex_pp

        # Camera defocus term
        term5 = 1 / 2 * rho * (zd_star - zd) / zd_star * zd

        # Zernike correction term
        term6 = self.Pupil_Plane.Zernike

        # Combine phase terms
        phase = -i * k * (term1 + term2 + term3 + term4 + term5 + term6)

        dPhiGL_dy0 = NA * rho / r * (y0 * np.cos(angle_diff) + (x - x0) * np.sin(angle_diff) - y * np.cos(angle_diff))

        integrand = k * np.exp(phase) * i * k * dPhiGL_dy0
        integral = np.sum(integrand * B_star, axis=2)
        intensity = np.real(integral * np.conj(integral))
        model = intensity / np.sum(intensity)
        numerator = (N * model) ** 2

        # plt.imshow(numerator / denominator)
        # plt.show()

        return np.sum(numerator / denominator)

    def I_22(self, x0, y0, zp, wl, N, B):
        x = self.x
        y = self.y
        self.x0 = np.array(x0, dtype=np.float32).flatten()
        self.y0 = np.array(y0, dtype=np.float32).flatten()
        self.zp = np.array(zp, dtype=np.float32).flatten()
        self.wl = np.array(wl, dtype=np.float32).flatten()

        model = self.GibsonLaniPSFModel(self.x0, self.y0, self.zp, self.wl)
        denominator = N * model + B * np.ones_like(model)

        k = 2 * np.pi / wl
        i = 1j
        NA = self.optics.NA
        M = self.optics.M
        n_p = self.optics.n_p
        n_s = self.optics.n_s
        n_i = self.optics.n_i
        n_a = self.optics.n_a
        zd = self.optics.zd
        zd_star = self.optics.zd_star
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
        self.HexPP.x_off = self.corrections[0]
        self.HexPP.y_off = self.corrections[1]
        self.HexPP.theta_off = self.corrections[2]

        hex_pp = self.HexPP.get_HexPP(rho, theta)
        term4 = (n_p - n_a) * hex_pp

        # Camera defocus term
        term5 = 1 / 2 * rho * (zd_star - zd) / zd_star * zd

        # Zernike correction term
        term6 = self.Pupil_Plane.Zernike

        # Combine phase terms
        phase = -i * k * (term1 + term2 + term3 + term4 + term5 + term6)

        dPhiGL_dzp = n_s * np.sqrt(1 - (NA ** 2 * rho ** 2) / n_s ** 2)

        integrand = k * np.exp(phase) * i * k * dPhiGL_dzp
        integral = np.sum(integrand * B_star, axis=2)
        intensity = np.real(integral * np.conj(integral))
        model = intensity / np.sum(intensity)
        numerator = (N * model) ** 2

        # plt.imshow(numerator / denominator)
        # plt.show()

        return np.sum(numerator / denominator)


    def set_plate(self, plate):
        self.HexPP.actuator_heights = plate

    def intensity(self, x0, y0, zp, wl, N, B, corrections, z_modes):
        self.set_corrections(corrections)
        # for i in range(len(z_modes)):
        #     self.Pupil_Plane.Zernike += z_modes[i] * zernike_polynomials(i, self.Pupil_Plane.rho, self.Pupil_Plane.theta)

        # Return the intensity profile of the PSF
        self.x0 = np.array(x0, dtype=np.float32).flatten()
        self.y0 = np.array(y0, dtype=np.float32).flatten()
        self.zp = np.array(zp, dtype=np.float32).flatten()
        self.wl = np.array(wl, dtype=np.float32).flatten()

        model = self.GibsonLaniPSFModel(self.x0, self.y0, self.zp, self.wl)
        model = N * model + B * np.ones_like(model)
        return model

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

        axs[1].imshow(add_noise(self.intensity(x0=0, y0=0, zp=0, wl=0.513, N=25000, B=100)))
        axs[2].imshow(add_noise(self.intensity(x0=0, y0=0, zp=0, wl=0.579, N=25000, B=100)))
        axs[3].imshow(add_noise(self.intensity(x0=0, y0=0, zp=0, wl=0.683, N=25000, B=100)))
        # axs[4].imshow(self.intensity(x0=0, y0=0, zp=0, wl=0.751, corrections=[0, 0, 0, 0]))

        axs[0].text(0.5, 1.15, "Phase Plate", ha='center', va='center', fontsize=12)
        axs[1].text(10, -4, "green", ha='center', va='center', fontsize=12)
        axs[2].text(10, -4, "yellow", ha='center', va='center', fontsize=12)
        axs[3].text(10, -4, "red", ha='center', va='center', fontsize=12)
        # axs[4].text(10, -4, "Dark Red", ha='center', va='center', fontsize=12)

        plt.autoscale(enable=True)
        plt.show()



# show_CRBs(psf)