import numpy as np
import matplotlib.pyplot as plt
from HexPSF import HexPSF, add_noise


def chernoff_information(nu1, nu2):
    nu1 = nu1.flatten()
    nu2 = nu2.flatten()

    beta = nu2/nu1
    # masked_beta = np.ma.masked_where(beta != 1, beta)
    # log_beta = np.log(beta)
    # masked_log_beta = np.ma.masked_where(log_beta != 0, log_beta)
    # return np.sum(nu1 * ((masked_beta - 1) * (np.log((masked_beta - 1) / masked_log_beta) - 1) + masked_log_beta) / masked_log_beta)

    total = 0.0
    for b in beta:
        if b == 1.0:
            b = 1.00001
        log_b = np.log(b)
        total += nu1 * ((b - 1) * (np.log((b - 1) / log_b) - 1) + log_b) / log_b
    return np.sum(total)


psf = HexPSF()
h1 = 2.24
h2 = 5.09
psf.set_plate([0,h1,h2,0,h1,h2])
# psf.show()

N = 2500
B = 100
# ref_psfs = [psf.intensity(0,0,0, wl, N, B) for wl in [0.513, 0.579, 0.683]]
# Cs = [[], [], []]
# Cs_err = [[], [], []]
# wls = np.arange(0.500, 0.700, 0.005)
# i=-1
# for wl in wls:
#     i+=1
#     Cs[0].append([])
#     Cs[1].append([])
#     Cs[2].append([])
#     for j in range(50):
#         temp_psf = add_noise(psf.intensity(0,0,0, wl, N, B))
#         C0 = chernoff_information(ref_psfs[0], temp_psf)
#         C1 = chernoff_information(ref_psfs[1], temp_psf)
#         C2 = chernoff_information(ref_psfs[2], temp_psf)
#         Cs[0][i].append(C0)
#         Cs[1][i].append(C1)
#         Cs[2][i].append(C2)
#
#     Cs_err[0].append(1.96 * np.std(Cs[0][i]) / np.sqrt(50))
#     Cs_err[1].append(1.96 * np.std(Cs[1][i]) / np.sqrt(50))
#     Cs_err[2].append(1.96 * np.std(Cs[2][i]) / np.sqrt(50))
#     Cs[0][i] = np.mean(Cs[0][i])
#     Cs[1][i] = np.mean(Cs[1][i])
#     Cs[2][i] = np.mean(Cs[2][i])
#
# plt.plot(wls, Cs[0], color='green')
# plt.plot(wls, Cs[1], color='orange')
# plt.plot(wls, Cs[2], color='red')
# plt.errorbar(wls, Cs[0], Cs_err[0], color='green', capsize=5)
# plt.errorbar(wls, Cs[1], Cs_err[1], color='orange', capsize=5)
# plt.errorbar(wls, Cs[2], Cs_err[2], color='red', capsize=5)
# plt.show()


Bs = [100, 250, 500, 1000, 2500, 5000, 7500, 10000, 50000, 100000, 500000, 1000000]
correct = [[0.0 for i in range(len(Bs))], [0.0 for i in range(len(Bs))], [0.0 for i in range(len(Bs))]]
wls = np.arange(0.500, 0.700, 0.001)
# green_neighborhood = np.arange(0.500, 0.545, 0.001)
# yellow_neighborhood = np.arange(0.560, 0.600, 0.001)
# red_neighborhood = np.arange(0.640, 0.700, 0.001)
# wls = np.concatenate((green_neighborhood, yellow_neighborhood, red_neighborhood))
# print(wls)

for i in range(len(Bs)):
    truth = [[0, 0, 0, 0], [0 ,0, 0, 0], [0, 0, 0, 0]]
    ref_psfs = [psf.intensity(0, 0, 0, wl, N, Bs[i]) for wl in [0.513, 0.579, 0.683]]
    ref_psfs.append(psf.intensity(0, 0, 0, 0.5, 0, Bs[i]))
    # for ref in ref_psfs:
        # plt.imshow(ref)
        # plt.show()
    for j in range(10):
        for wl in wls:
            temp_psf = add_noise(psf.intensity(0,0,0, wl, N, Bs[i]))
            # plt.imshow(temp_psf)
            # plt.show()
            C0 = chernoff_information(ref_psfs[0], temp_psf)
            C1 = chernoff_information(ref_psfs[1], temp_psf)
            C2 = chernoff_information(ref_psfs[2], temp_psf)
            C3 = chernoff_information(ref_psfs[3], temp_psf)
            # print(C0, C1, C2, C3)

            if wl < 0.550:
                if C0 < C1 and C0 < C2 and C0 < C3:
                    truth[0][0] += 1
                elif C1 < C2 and C1 < C3:
                    truth[0][1] += 1
                elif C2 < C3:
                    truth[0][2] += 1
                else:
                    truth[0][3] += 1

            elif wl < 0.622:
                if C1 < C0 and C1 < C2 and C1 < C3:
                    truth[1][1] += 1
                elif C0 < C2 and C0 < C3:
                    truth[1][0] += 1
                elif C2 < C3:
                    truth[1][2] += 1
                else:
                    truth[1][3] += 1

            else:
                if C2 < C0 and C2 < C1 and C2 < C3:
                    truth[2][2] += 1
                elif C0 < C1 and C0 < C3:
                    truth[2][0] += 1
                elif C1 < C3:
                    truth[2][1] += 1
                else:
                    truth[2][3] += 1

    print(truth)
    correct[0][i] = float(truth[0][0]) / np.sum(truth[0])
    correct[1][i] = float(truth[1][1]) / np.sum(truth[1])
    correct[2][i] = float(truth[2][2]) / np.sum(truth[2])
    print(f"Classifying PSFs... {round((i + 1) / len(Bs) * 100, 2)}% done...", flush=True)

plt.plot(Bs, correct[0], color='green', label='$\\nu_1$')
plt.plot(Bs, correct[1], color='orange', label='$\\nu_2$')
plt.plot(Bs, correct[2], color='red', label='$\\nu_3$')
plt.xlabel("Background Photons per Pixel")
plt.ylabel("Correct Classifications (%)")
plt.xscale('log')
plt.show()

# Test to identify signal from noise
# N=0
# B=100
# truth = [[0, 0, 0, 0], [0 ,0, 0, 0], [0, 0, 0, 0]]
#     ref_psfs = [psf.intensity(0, 0, 0, wl, N, B) for wl in [0.513, 0.579, 0.683]]
#     ref_psfs.append(psf.intensity(0, 0, 0, 0.5, 0, B))
#     # for ref in ref_psfs:
#         # plt.imshow(ref)
#         # plt.show()
#     for j in range(10):
#         for wl in wls:
#             temp_psf = add_noise(psf.intensity(0,0,0, wl, N, B))
#             # plt.imshow(temp_psf)
#             # plt.show()
#             C0 = chernoff_information(ref_psfs[0], temp_psf)
#             C1 = chernoff_information(ref_psfs[1], temp_psf)
#             C2 = chernoff_information(ref_psfs[2], temp_psf)
#             C3 = chernoff_information(ref_psfs[3], temp_psf)
#             # print(C0, C1, C2, C3)
#
#             if wl < 0.550:
#                 if C0 < C1 and C0 < C2 and C0 < C3:
#                     truth[0][0] += 1
#                 elif C1 < C2 and C1 < C3:
#                     truth[0][1] += 1
#                 elif C2 < C3:
#                     truth[0][2] += 1
#                 else:
#                     truth[0][3] += 1
#
#             elif wl < 0.622:
#                 if C1 < C0 and C1 < C2 and C1 < C3:
#                     truth[1][1] += 1
#                 elif C0 < C2 and C0 < C3:
#                     truth[1][0] += 1
#                 elif C2 < C3:
#                     truth[1][2] += 1
#                 else:
#                     truth[1][3] += 1
#
#             else:
#                 if C2 < C0 and C2 < C1 and C2 < C3:
#                     truth[2][2] += 1
#                 elif C0 < C1 and C0 < C3:
#                     truth[2][0] += 1
#                 elif C1 < C3:
#                     truth[2][1] += 1
#                 else:
#                     truth[2][3] += 1
#
#     print(truth)