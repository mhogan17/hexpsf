import numpy as np
import matplotlib.pyplot as plt


class PupilPlane:
    def __init__(self):
        with open("T53_FullSym.csv") as f:
            points = f.read()
            f.close()

        points = points.split("\n")
        w_j_sym = []
        th_j_sym = []
        u_j_sym = []
        threshold = 1e-12
        for point in points:
            point = point.split(",")
            u_j_sym.append(np.float64(point[1]))
            th_j_sym.append(np.float64(point[2]))
            w_j_sym.append(np.float64(point[3]))

        w_j_sym = np.array(w_j_sym)
        th_j_sym = np.array(th_j_sym)
        u_j_sym = np.array(u_j_sym)
        # Assuming u_j_sym, th_j_sym, w_j_sym, and threshold are already defined as numpy arrays/scalars
        # For example:
        # threshold = 1e-12

        # 1. Scale weights for points on lines of symmetry
        w_j_scaled = np.copy(w_j_sym)

        # Double weights of points on the x-axis
        mask_x = np.abs(th_j_sym) < threshold
        w_j_scaled[mask_x] = 2 * w_j_sym[mask_x]

        # Double weights of points on the y-axis (theta = -pi/2)
        mask_y = np.abs(th_j_sym + np.pi / 2) < threshold
        w_j_scaled[mask_y] = 2 * w_j_sym[mask_y]

        # Quadruple the weight of the origin point
        origin_mask = np.abs(u_j_sym) < threshold
        assert np.sum(origin_mask) == 1, "Can only have one point at the origin"
        w_j_scaled[origin_mask] = 4 * w_j_sym[origin_mask]

        # 2. Reflect points across the x-axis
        u_j_positiveY = np.copy(u_j_sym)
        th_j_positiveY = -th_j_sym
        w_j_positiveY = np.copy(w_j_scaled)

        # Determine indices of duplicated points on the x-axis
        # np.where returns a tuple, we take the first element [0]
        duplicate_pts = np.where(np.abs(th_j_positiveY) < threshold)[0]
        # Add index 0 (the origin) to duplicates
        duplicate_pts = np.unique(np.append(duplicate_pts, 0))

        # Remove duplicate points (using np.delete for index-based removal)
        u_j_positiveY = np.delete(u_j_positiveY, duplicate_pts)
        w_j_positiveY = np.delete(w_j_positiveY, duplicate_pts)
        th_j_positiveY = np.delete(th_j_positiveY, duplicate_pts)

        # Concatenate to get Quadrants 1 and 4
        u_j_q14 = np.concatenate([u_j_sym, u_j_positiveY])
        th_j_q14 = np.concatenate([th_j_sym, th_j_positiveY])
        w_j_q14 = np.concatenate([w_j_scaled, w_j_positiveY])

        # 3. Rotate to get Quadrants 2 and 3
        u_j_q23 = np.copy(u_j_q14)
        th_j_q23 = th_j_q14 + np.pi
        w_j_q23 = np.copy(w_j_q14)

        # Determine indices of duplicated points on the y-axis
        mask_y_pos = np.abs(th_j_q23 - np.pi / 2) < threshold
        mask_y_neg = np.abs(th_j_q23 - 3 * np.pi / 2) < threshold
        duplicate_pts_rot = np.where(mask_y_pos | mask_y_neg)[0]

        # Add origin (index 0) to duplicates
        duplicate_pts_rot = np.unique(np.append(duplicate_pts_rot, 0))

        # Remove duplicates from the rotated set
        u_j_q23 = np.delete(u_j_q23, duplicate_pts_rot)
        th_j_q23 = np.delete(th_j_q23, duplicate_pts_rot)
        w_j_q23 = np.delete(w_j_q23, duplicate_pts_rot)

        # 4. Final Concatenation
        u_j_full = np.concatenate([u_j_q14, u_j_q23])
        th_j_full = np.concatenate([th_j_q14, th_j_q23])
        w_j_full = np.concatenate([w_j_q14, w_j_q23])

        self.rho = u_j_full
        self.theta = th_j_full
        self.B_star = w_j_full
        self.Zernike = np.zeros_like(self.rho)

        # r = np.array([
        #     0.18089963670914432, 0.98391537714755427, 0.77364389614737803, 0.67975937134823609,
        #     0.59230560995561076, 0.41774444740353380, 0.89789289495208579, 0.93927352486965313,
        #     0.79099671385625722, 0.56605481445792489, 0.83180776468445906, 0.76010583069266321,
        #     0.62209339812108792, 0.36200059276768541
        # ])
        # s = np.array([
        #     0.0, 0.0, 0.77364389614737803, 0.67975937134823609,
        #     0.59230560995561076, 0.41774444740353380, 0.13375169526590821, 0.27985732696474981,
        #     0.39385576573951899, 0.10885005806786229, 0.51224072541565606, 0.12830595415488073,
        #     0.34691040719842391, 0.16676191877222966
        # ])
        # B = np.array([
        #     0.064712088156233115, 0.012203257346077736, 4.6214466764876138e-5,
        #     0.020941123674084990, 0.044321766954122515, 0.052715691149269700,
        #     0.028690025849304919, 0.013819709344648730, 0.037034888202855274,
        #     0.047520136607474831, 0.014296165120136605, 0.042923763669888949,
        #     0.047444099811669938, 0.063500222219468438
        # ])
        #
        #
        #
        # rs = []
        # B_vec = []
        #
        # for i in range(len(r)):
        #     if r[i] == s[i]:
        #         rs_temp = np.array([
        #             [r[i], -r[i], -r[i], r[i]],
        #             [s[i], s[i], -s[i], -s[i]]
        #         ])
        #         B_temp = [B[i]] * 4
        #     elif s[i] == 0:
        #         rs_temp = np.array([
        #             [r[i], -r[i], s[i], s[i]],
        #             [s[i], s[i], r[i], -r[i]]
        #         ])
        #         B_temp = [B[i]] * 4
        #     else:
        #         rs_temp = np.array([
        #             [r[i], -r[i], -r[i], r[i], s[i], -s[i], -s[i], s[i]],
        #             [s[i], s[i], -s[i], -s[i], r[i], r[i], -r[i], -r[i]]
        #         ])
        #         B_temp = [B[i]] * 8
        #
        #     rs.append(rs_temp)
        #     B_vec.extend(B_temp)
        #
        # rs = np.hstack(rs)
        # rt = np.zeros_like(rs)
        # rt[0, :] = np.sqrt(rs[0, :] ** 2 + rs[1, :] ** 2)
        # rt[1, :] = np.arctan2(rs[1, :], rs[0, :])
        #
        # self.B_star = np.array(B_vec, dtype=np.float32)
        # self.rho = rt[0, :].astype(np.float32)
        # self.theta = rt[1, :].astype(np.float32)
        # self.Zernike = np.zeros_like(self.rho)
