import numpy as np

def custom_sigmf(x, a, c):
    return 1 / (1 + np.exp(-a * (x - c)))

class HexPP:
    def __init__(self, x_off, y_off, theta_off, actuator_heights):
        self.x_off = x_off
        self.y_off = y_off
        self.theta_off = theta_off
        self.actuator_heights = np.array(actuator_heights)

    def get_HexPP(self, rho, theta):
        phase_plate = np.zeros_like(rho)
        for i in range(len(theta)):
            x = rho[i] * np.cos(theta[i]) - self.x_off
            y = rho[i] * np.sin(theta[i]) - self.y_off
            theta[i] = np.arctan2(y, x)
            angle = (theta[i] + self.theta_off) % (2 * np.pi)
            alpha = 10
            if 0 < angle <= np.pi / 3:
                phase_plate[i] = self.actuator_heights[0] * custom_sigmf(rho[i], alpha, -0.1)
            elif np.pi / 3 < angle <= 2 * np.pi / 3:
                phase_plate[i] = self.actuator_heights[1] * custom_sigmf(rho[i], alpha, -0.1)
            elif 2 * np.pi / 3 < angle <= np.pi:
                phase_plate[i] = self.actuator_heights[2] * custom_sigmf(rho[i], alpha, -0.1)
            elif np.pi / 3 < angle <= 4 * np.pi / 3:
                phase_plate[i] = self.actuator_heights[3] * custom_sigmf(rho[i], alpha, -0.1)
            elif 4 * np.pi / 3 < angle <= 5 * np.pi / 3:
                phase_plate[i] = self.actuator_heights[4] * custom_sigmf(rho[i], alpha, -0.1)
            else:
                phase_plate[i] = self.actuator_heights[5] * custom_sigmf(rho[i], alpha, -0.1)

        return phase_plate




