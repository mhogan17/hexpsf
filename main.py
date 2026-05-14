import sys
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from Calibrator import Calibrator
from HexPSF import HexPSF


if len(sys.argv) < 2:
    print("Usage: python3 main.py [img_directory/] [# of frames]")
    exit()
else:
    img_dir = sys.argv[1]
    cal_dir = sys.argv[1][0:-1] + '_cal/'
    n = int(sys.argv[2])

    psf = HexPSF()
    h1 = 2.24
    h2 = 5.09
    psf.set_plate([0, h1, h2, 0, h1, h2])
    psf.show()

    # cal = Calibrator(cal_dir)
    # cal.calibrate(psf)
