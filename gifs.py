from PIL import Image
import os
import matplotlib.pyplot as plt
import tifffile

import tifffile
import numpy as np



def create_gif(image_paths, output_path, duration=100):
    """
    Creates a GIF from a list of image paths.

    Args:
        image_paths (list): List of paths to the TIFF images.
        output_path (str): Path to save the output GIF file.
        duration (int, optional): Duration of each frame in milliseconds. Defaults to 100.
    """
    frames = [Image.open(image_path) for image_path in image_paths]
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=duration, loop=0)


def correct_image(raw_image, gain_image, offset_image, variance_image=None, return_variance=False):
    """
    Apply gain and offset correction to a raw fluorescence image.

    Parameters:
    - raw_image: 2D NumPy array (raw image from camera)
    - gain_image: 2D NumPy array (gain per pixel)
    - offset_image: 2D NumPy array (offset per pixel)
    - variance_image: 2D NumPy array (optional, variance per pixel)
    - return_variance: if True, returns corrected variance as well

    Returns:
    - corrected_image: photon-count estimate per pixel
    - (optional) corrected_variance
    """
    corrected = (raw_image.astype(np.float32) - offset_image) / gain_image

    if return_variance:
        if variance_image is not None:
            corrected_var = variance_image / (gain_image ** 2)
        else:
            corrected_var = np.ones_like(corrected)  # dummy if not provided
        return corrected, corrected_var

    return corrected
#
#

gain = tifffile.imread('PIXSTAT/gain.tiff')
offset = tifffile.imread('PIXSTAT/offset.tiff')
variance = tifffile.imread('PIXSTAT/variance.tiff')

tif_dir = 'hexpsf5/'
tifs = []
n = 4000
count = -1
for fname in sorted(os.listdir(tif_dir)):
    print(f"\rProgress: {round(count/n * 100, 3)}%", flush=True, end='')
    count+=1
    if count > n:
        break
    if fname == 'ROIs.csv' or fname == 'particles.csv':
        continue
    if fname.endswith('2_1.tif'):
        with tifffile.TiffFile(tif_dir + fname) as tif:
            image = tif.asarray()
            image = 5*correct_image(image, gain, offset)[150:, 50:450]
            # plt.imshow(image)
            # plt.show()

        tifffile.imwrite('gif_tifs/' + fname, image)
        tifs.append('gif_tifs/' + fname)

        # im = tifffile.imread('h_tifs/' + fname)
        # plt.imshow(im)
        # plt.colorbar()
        # plt.show()


create_gif(tifs, 'hexpsf5_green.gif', 10)