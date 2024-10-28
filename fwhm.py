#! /usr/bin/env python
import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from astropy.stats import mad_std
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture
from plotting_tools import plot_images
import warnings
import argparse

warnings.filterwarnings('ignore', category=UserWarning)

plot_images()


# Function to fit a 2D Gaussian
def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
    g = offset + amplitude * np.exp(-(a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2)))
    return g.ravel()


# Argument parser to allow command-line input
parser = argparse.ArgumentParser(description='Measure FWHM from a FITS image.')
parser.add_argument('image', type=str, help='Path to the FITS image')
parser.add_argument('--crop_size', type=int, default=800, help='CMOS = 800, CCD = 652')
args = parser.parse_args()

# Load the FITS file
fits_image_filename = args.image
hdul = fits.open(fits_image_filename)
image_data = hdul[0].data
hdul.close()

# Define the central region
center_x, center_y = image_data.shape[1] // 2, image_data.shape[0] // 2
crop_size = args.crop_size  # Half the size of the 500x500 area
cropped_image_data = image_data[center_y - crop_size:center_y + crop_size, center_x - crop_size:center_x + crop_size]

# Estimate background noise level
mean, median, std = np.mean(cropped_image_data), np.median(cropped_image_data), mad_std(cropped_image_data)
daofind = DAOStarFinder(fwhm=4, threshold=5. * std, brightest=250)
selected_sources = daofind(cropped_image_data - median)
print(f"Number of sources found: {len(selected_sources)}")

fwhms_x = []
fwhms_y = []

# Iterate over a subset of stars for FWHM calculation
for i, (x_star, y_star) in enumerate(zip(selected_sources['xcentroid'], selected_sources['ycentroid'])):
    x_star, y_star = int(x_star), int(y_star)
    # print(f"Star {i}: x_star = {x_star}, y_star = {y_star}")

    # Adjust star coordinates to match the original image
    x_star_global = x_star + (center_x - crop_size)  # Global x-coordinate
    y_star_global = y_star + (center_y - crop_size)  # Global y-coordinate

    # Ensure star is within the image bounds
    if (0 <= x_star_global < image_data.shape[1]) and (0 <= y_star_global < image_data.shape[0]):
        # Ensure we don't go out of bounds when extracting the sub-image
        x_start = max(0, x_star - 3)
        x_end = min(cropped_image_data.shape[1], x_star + 3)
        y_start = max(0, y_star - 3)
        y_end = min(cropped_image_data.shape[0], y_star + 3)

        # Check if the defined area is valid
        if x_end > x_start and y_end > y_start:
            # Create a meshgrid for the fitting
            x = np.arange(x_end - x_start)
            y = np.arange(y_end - y_start)
            x, y = np.meshgrid(x, y)

            # Extract the sub-image
            sub_image = cropped_image_data[y_start:y_end, x_start:x_end]

            # Create an initial guess for the Gaussian fit
            initial_guess = (np.max(sub_image), (x_end - x_start) // 2, (y_end - y_start) // 2, 3, 3, 0, 0)

            try:
                # Fit the 2D Gaussian
                popt, _ = curve_fit(gaussian_2d, (x.ravel(), y.ravel()), sub_image.ravel(), p0=initial_guess)

                # Extract the fitted parameters
                sigma_x, sigma_y = popt[3], popt[4]
                fwhm_x = 2.355 * sigma_x  # 2 * sqrt(2 * ln(2)) * sigma
                fwhm_y = 2.355 * sigma_y  # 2 * sqrt(2 * ln(2)) * sigma
                # print(f"Star {i}: FWHM_x = {fwhm_x:.2f}, FWHM_y = {fwhm_y:.2f}")

                # Append the FWHM values to the lists
                fwhms_x.append(fwhm_x)
                fwhms_y.append(fwhm_y)

            except Exception as e:
                print(f"Error fitting star {i}: {e}")

# Plot the selected stars with the fitted Gaussian
plt.figure()
vmin, vmax = np.percentile(image_data, [5, 95])
plt.imshow(image_data, cmap='hot', origin='lower', vmin=vmin, vmax=vmax)
positions = np.transpose((selected_sources['xcentroid'] + (center_x - crop_size),
                          selected_sources['ycentroid'] + (center_y - crop_size)))
apertures = CircularAperture(positions, r=5.)
apertures.plot(color='blue', lw=1.5, alpha=0.5)

# Calculate the average FWHM values
average_fwhm_x = np.median(fwhms_x)
average_fwhm_y = np.median(fwhms_y)
print(f'Average FWHM_x: {average_fwhm_x:.2f} pixels')
print(f'Average FWHM_y: {average_fwhm_y:.2f} pixels')
FWHM = (average_fwhm_x + average_fwhm_y) / 2
print(f'Final FWHM: {FWHM:.2f} pixels')
plt.title(f'Measured FWHM: {FWHM:.2f} pixels')
plt.show()
