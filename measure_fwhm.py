#! /usr/bin/env python
import os
import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from astropy.stats import mad_std
from photutils.detection import DAOStarFinder
from astropy.time import Time
import astropy.units as u
from plotting_tools import get_location, get_light_travel_times, plot_images
import warnings
import json
import argparse

plot_images()

warnings.filterwarnings('ignore', category=UserWarning)
parser = argparse.ArgumentParser(description='Measure FWHM from a FITS image.')
parser.add_argument('--crop_size', type=int, default=800, help='CMOS = 800, CCD = 652')
parser.add_argument('--size', type=float, default=11, help='pixel size to convert to microns')
args = parser.parse_args()
crop_size = args.crop_size
size = args.size


# Function to fit a 2D Gaussian
def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
    g = offset + amplitude * np.exp(-(a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2)))
    return g.ravel()


# Get airmass from altitude
def calculate_airmass(altitude):
    return 1 / np.cos(np.radians(90 - altitude))


# Function to calculate FWHM for stars in an image
def calculate_fwhm(image_data, crop_size):
    # Define the central region
    center_x, center_y = image_data.shape[1] // 2, image_data.shape[0] // 2
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
                initial_guess = (np.max(sub_image),
                                 (x_end - x_start) // 2, (y_end - y_start) // 2, 3, 3, 0, 0)

                try:
                    popt, _ = curve_fit(gaussian_2d,
                                        (x.ravel(), y.ravel()),  # Use the meshgrid coordinates
                                        sub_image.ravel(),  # Flatten the sub-image
                                        p0=initial_guess)

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

    # Return the average FWHM for the stars
    if fwhms_x and fwhms_y:
        average_fwhms_x = np.median(fwhms_x)
        average_fwhms_y = np.median(fwhms_y)
        ratio_each = [fwhms_x[i] / fwhms_y[i] for i in range(len(fwhms_x))]
        ratio = np.median(ratio_each)
        return (average_fwhms_x + average_fwhms_y) / 2, ratio


# Process each FITS file in the directory
directory = os.getcwd()
times, fwhm_values, airmass_values, ratio_values = [], [], [], []

# List all FITS filenames and exclude those containing specific words
filenames = [
    f for f in os.listdir(directory)
    if f.endswith('.fits') and not any(word in f.lower() for word in ["evening", "morning",
                                                                      "flat", "bias", "dark",
                                                                      "catalog", "phot", "catalog_input"])
]

# Sort filenames alphabetically
sorted_filenames = sorted(filenames)

# Process each file in sorted order
for i, filename in enumerate(sorted_filenames):
    full_path = os.path.join(directory, filename)
    print(f"Processing file {i + 1}: {filename}")
    with fits.open(full_path, mode='update') as hdul:
        header = hdul[0].header
        image_data = hdul[0].data

        # Get BJD from header or calculate if missing
        if 'BJD' not in header:
            exptime = float(header.get('EXPTIME', header.get('EXPOSURE', 10)))  # Default exposure time if missing
            time_isot = Time(header['DATE-OBS'], format='isot', scale='utc', location=get_location())
            time_jd = Time(time_isot.jd, format='jd', scale='utc', location=get_location())
            time_jd += (exptime / 2.) * u.second

            # Check if TELRAD and TELDECD are in the header; if not, use RA and DEC
            ra = header.get('TELRAD', header.get('RA'))
            dec = header.get('TELDECD', header.get('DEC'))
            if ra and dec:
                ltt_bary, _ = get_light_travel_times(ra, dec, time_jd)
                time_bary = time_jd.tdb + ltt_bary
                header['BJD'] = time_bary.value
                print(f"Calculated BJD for {filename}: {time_bary.value}")
            else:
                print(f"No RA/DEC information for {filename}. Skipping BJD calculation.")

        else:
            print(f"BJD found in header for {filename}: {header['BJD']}")

        # Get airmass from header or calculate if missing
        if 'AIRMASS' not in header:
            altitude = header.get('ALTITUDE', 45)  # Default altitude if missing
            header['AIRMASS'] = calculate_airmass(altitude)
            print(f"Calculated airmass for {filename}: {header['AIRMASS']}")
        else:
            print(f"Airmass found in header for {filename}: {header['AIRMASS']}")

        # Calculate and store FWHM
        fwhm, ratio = calculate_fwhm(image_data, crop_size)
        if fwhm and ratio:
            times.append(header['BJD'])
            fwhm_values.append(fwhm)
            airmass_values.append(header['AIRMASS'])
            ratio_values.append(ratio)
            print(f"Calculated FWHM for {filename}: {fwhm:.2f} pixels, ratio: {ratio:.2f}\n")
        else:
            print(f"FWHM calculation failed for {filename}\n")

# Sort by BJD for plotting
sorted_data = sorted(zip(times, fwhm_values, airmass_values, ratio_values))
times, fwhm_values, airmass_values, ratio_values = zip(*sorted_data)

# Plot FWHM vs Time and FWHM vs Airmass
print("Plotting results...")
fig, ax1 = plt.subplots()

ax1.plot(times, fwhm_values, 'o', label=f'FWHM (median={np.median(fwhm_values):.2f})', color='blue')
ax1.set_xlabel("BJD")
ax1.set_ylabel("FWHM (pixels)")

# Airmass on top x-axis
ax2 = ax1.twiny()
ax2.set_xlim(ax1.get_xlim())
ax2.set_xlabel('Airmass')
interpolated_airmass = np.interp(ax1.get_xticks(), times, airmass_values)
ax2.set_xticks(ax1.get_xticks())
ax2.set_xticklabels([f'{a:.2f}' for a in interpolated_airmass], rotation=45, ha='right')

ax1.legend()
plt.tight_layout()
plt.show()


fig, ax1 = plt.subplots()
fwhm_values = np.array(fwhm_values)
ax1.plot(times, fwhm_values * size, 'o', label=f'FWHM (median={np.median(fwhm_values * size):.2f})', color='blue')
ax1.set_xlabel("BJD")
ax1.set_ylabel("FWHM (microns)")

# Airmass on top x-axis
ax2 = ax1.twiny()
ax2.set_xlim(ax1.get_xlim())
ax2.set_xlabel('Airmass')
interpolated_airmass = np.interp(ax1.get_xticks(), times, airmass_values)
ax2.set_xticks(ax1.get_xticks())
ax2.set_xticklabels([f'{a:.2f}' for a in interpolated_airmass], rotation=45, ha='right')

ax1.legend()
plt.tight_layout()
plt.show()

# plot the ratio
fig, ax1 = plt.subplots()

ax1.plot(times, ratio_values, 'o', label=f'FWHM (median={np.median(ratio_values):.2f})', color='blue')
ax1.set_xlabel("BJD")
ax1.set_ylabel("FWHM X / FWHM Y")

# Airmass on top x-axis
ax2 = ax1.twiny()
ax2.set_xlim(ax1.get_xlim())
ax2.set_xlabel('Airmass')
interpolated_airmass = np.interp(ax1.get_xticks(), times, airmass_values)
ax2.set_xticks(ax1.get_xticks())
ax2.set_xticklabels([f'{a:.2f}' for a in interpolated_airmass], rotation=45, ha='right')

ax1.legend()
plt.tight_layout()
plt.show()


# Prepare data in dictionary format for JSON output
data_dict = {
    "results": [
        {
            "BJD": bjd,
            "Airmass": airmass,
            "FWHM": fwhm,
            "FWHM_microns": fwhm_microns,
            "Ratio": ratio

        }
        for bjd, airmass, fwhm, fwhm_microns, ratio in zip(times, airmass_values, fwhm_values,
                                                           fwhm_values * size, ratio_values)
    ]
}

# Write data to JSON file
with open("fwhm.json", "w") as json_file:
    json.dump(data_dict, json_file, indent=4)

print("Results saved to fwhm.json")
