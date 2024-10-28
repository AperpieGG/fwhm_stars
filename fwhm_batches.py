#! /usr/bin/env python
import json
import os
import numpy as np
from astropy.io import fits
from photutils.aperture import CircularAperture
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from astropy.stats import mad_std
from photutils.detection import DAOStarFinder
from astropy.time import Time
import astropy.units as u
from plotting_tools import get_location, get_light_travel_times, plot_images, gaussian_2d, calculate_airmass
import argparse
import warnings
plot_images()

warnings.filterwarnings('ignore', category=UserWarning)
parser = argparse.ArgumentParser(description='Measure FWHM from a FITS image.')
parser.add_argument('--size', type=int, default=800, help='CMOS = 11, CCD = 13.5')
args = parser.parse_args()
pixel_size = args.size


def plot_image_with_sources(image_data, fwhm_results):
    """
    Plot the image with the identified sources and FWHM results.

    Parameters:
        image_data (numpy.ndarray): The 2D array of image data.
        fwhm_results (dict): Dictionary containing FWHM results for each region.
    """
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    # Split the image into quadrants
    h, w = image_data.shape
    regions = {
        "Region_11": image_data[0:h//2, 0:w//2],
        "Region_12": image_data[0:h//2, w//2:w],
        "Region_21": image_data[h//2:h, 0:w//2],
        "Region_22": image_data[h//2:h, w//2:w],
    }

    # Plot each region
    for i, (region_name, region_data) in enumerate(regions.items()):
        ax_row = i // 2
        ax_col = i % 2
        ax[ax_row, ax_col].imshow(region_data, cmap='gray', origin='lower')
        ax[ax_row, ax_col].set_title(region_name)

        # Show apertures for sources found
        if region_name in fwhm_results:
            for source in fwhm_results[region_name]:
                x_star, y_star = source['xcentroid'], source['ycentroid']
                aperture = CircularAperture((x_star, y_star), r=5)  # Radius of 5 pixels
                aperture.plot(color='red', lw=1, ax=ax[ax_row, ax_col])

                # Display the average FWHM for the region
                avg_fwhm = fwhm_results[region_name]["FWHM"]
                ax[ax_row, ax_col].text(0.05, 0.95, f'Avg FWHM: {avg_fwhm:.2f} px',
                                         transform=ax[ax_row, ax_col].transAxes,
                                         color='white', fontsize=10, ha='left')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout
    plt.show()


# Function to calculate FWHM for a given region
def calculate_fwhm(image_data, pixel_size):
    # Estimate background noise level
    mean, median, std = np.mean(image_data), np.median(image_data), mad_std(image_data)
    daofind = DAOStarFinder(fwhm=4, threshold=5. * std, brightest=50)
    selected_sources = daofind(image_data - median)

    if selected_sources is None:
        print("No sources found.")
        return None, None

    fwhms_x, fwhms_y = [], []
    for x_star, y_star in zip(selected_sources['xcentroid'], selected_sources['ycentroid']):
        x_star, y_star = int(x_star), int(y_star)
        x_start = max(0, x_star - 3)
        x_end = min(image_data.shape[1], x_star + 3)
        y_start = max(0, y_star - 3)
        y_end = min(image_data.shape[0], y_star + 3)

        if x_end > x_start and y_end > y_start:
            x, y = np.meshgrid(np.arange(x_end - x_start), np.arange(y_end - y_start))
            sub_image = image_data[y_start:y_end, x_start:x_end]
            initial_guess = (np.max(sub_image), (x_end - x_start) // 2, (y_end - y_start) // 2, 3, 3, 0, 0)

            try:
                popt, _ = curve_fit(gaussian_2d, (x.ravel(), y.ravel()), sub_image.ravel(), p0=initial_guess)
                sigma_x, sigma_y = popt[3], popt[4]
                fwhms_x.append(2.355 * sigma_x)
                fwhms_y.append(2.355 * sigma_y)
            except Exception as e:
                print(f"Error fitting source: {e}")

    if fwhms_x and fwhms_y:
        avg_fwhm_x, avg_fwhm_y = np.median(fwhms_x), np.median(fwhms_y)
        ratio = np.median([fwhms_x[i] / fwhms_y[i] for i in range(len(fwhms_x))])
        return (avg_fwhm_x + avg_fwhm_y) * pixel_size / 2, ratio
    return None, None


# Function to split image into 9 regions and calculate FWHM for each
def split_image_and_calculate_fwhm(image_data, pixel_size):
    h, w = image_data.shape
    # do 2 x 2 grid
    h_step, w_step = h // 2, w // 2
    fwhm_results = {}

    for i in range(2):
        for j in range(2):
            region_name = f"Region_{i + 1}{j + 1}"
            x_start, x_end = j * w_step, (j + 1) * w_step
            y_start, y_end = i * h_step, (i + 1) * h_step
            region_data = image_data[y_start:y_end, x_start:x_end]

            # Calculate FWHM for the region
            fwhm, ratio = calculate_fwhm(region_data, pixel_size)
            if fwhm and ratio:
                fwhm_results[region_name] = {"FWHM": fwhm, "Ratio": ratio}
            else:
                print(f"FWHM calculation failed for {region_name}")

    return fwhm_results


# Process each FITS file in the directory
directory = os.getcwd()
times, fwhm_values, airmass_values, ratio_values = [], [], [], []

filenames = [
    f for f in os.listdir(directory)
    if f.endswith('.fits') and not any(word in f.lower() for word in ["evening", "morning",
                                                                      "flat", "bias", "dark",
                                                                      "catalog", "phot", "catalog_input"])
]
sorted_filenames = sorted(filenames)

# Process each file in sorted order
for i, filename in enumerate(sorted_filenames):
    full_path = os.path.join(directory, filename)
    print(f"Processing file {i + 1}: {filename}")
    with fits.open(full_path, mode='update') as hdul:
        header = hdul[0].header
        image_data = hdul[0].data

        if 'BJD' not in header:
            exptime = float(header.get('EXPTIME', header.get('EXPOSURE', 10)))
            time_isot = Time(header['DATE-OBS'], format='isot', scale='utc', location=get_location())
            time_jd = Time(time_isot.jd, format='jd', scale='utc', location=get_location())
            time_jd += (exptime / 2.) * u.second

            ra = header.get('TELRAD', header.get('RA'))
            dec = header.get('TELDECD', header.get('DEC'))
            if ra and dec:
                ltt_bary, _ = get_light_travel_times(ra, dec, time_jd)
                time_bary = time_jd.tdb + ltt_bary
                header['BJD'] = time_bary.value
                print(f"Calculated BJD for {filename}: {time_bary.value}")
            else:
                print(f"No RA/DEC information for {filename}. Skipping BJD calculation.")

        if 'AIRMASS' not in header:
            altitude = header.get('ALTITUDE', 45)
            header['AIRMASS'] = calculate_airmass(altitude)
            print(f"Calculated airmass for {filename}: {header['AIRMASS']}")
        else:
            print(f"Airmass found in header for {filename}: {header['AIRMASS']}")

        fwhm_results = split_image_and_calculate_fwhm(image_data, pixel_size)

        # Plot the first or last image after processing
        if i == 0:  # Change to just the last or first
            plot_image_with_sources(image_data, fwhm_results)

        # Print FWHM results for each region
        print(f"FWHM Results for {filename}:")
        for region, results in fwhm_results.items():
            print(f"{region} - FWHM: {results['FWHM']:.2f}, Ratio: {results['Ratio']:.2f}")
            print()

# Sort by BJD for plotting
sorted_data = sorted(zip(times, fwhm_values, airmass_values, ratio_values))
times, fwhm_values, airmass_values, ratio_values = zip(*sorted_data)




