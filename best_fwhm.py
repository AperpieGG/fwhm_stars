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
parser.add_argument('--size', type=float, default=11, help='CMOS = 11, CCD = 13.5')
args = parser.parse_args()
pixel_size = args.size

# Initialize a global dictionary to store results for all images
all_results = []


def calculate_fwhm(image_data, pixel_size):
    mean, median, std = np.mean(image_data), np.median(image_data), mad_std(image_data)
    daofind = DAOStarFinder(fwhm=4, threshold=5. * std, brightest=100)
    selected_sources = daofind(image_data - median)
    if selected_sources is None:
        print("No sources found.")
        return None, None, None

    fwhms_x, fwhms_y, sources = [], [], []
    for x_star, y_star in zip(selected_sources['xcentroid'], selected_sources['ycentroid']):
        x_star, y_star = int(x_star), int(y_star)
        x_start, x_end = max(0, x_star - 3), min(image_data.shape[1], x_star + 3)
        y_start, y_end = max(0, y_star - 3), min(image_data.shape[0], y_star + 3)
        if x_end > x_start and y_end > y_start:
            x, y = np.meshgrid(np.arange(x_end - x_start), np.arange(y_end - y_start))
            sub_image = image_data[y_start:y_end, x_start:x_end]
            initial_guess = (np.max(sub_image), (x_end - x_start) // 2, (y_end - y_start) // 2, 3, 3, 0, 0)
            try:
                popt, _ = curve_fit(gaussian_2d, (x.ravel(), y.ravel()), sub_image.ravel(), p0=initial_guess)
                sigma_x, sigma_y = popt[3], popt[4]
                fwhms_x.append(2.355 * sigma_x)
                fwhms_y.append(2.355 * sigma_y)
                # Store coordinates in a dictionary
                sources.append({'xcentroid': x_star, 'ycentroid': y_star})
            except Exception as e:
                print(f"Error fitting source: {e}")
    if fwhms_x and fwhms_y:
        avg_fwhm_x, avg_fwhm_y = np.median(fwhms_x), np.median(fwhms_y)
        ratio = np.median([fwhms_x[i] / fwhms_y[i] for i in range(len(fwhms_x))])
        return (avg_fwhm_x + avg_fwhm_y) * pixel_size / 2, ratio, avg_fwhm_x * pixel_size, avg_fwhm_y * pixel_size, sources
    return None, None, None, None, None


def split_image_and_calculate_fwhm(image_data, pixel_size):
    h, w = image_data.shape
    h_step, w_step = h // 3, w // 3  # Divide image into 3x3 grid
    fwhm_results = {}

    for i in range(3):  # Loop through rows
        for j in range(3):  # Loop through columns
            region_name = f"Region_{i + 1}{j + 1}"
            x_start, x_end = j * w_step, (j + 1) * w_step
            y_start, y_end = i * h_step, (i + 1) * h_step
            region_data = image_data[y_start:y_end, x_start:x_end]
            fwhm, ratio, fwhm_x, fwhm_y, sources = calculate_fwhm(region_data, pixel_size)
            if fwhm and ratio:
                fwhm_results[region_name] = {"FWHM": fwhm, "Ratio": ratio, "sources": sources, "FWHM_X": fwhm_x, "FWHM_Y": fwhm_y}
            else:
                print(f"FWHM calculation failed for {region_name}")
    return fwhm_results


# Process each FITS file
directory = os.getcwd()
times, fwhm_values, airmass_values, ratio_values = [], [], [], []
filenames = sorted([
    f for f in os.listdir(directory)
    if f.endswith('.fits') and not any(word in f.lower() for word in ["evening", "morning", "flat", "bias", "dark",
                                                                      "catalog", "phot", "catalog_input"])])[:10]

# Initialize cumulative FWHM results for averaging FWHM across images
cumulative_fwhm_results = {f"Region_{i + 1}{j + 1}": [] for i in range(3) for j in range(3)}

for i, filename in enumerate(filenames):
    full_path = os.path.join(directory, filename)
    print(f"Processing file {i + 1}: {filename}")
    with fits.open(full_path, mode='update') as hdul:
        header, image_data = hdul[0].header, hdul[0].data
        exptime = float(header.get('EXPTIME', 10))

        # Calculate BJD if not present
        if 'BJD' not in header:
            time_isot = Time(header['DATE-OBS'], format='isot', scale='utc', location=get_location())
            time_jd = Time(time_isot.jd, format='jd', scale='utc', location=get_location())
            time_jd += (exptime / 2.) * u.second
            if 'TELRAD' in header and 'TELDECD' in header:
                ltt_bary, _ = get_light_travel_times(header['TELRAD'], header['TELDECD'], time_jd)
                header['BJD'] = (time_jd.tdb + ltt_bary).value

        # Calculate Airmass if not present
        if 'AIRMASS' not in header:
            altitude = header.get('ALTITUDE', 45)
            header['AIRMASS'] = calculate_airmass(altitude)

        fwhm_results = split_image_and_calculate_fwhm(image_data, pixel_size)
        print(f"FWHM Results for {filename}:")
        for region, results in fwhm_results.items():
            print(f"{region} - FWHM: {results['FWHM']:.2f}, Ratio: {results['Ratio']:.2f}")
            # Append FWHM for this region to cumulative results
            cumulative_fwhm_results[region].append(results['FWHM'])

            # After processing all images, calculate and print the average FWHM for each region
            print("\nAverage FWHM for each region across all images:")
            average_fwhm_per_region = {}
            for region, fwhm_list in cumulative_fwhm_results.items():
                if fwhm_list:  # Check if the list is not empty
                    average_fwhm = np.mean(fwhm_list)  # Calculate the average
                    average_fwhm_per_region[region] = average_fwhm  # Store the average FWHM for later comparison
                    print(f"{region}: Average FWHM = {average_fwhm:.2f} pixels")
                else:
                    print(f"{region}: No FWHM values found.")

            # Check central region (Region 22) and find similar regions
            central_region = "Region_22"
            threshold = 0.5  # Microns

            if central_region in average_fwhm_per_region:
                central_avg_fwhm = average_fwhm_per_region[central_region]
                lower_bound = central_avg_fwhm - threshold
                upper_bound = central_avg_fwhm + threshold
                print(
                    f"\nAverage FWHM for {central_region} is {central_avg_fwhm:.2f} pixels with threshold range: {lower_bound:.2f} to {upper_bound:.2f}")

                similar_regions = []
                for region, avg_fwhm in average_fwhm_per_region.items():
                    if region != central_region and lower_bound <= avg_fwhm <= upper_bound:
                        similar_regions.append(region)

                if similar_regions:
                    print(f"Regions similar to {central_region} within the threshold: {', '.join(similar_regions)}")
                else:
                    print(f"No regions found similar to {central_region} within the threshold.")
            else:
                print(f"{central_region} not found in the results.")