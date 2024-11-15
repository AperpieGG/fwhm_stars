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

# Check if the PNG file already exists
output_png = "fwhm_positions.json"
if os.path.exists(output_png):
    print(f"{output_png} already exists. Skipping script execution.")
    exit()


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


def calculate_region_positions(h, w):
    region_positions = {}
    h_step, w_step = h // 3, w // 3  # Divide image into 3x3 grid

    for i in range(3):  # Loop through rows
        for j in range(3):  # Loop through columns
            region_name = f"Region_{i + 1}{j + 1}"
            x_start, x_end = j * w_step, (j + 1) * w_step
            y_start, y_end = i * h_step, (i + 1) * h_step
            region_positions[region_name] = {
                "position": {
                    "x_start": x_start,
                    "x_end": x_end,
                    "y_start": y_start,
                    "y_end": y_end
                }
            }
    return region_positions


# Process each FITS file
directory = os.getcwd()
times, fwhm_values, airmass_values, ratio_values = [], [], [], []
filenames = sorted([
    f for f in os.listdir(directory)
    if f.endswith('.fits') and not any(word in f.lower() for word in ["evening", "morning", "flat", "bias", "dark",
                                                                      "catalog", "phot", "catalog_input"])])

# Initialize cumulative FWHM results for averaging FWHM across images
cumulative_fwhm_results = {f"Region_{i + 1}{j + 1}": [] for i in range(3) for j in range(3)}
region_positions = {}

# Process every fifth filename
for i, filename in enumerate(filenames[::5]):
    full_path = os.path.join(directory, filename)
    print(f"Processing file {i + 1}: {filename}")
    with fits.open(full_path, mode='update') as hdul:
        header, image_data = hdul[0].header, hdul[0].data
        h, w = image_data.shape  # Get image dimensions
        region_positions = calculate_region_positions(h, w)  # Calculate region positions

        # [The rest of your existing processing code remains unchanged...]

        fwhm_results = split_image_and_calculate_fwhm(image_data, pixel_size)

        print(f"FWHM Results for {filename}:")
        for region, results in fwhm_results.items():
            print(f"{region} - FWHM: {results['FWHM']:.2f}, Ratio: {results['Ratio']:.2f}")
            # Append FWHM for this region to cumulative results
            cumulative_fwhm_results[region].append(results['FWHM'])

# After processing all images, calculate average FWHM for each region and prepare output
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

results_to_save = {
    "central_region": {
        "name": central_region,
        "average_fwhm": None,
        "position": None,  # Initialize position for the central region
        "similar_regions": []
    }
}

if central_region in average_fwhm_per_region:
    central_avg_fwhm = average_fwhm_per_region[central_region]
    lower_bound = central_avg_fwhm - threshold
    upper_bound = central_avg_fwhm + threshold
    print(
        f"\nAverage FWHM for {central_region} is {central_avg_fwhm:.2f} pixels with threshold range: {lower_bound:.2f} to {upper_bound:.2f}")

    results_to_save["central_region"]["average_fwhm"] = central_avg_fwhm

    # Store the position of the central region
    if central_region in region_positions:
        results_to_save["central_region"]["position"] = region_positions[central_region]["position"]

    for region, avg_fwhm in average_fwhm_per_region.items():
        if region != central_region and lower_bound <= avg_fwhm <= upper_bound:
            region_position = region_positions[region]["position"]
            results_to_save["central_region"]["similar_regions"].append({
                "name": region,
                "average_fwhm": avg_fwhm,
                "position": region_position  # Use the position calculated earlier
            })

    if results_to_save["central_region"]["similar_regions"]:
        print(f"Regions similar to {central_region} within the threshold: {', '.join([r['name'] for r in results_to_save['central_region']['similar_regions']])}")
    else:
        print(f"No regions found similar to {central_region} within the threshold.")
else:
    print(f"{central_region} not found in the results.")

# Save results to a JSON file
output_filename = "fwhm_positions.json"
with open(output_filename, 'w') as json_file:
    json.dump(results_to_save, json_file, indent=4)

print(f"Results saved to {output_filename}.")