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
output_png = "fwhm_batches_images.png"
if os.path.exists(output_png):
    print(f"{output_png} already exists. Skipping script execution.")
    exit()


warnings.filterwarnings('ignore', category=UserWarning)
parser = argparse.ArgumentParser(description='Measure FWHM from a FITS image.')
parser.add_argument('--size', type=float, default=11, help='CMOS = 11, CCD = 13.5')
parser.add_argument('--cam', type=str, default='CMOS', help='CMOS, CCD, or both')
args = parser.parse_args()
pixel_size = args.size
cam = args.cam

# Initialize a global dictionary to store results for all images
all_results = []


def save_results_json(bjd, airmass, pixel_size, fwhm_results):
    # Prepare results to include only the necessary fields for each region
    regions_data = []
    for region_name, results in fwhm_results.items():
        regions_data.append({
            "Region": region_name,
            "FWHM": results["FWHM"],
            "Ratio": results["Ratio"],
            "FWHM_X": results["FWHM_X"],
            "FWHM_Y": results["FWHM_Y"]
        })

    result_data = {
        "BJD": bjd,
        "Airmass": airmass,
        "Pixel_size": pixel_size,
        "Regions": regions_data  # Save regions data as a list
    }

    all_results.append(result_data)  # Append result data for this image


# After processing all images, save all results to a single JSON file

def save_all_results_to_json():
    with open(f"fwhm_results_{cam}.json", "w") as json_file:
        json.dump(all_results, json_file, indent=4)
    print(f"All results saved to fwhm_results_{cam}.json")


def plot_full_image_with_sources(image_data, fwhm_results, cumulative_fwhm_results):
    # Calculate overall average FWHM for each region from cumulative results
    avg_fwhm_per_region = {
        region: np.mean(values) if values else 0
        for region, values in cumulative_fwhm_results.items()
    }

    # Plotting code as before
    vmin, vmax = np.percentile(image_data, [5, 95])
    plt.figure(figsize=(10, 10))
    plt.imshow(image_data, cmap='hot', origin='lower', vmin=vmin, vmax=vmax)

    h, w = image_data.shape
    h_step, w_step = h // 3, w // 3
    regions = {
        f"Region_{i+1}{j+1}": (i * h_step, (i + 1) * h_step, j * w_step, (j + 1) * w_step)
        for i in range(3) for j in range(3)
    }
    positions = []  # Store positions of sources for lines

    for region_name, (y_start, y_end, x_start, x_end) in regions.items():
        if region_name in fwhm_results:
            avg_fwhm = avg_fwhm_per_region.get(region_name, 0)
            region_sources = fwhm_results[region_name].get('sources', [])
            for source in region_sources:  # Corrected line, no additional ['sources'] index
                x_pos = source['xcentroid'] + x_start
                y_pos = source['ycentroid'] + y_start
                positions.append((x_pos, y_pos))  # Store position for lines
                aperture = CircularAperture((x_pos, y_pos), r=5.)
                aperture.plot(color='blue', lw=1.5, alpha=0.5)

            plt.text(x_start + 10, y_start + 20, f'{region_name} Avg FWHM: {avg_fwhm:.2f} μm',
                     color='white', fontsize=10, bbox=dict(facecolor='black', alpha=0.7))

    # Draw boundary lines for regions
    for i in range(1, 3):
        plt.axhline(y=i * h_step, color='black', linestyle='-', lw=1)  # Horizontal lines
        plt.axvline(x=i * w_step, color='black', linestyle='-', lw=1)  # Vertical lines

    # Set the axes to match the image dimensions
    plt.xlim(0, w)
    plt.ylim(0, h)

    # Calculate the overall average FWHM
    fwhm_values = [results['FWHM'] for results in fwhm_results.values() if 'FWHM' in results]
    average_fwhm = np.median(fwhm_values) if fwhm_values else 0

    plt.title(f"Avg Regions: FWHM = {average_fwhm:.2f} microns")
    # Save the plot to a file
    plt.savefig("fwhm_batches_images.png")


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
    if (f.endswith('.fits') or f.endswith('.fits.bz2')) and not any(word in f.lower() for word in
                                                                    ["evening", "morning", "flat", "bias", "dark",
                                                                     "catalog", "phot", "catalog_input"])])

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

        # Save results for the current image
        save_results_json(header['BJD'], header['AIRMASS'], pixel_size, fwhm_results)

# After processing all images, save to a single JSON file
save_all_results_to_json()

# Optionally, plot the last image with average FWHM for each region
if len(filenames) > 0:
    plot_full_image_with_sources(image_data, fwhm_results, cumulative_fwhm_results)