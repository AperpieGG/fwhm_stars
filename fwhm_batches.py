#!/usr/bin/env python
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

warnings.filterwarnings('ignore', category=UserWarning)

# Parse arguments for pixel size
parser = argparse.ArgumentParser(description='Measure FWHM from a FITS image.')
parser.add_argument('--size', type=float, default=11, help='CMOS = 11, CCD = 13.5')
args = parser.parse_args()
pixel_size = args.size

# Global dictionary to store cumulative FWHM results for averaging across all images
cumulative_fwhm_results = {f"Region_{i + 1}{j + 1}": [] for i in range(3) for j in range(3)}
all_results = []


def save_results_json(bjd, airmass, pixel_size, fwhm_results):
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
        "Regions": regions_data
    }
    all_results.append(result_data)


def save_all_results_to_json():
    with open("fwhm_results.json", "w") as json_file:
        json.dump(all_results, json_file, indent=4)
    print("All results saved to fwhm_results.json")


def plot_full_image_with_sources(image_data, fwhm_results, cumulative_fwhm_results):
    avg_fwhm_per_region = {
        region: np.mean(values) if values else 0
        for region, values in cumulative_fwhm_results.items()
    }

    vmin, vmax = np.percentile(image_data, [5, 95])
    plt.figure(figsize=(10, 10))
    plt.imshow(image_data, cmap='hot', origin='lower', vmin=vmin, vmax=vmax)

    h, w = image_data.shape
    h_step, w_step = h // 3, w // 3
    regions = {
        f"Region_{i + 1}{j + 1}": (i * h_step, (i + 1) * h_step, j * w_step, (j + 1) * w_step)
        for i in range(3) for j in range(3)
    }

    for region_name, (y_start, y_end, x_start, x_end) in regions.items():
        avg_fwhm = avg_fwhm_per_region.get(region_name, 0)
        plt.text(x_start + 10, y_start + 20, f'{region_name} Avg FWHM: {avg_fwhm:.2f}px',
                 color='white', fontsize=10, bbox=dict(facecolor='black', alpha=0.7))

    for i in range(1, 3):
        plt.axhline(y=i * h_step, color='black', linestyle='-', lw=1)
        plt.axvline(x=i * w_step, color='black', linestyle='-', lw=1)

    plt.xlim(0, w)
    plt.ylim(0, h)

    fwhm_values = [results['FWHM'] for results in fwhm_results.values() if 'FWHM' in results]
    average_fwhm = np.median(fwhm_values) if fwhm_values else 0

    plt.title(f"Avg Regions: FWHM = {average_fwhm:.2f} microns")
    plt.show()


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
    h_step, w_step = h // 3, w // 3
    fwhm_results = {}

    for i in range(3):
        for j in range(3):
            region_name = f"Region_{i + 1}{j + 1}"
            x_start, x_end = j * w_step, (j + 1) * w_step
            y_start, y_end = i * h_step, (i + 1) * h_step
            region_data = image_data[y_start:y_end, x_start:x_end]
            fwhm, ratio, fwhm_x, fwhm_y, sources = calculate_fwhm(region_data, pixel_size)
            if fwhm and ratio:
                fwhm_results[region_name] = {"FWHM": fwhm, "Ratio": ratio, "sources": sources, "FWHM_X": fwhm_x,
                                             "FWHM_Y": fwhm_y}
                cumulative_fwhm_results[region_name].append(fwhm)
            else:
                print(f"FWHM calculation failed for {region_name}")
    return fwhm_results


# Process each FITS file
directory = os.getcwd()
filenames = sorted([f for f in os.listdir(directory) if f.endswith('.fits') and not any(
    word in f.lower() for word in ["evening", "morning", "flat", "bias", "dark", "catalog", "phot", "catalog_input"])])[:10]

for i, filename in enumerate(filenames):
    full_path = os.path.join(directory, filename)
    print(f"Processing file {i + 1}: {filename}")
    with fits.open(full_path, mode='update') as hdul:
        header, image_data = hdul[0].header, hdul[0].data
        exptime = float(header.get('EXPTIME', 10))

        # Calculate BJD if not present in the header
        if 'BJD' not in header:
            time_isot = Time(header['DATE-OBS'], format='isot', scale='utc', location=get_location())
            time_jd = Time(time_isot.jd, format='jd', scale='utc', location=get_location())
            time_jd += (exptime / 2.) * u.second
            if 'TELRAD' in header and 'TELDECD' in header:
                ltt_bary, _ = get_light_travel_times(header['TELRAD'], header['TELDECD'], time_jd)
                header['BJD'] = (time_jd.tdb + ltt_bary).value

        hdul.flush()

        # Process and store results
        bjd = header['BJD']
        airmass = calculate_airmass(header.get('ALTITUDE', 45))
        fwhm_results = split_image_and_calculate_fwhm(image_data, pixel_size)
        save_results_json(bjd, airmass, pixel_size, fwhm_results)

        # Plot last image after all files are processed
        if i == len(filenames) - 1:
            plot_full_image_with_sources(image_data, fwhm_results, cumulative_fwhm_results)

# Save all JSON results after processing
save_all_results_to_json()