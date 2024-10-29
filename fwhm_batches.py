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
parser.add_argument('--size', type=int, default=11, help='CMOS = 11, CCD = 13.5')
args = parser.parse_args()
pixel_size = args.size


def save_results_json(bjd, airmass, fwhm_values, ratio_values, fwhm_results, pixel_size):
    result_data = {
        "BJD": bjd,
        "Airmass": airmass,
        "FWHM_values": fwhm_values,
        "Ratio_values": ratio_values,
        "Pixel_size": pixel_size,
        "Regions": fwhm_results
    }
    with open("fwhm_results.json", "w") as json_file:
        json.dump(result_data, json_file, indent=4)


def plot_image_with_sources(image_data, fwhm_results):
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    h, w = image_data.shape
    regions = {
        "Region_11": image_data[0:h // 2, 0:w // 2],
        "Region_12": image_data[0:h // 2, w // 2:w],
        "Region_21": image_data[h // 2:h, 0:w // 2],
        "Region_22": image_data[h // 2:h, w // 2:w],
    }
    for i, (region_name, region_data) in enumerate(regions.items()):
        ax_row = i // 2
        ax_col = i % 2
        ax[ax_row, ax_col].imshow(region_data, cmap='gray', origin='lower')
        ax[ax_row, ax_col].set_title(region_name)

        if region_name in fwhm_results:
            for source in fwhm_results[region_name]["sources"]:
                x_star, y_star = source['xcentroid'], source['ycentroid']
                aperture = CircularAperture((x_star, y_star), r=5)
                aperture.plot(color='red', lw=1, ax=ax[ax_row, ax_col])

            avg_fwhm = fwhm_results[region_name]["FWHM"]
            ax[ax_row, ax_col].text(0.05, 0.95, f'Avg FWHM: {avg_fwhm:.2f} px',
                                    transform=ax[ax_row, ax_col].transAxes,
                                    color='white', fontsize=10, ha='left')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def calculate_fwhm(image_data, pixel_size):
    mean, median, std = np.mean(image_data), np.median(image_data), mad_std(image_data)
    daofind = DAOStarFinder(fwhm=4, threshold=5. * std, brightest=50)
    selected_sources = daofind(image_data - median)
    if selected_sources is None:
        print("No sources found.")
        return None, None

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
        return (avg_fwhm_x + avg_fwhm_y) * pixel_size / 2, ratio, sources
    return None, None, None


def split_image_and_calculate_fwhm(image_data, pixel_size):
    h, w = image_data.shape
    h_step, w_step = h // 2, w // 2
    fwhm_results = {}
    for i in range(2):
        for j in range(2):
            region_name = f"Region_{i + 1}{j + 1}"
            x_start, x_end = j * w_step, (j + 1) * w_step
            y_start, y_end = i * h_step, (i + 1) * h_step
            region_data = image_data[y_start:y_end, x_start:x_end]
            fwhm, ratio, sources = calculate_fwhm(region_data, pixel_size)
            if fwhm and ratio:
                fwhm_results[region_name] = {"FWHM": fwhm, "Ratio": ratio, "sources": sources}
            else:
                print(f"FWHM calculation failed for {region_name}")
    return fwhm_results


directory = os.getcwd()
times, fwhm_values, airmass_values, ratio_values = [], [], [], []
filenames = sorted([
    f for f in os.listdir(directory)
    if f.endswith('.fits') and not any(word in f.lower() for word in ["evening", "morning", "flat", "bias", "dark",
                                                                      "catalog", "phot", "catalog_input"])
])

for i, filename in enumerate(filenames):
    full_path = os.path.join(directory, filename)
    print(f"Processing file {i + 1}: {filename}")
    with fits.open(full_path, mode='update') as hdul:
        header, image_data = hdul[0].header, hdul[0].data
        exptime = float(header.get('EXPTIME', 10))
        if 'BJD' not in header:
            time_isot = Time(header['DATE-OBS'], format='isot', scale='utc', location=get_location())
            time_jd = Time(time_isot.jd, format='jd', scale='utc', location=get_location())
            time_jd += (exptime / 2.) * u.second
            if 'TELRAD' in header and 'TELDECD' in header:
                ltt_bary, _ = get_light_travel_times(header['TELRAD'], header['TELDECD'], time_jd)
                header['BJD'] = (time_jd.tdb + ltt_bary).value

        if 'AIRMASS' not in header:
            altitude = header.get('ALTITUDE', 45)
            header['AIRMASS'] = calculate_airmass(altitude)

        fwhm_results = split_image_and_calculate_fwhm(image_data, pixel_size)
        print(f"FWHM Results for {filename}:")
        for region, results in fwhm_results.items():
            print(f"{region} - FWHM: {results['FWHM']:.2f}, Ratio: {results['Ratio']:.2f}")

        save_results_json(header['BJD'], header['AIRMASS'], fwhm_values, ratio_values, fwhm_results, pixel_size)
        if i == len(filenames) - 1:
            plot_image_with_sources(image_data, fwhm_results)
