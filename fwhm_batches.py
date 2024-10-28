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
from plotting_tools import get_location, get_light_travel_times, plot_images, gaussian_2d, calculate_airmass

plot_images()


# Function to calculate FWHM for a given region
def calculate_fwhm(image_data):
    # Estimate background noise level
    mean, median, std = np.mean(image_data), np.median(image_data), mad_std(image_data)
    daofind = DAOStarFinder(fwhm=4, threshold=5. * std, brightest=100)
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
        return (avg_fwhm_x + avg_fwhm_y) / 2, ratio
    return None, None


# Function to split image into 9 regions and calculate FWHM for each
def split_image_and_calculate_fwhm(image_data):
    h, w = image_data.shape
    h_step, w_step = h // 3, w // 3  # 3x3 grid
    fwhm_results = {}

    for i in range(3):
        for j in range(3):
            region_name = f"Region_{i * 3 + j + 1}"
            x_start, x_end = j * w_step, (j + 1) * w_step
            y_start, y_end = i * h_step, (i + 1) * h_step
            region_data = image_data[y_start:y_end, x_start:x_end]

            # Calculate FWHM for the region
            fwhm, ratio = calculate_fwhm(region_data)
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

        fwhm_results = split_image_and_calculate_fwhm(image_data)

        # Print FWHM results for each region
        print(f"FWHM Results for {filename}:")
        for region, results in fwhm_results.items():
            print(f"{region} - FWHM: {results['FWHM']:.2f}, Ratio: {results['Ratio']:.2f}")

# Sort by BJD for plotting
sorted_data = sorted(zip(times, fwhm_values, airmass_values, ratio_values))
times, fwhm_values, airmass_values, ratio_values = zip(*sorted_data)

# Plot FWHM vs Time and FWHM vs Airmass
print("Plotting results...")
fig, ax1 = plt.subplots()

ax1.plot([data[0] for data in sorted_data], [data[1] for data in sorted_data], 'o', label='FWHM')
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
