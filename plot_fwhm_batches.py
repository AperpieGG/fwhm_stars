#! /usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
from plotting_tools import plot_images


# Function to load and process FWHM data from JSON file for each region
# Function to load and process FWHM data from JSON file for each region
def load_fwhm_data_per_region(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)  # Load JSON, expecting it to be a list

    # Initialize empty lists and dictionaries
    bjds, airmass = [], []
    fwhm_x = {f'Region_{i}{j}': [] for i in range(1, 4) for j in range(1, 4)}
    fwhm_y = {f'Region_{i}{j}': [] for i in range(1, 4) for j in range(1, 4)}

    # Process each entry in the list
    for entry in data:
        bjds.append(entry["BJD"])
        airmass.append(entry["Airmass"])

        for region in entry["Regions"]:
            region_name = region["Region"]
            fwhm_x[region_name].append(region["FWHM_X"])
            fwhm_y[region_name].append(region["FWHM_Y"])

    return bjds, airmass, fwhm_x, fwhm_y


# Load data from JSON files for CMOS and CCD
bjds1, airmass1, fwhm_x_cmos, fwhm_y_cmos = load_fwhm_data_per_region('fwhm_results_CMOS.json')
bjds2, airmass2, fwhm_x_ccd, fwhm_y_ccd = load_fwhm_data_per_region('fwhm_results_CCD.json')

# Plot FWHM_X and FWHM_Y for each region
fig, axs = plt.subplots(3, 3, figsize=(15, 15), sharex=True, sharey=True)
for i in range(1, 4):
    for j in range(1, 4):
        region_name = f'Region_{i}{j}'
        ax = axs[i - 1, j - 1]

        # Plot FWHM_X for CMOS and CCD
        avg_fwhm_x_cmos = np.mean(fwhm_x_cmos[region_name])
        avg_fwhm_y_cmos = np.mean(fwhm_y_cmos[region_name])
        avg_fwhm_x_ccd = np.mean(fwhm_x_ccd[region_name])
        avg_fwhm_y_ccd = np.mean(fwhm_y_ccd[region_name])
        ax.plot(bjds1, fwhm_x_cmos[region_name], 'o', label=f'FWHM_X CMOS, Avg= {avg_fwhm_x_cmos} microns', color='red', alpha=0.5)
        ax.plot(bjds2, fwhm_x_ccd[region_name], 'o', label=f'FWHM_X CCD, Avg= {avg_fwhm_x_ccd} microns', color='blue', alpha=0.5)

        # Plot FWHM_Y for CMOS and CCD
        ax.plot(bjds1, fwhm_y_cmos[region_name], 's', label=f'FWHM_Y CMOS, Avg= {avg_fwhm_y_cmos} microns', color='green', alpha=0.5)
        ax.plot(bjds2, fwhm_y_ccd[region_name], 's', label=f'FWHM_Y CCD, Avg= {avg_fwhm_y_ccd} microns', color='purple', alpha=0.5)

        # Set region title and labels
        ax.set_title(f'{region_name}')
        ax.set_xlabel("BJD")
        ax.set_ylabel("FWHM (pixels)")

        # Airmass on top x-axis
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        interpolated_airmass = np.interp(ax.get_xticks(), bjds1, airmass1)
        ax2.set_xticks(ax.get_xticks())
        ax2.set_xticklabels([f'{a:.2f}' for a in interpolated_airmass], rotation=45, ha='right')
        ax2.set_xlabel('Airmass')

# Show the legend for each subplot
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
plt.tight_layout()
plt.show()