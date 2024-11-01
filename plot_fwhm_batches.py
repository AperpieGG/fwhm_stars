#! /usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
from plotting_tools import plot_images
import argparse

plot_images()


# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot FWHM data from JSON files.')
parser.add_argument('--cam', type=str, default='both', help='CMOS, CCD, or both')
args = parser.parse_args()


# Function to load and process FWHM data from JSON file for each region
# Function to load and process FWHM data from JSON file for each region
def load_fwhm_data_per_region(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)  # Load JSON, expecting it to be a list

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
# load them else return None
if args.cam == 'CMOS':
    bjds1, airmass1, fwhm_x_cmos, fwhm_y_cmos = load_fwhm_data_per_region('fwhm_results_CMOS.json')
    bjds2, airmass2, fwhm_x_ccd, fwhm_y_ccd = 1, 1, 1, 1
elif args.cam == 'CCD':
    bjds1, airmass1, fwhm_x_cmos, fwhm_y_cmos = 1, 1, 1, 1
    bjds2, airmass2, fwhm_x_ccd, fwhm_y_ccd = load_fwhm_data_per_region('fwhm_results_CCD.json')
else:
    bjds1, airmass1, fwhm_x_cmos, fwhm_y_cmos = load_fwhm_data_per_region('fwhm_results_CMOS.json')
    bjds2, airmass2, fwhm_x_ccd, fwhm_y_ccd = load_fwhm_data_per_region('fwhm_results_CCD.json')

# Plot FWHM_X and FWHM_Y for each region
fig, axs = plt.subplots(3, 3, figsize=(15, 15), sharex=True, sharey=True)
for i in range(1, 4):
    for j in range(1, 4):
        region_name = f'Region_{i}{j}'
        ax = axs[i - 1, j - 1]

        # Calculate averages
        avg_fwhm_x_cmos = np.mean(fwhm_x_cmos[region_name])
        avg_fwhm_y_cmos = np.mean(fwhm_y_cmos[region_name])
        avg_fwhm_x_ccd = np.mean(fwhm_x_ccd[region_name])
        avg_fwhm_y_ccd = np.mean(fwhm_y_ccd[region_name])

        # Plot FWHM_X and Y for CMOS
        if args.cam == 'CMOS':
            ax.plot(bjds1, fwhm_x_cmos[region_name], 'o', label=f'FWHM_X CMOS, Avg= {avg_fwhm_x_cmos:.2f} μm', color='red',
                    markerfacecolor='none', alpha=0.3)
            ax.plot(bjds1, fwhm_y_cmos[region_name], 'o', label=f'FWHM_Y CMOS, Avg= {avg_fwhm_y_cmos:.2f} μm', color='peru',
                    markerfacecolor='none', alpha=0.3)

        elif args.cam == 'CCD':
            # Plot FWHM_Y ans X for CCD
            ax.plot(bjds2, fwhm_x_ccd[region_name], 's', label=f'FWHM_X CCD, Avg= {avg_fwhm_x_ccd:.2f} μm', color='blue',
                    markerfacecolor='none', alpha=0.3)
            ax.plot(bjds2, fwhm_y_ccd[region_name], 's', label=f'FWHM_Y CCD, Avg= {avg_fwhm_y_ccd:.2f} μm', color='blueviolet',
                    markerfacecolor='none', alpha=0.3)

        # else plot both CMOS and CCD
        else:
            ax.plot(bjds1, fwhm_x_cmos[region_name], 'o', label=f'FWHM_X CMOS, Avg= {avg_fwhm_x_cmos:.2f} μm', color='red',
                    markerfacecolor='none', alpha=0.3)
            ax.plot(bjds1, fwhm_y_cmos[region_name], 'o', label=f'FWHM_Y CMOS, Avg= {avg_fwhm_y_cmos:.2f} μm', color='peru',
                    markerfacecolor='none', alpha=0.3)
            ax.plot(bjds2, fwhm_x_ccd[region_name], 's', label=f'FWHM_X CCD, Avg= {avg_fwhm_x_ccd:.2f} μm', color='blue',
                    markerfacecolor='none', alpha=0.3)
            ax.plot(bjds2, fwhm_y_ccd[region_name], 's', label=f'FWHM_Y CCD, Avg= {avg_fwhm_y_ccd:.2f} μm', color='blueviolet',
                    markerfacecolor='none', alpha=0.3)

        # Set region title and labels
        ax.set_title(f'{region_name}', fontsize=10)

        # Airmass on top x-axis for only the top row plots
        if i == 1:
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            interpolated_airmass = np.interp(ax.get_xticks(), bjds1, airmass1)
            ax2.set_xticks(ax.get_xticks())
            ax2.set_xticklabels([f'{a:.2f}' for a in interpolated_airmass], rotation=45, ha='right')
            ax2.set_xlabel('Airmass')

        if i == 3:
            ax.set_xlabel("BJD")
        else:
            ax.set_xlabel("")

        if j == 1:
            ax.set_ylabel("FWHM (microns)")
        else:
            ax.set_ylabel("")

        # Add legend with average values in each subplot
        ax.legend(loc='upper left', fontsize='small')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()