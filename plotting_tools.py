import json

import matplotlib.pyplot as plt
from astropy.coordinates import EarthLocation, SkyCoord
import astropy.units as u
import numpy as np


def plot_images():
    # Set plot parameters
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['xtick.top'] = True
    plt.rcParams['xtick.labeltop'] = False
    plt.rcParams['xtick.labelbottom'] = True
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['xtick.major.top'] = True
    plt.rcParams['xtick.minor.top'] = True
    plt.rcParams['xtick.minor.bottom'] = True
    plt.rcParams['xtick.alignment'] = 'center'

    plt.rcParams['ytick.left'] = True
    plt.rcParams['ytick.labelleft'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['ytick.major.right'] = True
    plt.rcParams['ytick.major.left'] = True
    plt.rcParams['ytick.minor.right'] = True
    plt.rcParams['ytick.minor.left'] = True

    # Font and fontsize
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12

    # Legend
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.8
    plt.rcParams['legend.loc'] = 'best'
    plt.rcParams['legend.fancybox'] = True
    plt.rcParams['legend.fontsize'] = 12


def get_location():
    """
    Get the location of the observatory

    Returns
    -------
    loc : EarthLocation
        Location of the observatory

    Raises
    ------
    None
    """
    site_location = EarthLocation(
        lat=-24.615662 * u.deg,
        lon=-70.391809 * u.deg,
        height=2433 * u.m
    )

    return site_location


def get_light_travel_times(ra, dec, time_to_correct):
    """
    Get the light travel times to the helio- and
    barycentric

    Parameters
    ----------
    ra : str
        The Right Ascension of the target in hour-angle
        e.g. 16:00:00
    dec : str
        The Declination of the target in degrees
        e.g. +20:00:00
    time_to_correct : astropy.Time object
        The time of observation to correct. The astropy.Time
        object must have been initialised with an EarthLocation

    Returns
    -------
    ltt_bary : float
        The light travel time to the barycentre
    ltt_helio : float
        The light travel time to the heliocentric

    Raises
    ------
    None
    """
    target = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame='icrs')
    ltt_bary = time_to_correct.light_travel_time(target)
    ltt_helio = time_to_correct.light_travel_time(target, 'heliocentric')
    return ltt_bary, ltt_helio


# Function to fit a 2D Gaussian
def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
    g = offset + amplitude * np.exp(-(a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2)))
    return g.ravel()


def calculate_airmass(altitude):
    return 1 / np.cos(np.radians(90 - altitude))


def save_results_json(times, airmass_values, fwhm_values, ratio_values, fwhm_results, pixel_size):
    """
    Save the results of FWHM calculations to a JSON file with regional details.

    Parameters:
        times (list): List of BJD values.
        airmass_values (list): List of airmass values.
        fwhm_values (list): List of overall FWHM values.
        ratio_values (list): List of overall ratio values.
        fwhm_results (dict): Dictionary containing FWHM results for each region.
        pixel_size (float): The pixel size used to determine the camera type.
    """

    # Prepare data in dictionary format for JSON output with regional details
    data_dict = {"results": []}

    for i, (bjd, airmass, fwhm, ratio) in enumerate(zip(times, airmass_values, fwhm_values, ratio_values)):
        # Create a data structure with details for each region
        region_data = {
            "BJD": bjd,
            "Airmass": airmass,
            "Overall_FWHM": fwhm,
            "Overall_Ratio": ratio,
            "Regions": [
                {
                    "Region": region,
                    "FWHM": results["FWHM"],
                    "Ratio": results["Ratio"]
                }
                for region, results in fwhm_results.items()
            ]
        }
        data_dict["results"].append(region_data)

    # Determine the camera type based on pixel size
    camera = "Unknown"
    if pixel_size == 11:
        camera = "CMOS"
    elif pixel_size == 13.5:
        camera = "CCD"

    # Write data to JSON file with camera type in the filename
    json_filename = f"fwhm_batches_{camera}.json"
    with open(json_filename, "w") as json_file:
        json.dump(data_dict, json_file, indent=4)

    print(f"Results saved to {json_filename}")