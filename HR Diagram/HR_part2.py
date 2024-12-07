import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord, Angle
from astropy.wcs.utils import skycoord_to_pixel
from photutils.background import Background2D, MedianBackground
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, aperture_photometry
from matplotlib.patches import Ellipse
import sep
import mplhep as hep
from scipy.stats import sigmaclip
import sep
sep.set_sub_object_limit(100000)  # Increase the limit as needed

def load_image(imagefile):
    """
    Load a FITS image and its header.

    Parameters:
        imagefile (str): Path to the FITS file.

    Returns:
        tuple: Image data and header.
    """
    with fits.open(imagefile) as hdul:
        image = hdul[0].data
        header = hdul[0].header
    return image, header

def plot_image(image, log_scale=True, origin='lower', cmap='gray', vmin=6, vmax=11, title=''):
    """
    Plot the image with specified parameters.

    Parameters:
        image (ndarray): Image data.
        log_scale (bool): Whether to apply logarithmic scaling.
        origin (str): Origin of the image.
        cmap (str): Colormap.
        vmin (float): Minimum data value for colormap.
        vmax (float): Maximum data value for colormap.
        title (str): Plot title.
    """
    plt.figure(figsize=(10, 10))
    if log_scale:
        plt.imshow(np.log(image + 1), origin=origin, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        plt.imshow(image, origin=origin, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis('off')
    if title:
        plt.title(title)
    plt.show()

def find_sources(image, bkg_median, fwhm=1, threshold_sigma=5, brightest=200):
    """
    Find sources in the image using DAOStarFinder.

    Parameters:
        image (ndarray): Image data.
        bkg_median (float): Median background level.
        fwhm (float): FWHM for the Gaussian kernel.
        threshold_sigma (float): Threshold in terms of sigma above background.
        brightest (int): Number of brightest sources to consider.

    Returns:
        Table: Detected sources.
    """
    mean, median, std = sigma_clipped_stats(image, sigma=3, cenfunc='mean')

    daofind = DAOStarFinder(fwhm=fwhm, threshold=std * threshold_sigma, brightest=brightest)
    sources = daofind(image - bkg_median)
    print("Detected Sources:\n", sources)
    return sources

def plot_apertures(image, apertures, origin='lower', cmap='gray', vmin=4, vmax=11, title=''):
    """
    Plot image with overplotted apertures.

    Parameters:
        image (ndarray): Image data.
        apertures (ApertureStats): Aperture objects to plot.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(np.log(image + 1), origin=origin, cmap=cmap, vmin=vmin, vmax=vmax)
    apertures.plot(color='red', lw=3, alpha=0.5)
    if title:
        plt.title(title)
    plt.show()

def extract_sep_sources(image, threshold=3, min_area=19):
    """
    Extract sources using SEP.

    Parameters:
        image (ndarray): Background-subtracted image data.
        threshold (float): Detection threshold in units of background RMS.
        min_area (int): Minimum area of sources to detect.

    Returns:
        ndarray: Array of detected objects.
    """
    objects = sep.extract(image, threshold, minarea=min_area, segmentation_map=False)
    print("SEP Extracted Objects:\n", objects)
    return objects

def plot_ellipses(image, objects, m, s, cmap='gray', vmin=None, vmax=None, title=None):
    plt.figure(figsize=(10, 10))
    fig, ax = plt.subplots()
    if vmin is None or vmax is None:
        im = ax.imshow(image, interpolation='nearest', cmap=cmap, origin='lower')
    else:
        im = ax.imshow(image, interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
    
    for obj in objects:
        e = Ellipse(
            xy=(obj['x'], obj['y']),
            width=6 * obj['a'],
            height=6 * obj['b'],
            angle=np.degrees(obj['theta'])
        )
        e.set_facecolor('none')
        e.set_edgecolor('red')
        ax.add_artist(e)
    
    if title is not None:
        plt.title(title)
    plt.show()


def perform_photometry(data_B, data_V):
    """
    Perform aperture photometry on B and V images.

    Parameters:
        data_B (ndarray): B-band image data.
        data_V (ndarray): V-band image data.

    Returns:
        tuple: positions, aperture sums for B and V images, and background-subtracted images.
    """
    # B-band
    data_B_sw = data_B.byteswap(False).newbyteorder()
    bkg_B = sep.Background(data_B_sw)
    data_sub_B = data_B_sw - bkg_B
    objects_B = sep.extract(data_sub_B, bkg_B.globalrms * 3, minarea=10, segmentation_map=False)
    positions_ = np.transpose((objects_B['x'], objects_B['y']))

    apertures = CircularAperture(positions_, r=5)
    phot_table_B = aperture_photometry(data_sub_B, apertures)
    aperture_sums_B = phot_table_B['aperture_sum']

    # V-band
    data_V_sw = data_V.byteswap(False).newbyteorder()
    bkg_V = sep.Background(data_V_sw)
    data_sub_V = data_V_sw - bkg_V
    phot_table_V = aperture_photometry(data_sub_V, apertures)
    aperture_sums_V = phot_table_V['aperture_sum']

    # Filter out invalid measurements
    valid = (aperture_sums_B > 0) & (aperture_sums_V > 0)
    positions = positions_[valid]
    aperture_sums_B = aperture_sums_B[valid]
    aperture_sums_V = aperture_sums_V[valid]

    return positions, aperture_sums_B, aperture_sums_V, data_sub_B, data_sub_V, bkg_V

def calibrate_zero_point(instrumental, reference):
    """
    Calibrate zero point based on reference stars.

    Parameters:
        instrumental (array-like): Instrumental magnitudes.
        reference (array-like): Reference magnitudes.

    Returns:
        tuple: slope and intercept for calibration.
    """
    x = instrumental
    y = reference
    xbar = np.mean(x)
    ybar = np.mean(y)
    # Assume slope = 1 for simplicity
    m = 1
    b = ybar - m * xbar
    return m, b

def main():
    # Load B and V images
    imagefile_B = "B.tif.fits"
    imagefile_V = "V.tif.fits"
    image_B, header_B = load_image(imagefile_B)
    image_V, header_V = load_image(imagefile_V)
    print(f"B-band Image Shape: {image_B.shape}")

    # Print Modified Julian Date if present
    mjd = header_B.get('MJD-OBS', 'N/A')
    print("The Modified Julian Date is", mjd)

    # Background estimation for B-band
    sigma_clip = SigmaClip(sigma=3, cenfunc='mean')
    mean, median, std = sigma_clipped_stats(image_B, sigma=3, cenfunc='mean')

    bkg_estimator = MedianBackground()
    bkg = Background2D(
        image_B,
        box_size=(10, 10),
        filter_size=(9, 9),
        sigma_clip=sigma_clip,
        bkg_estimator=bkg_estimator
    )
    bkg_median = bkg.background_median
    bkg_rms = bkg.background_rms_median

    # Find sources in B-band
    sources = find_sources(image_B, bkg_median, fwhm=1, threshold_sigma=5, brightest=500)

    if sources is not None:
        # Plot apertures on B-band image
        positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
        apertures = CircularAperture(positions, r=15)
    else:
        print("No sources detected in B-band image.")

    # Perform SEP source extraction on B-band
    data_B_sw = image_B.byteswap(False).newbyteorder()
    objects_sep = extract_sep_sources(data_B_sw - bkg_median, threshold=3, min_area=19)

    # Plot ellipses for SEP-detected objects
    m, s = np.mean(data_B_sw - bkg_median), np.std(data_B_sw - bkg_median)

    # Perform photometry on B and V images
    positions, sums_B, sums_V, data_sub_B, data_sub_V, bkg_V = perform_photometry(image_B, image_V)

    # Calculate magnitudes and color index (B - V)
    Bmag = -2.5 * np.log10(sums_B)
    Vmag = -2.5 * np.log10(sums_V)
    color = Bmag - Vmag

    # Plot B vs V magnitudes (uncalibrated)
    plt.figure(figsize=(8, 6))
    plt.scatter(Bmag, Vmag, alpha=0.7)
    plt.xlabel('B magnitude (instrumental)')
    plt.ylabel('V magnitude (instrumental)')
    plt.title('M53: B vs V Magnitudes (Uncalibrated)')
    plt.grid(True)
    plt.show()

    # Use mplhep style if desired
    plt.style.use(hep.style.ROOT)

    # Plot Uncalibrated CMD
    plt.figure(figsize=(6, 5))
    plt.scatter(color, Vmag, marker='o', edgecolors='none', s=10, c='black', alpha=0.7)
    plt.xlabel('B - V Mag (instrumental)',fontsize = 28)
    plt.ylabel('V Mag (instrumental)',fontsize = 28)
    plt.gca().invert_yaxis()
    plt.title('M53 HR Diagram (Uncalibrated)',fontsize = 38)
    plt.grid(True)
    plt.show()



    RA = ['198.310127', '198.286042', '198.070813', '198.156493', '198.318738'] 
    DEC = ['18.033275', '18.024618', '18.319831', '18.341921', '18.365208']  
    ref_mags_B = [9.808, 10.193, 10.87, 11.992, 12.720] 
    ref_mags_V = [10.2, 11.993, 12.152, 12.603, 13.422] 
    

    # Define WCS for B-band
    wcs_B = WCS(header_B)

    # Convert reference coordinates to pixel positions
    ra_angle = Angle(RA, unit=u.degree)
    dec_angle = Angle(DEC, unit=u.degree)
    ref_coord = SkyCoord(ra=ra_angle, dec=dec_angle, frame='icrs')
    ref_xy = skycoord_to_pixel(ref_coord, wcs_B)
    ref_positions = np.transpose((ref_xy[0], ref_xy[1]))
    ref_apertures = CircularAperture(ref_positions, r=5)

    # Plot reference apertures on B-band image
    plt.figure(figsize=(10, 10))
    plt.imshow(data_sub_B, origin='lower', cmap='gray', vmin=m-s, vmax=m+s)
    ref_apertures.plot(color='blue', lw=5, alpha=1)
    plt.title('Reference Stars (M53)')
    plt.show()

    # Perform photometry on reference stars for calibration
    phot_table_ref_B = aperture_photometry(data_sub_B, ref_apertures)
    phot_table_ref_V = aperture_photometry(data_sub_V, ref_apertures)

    ref_B_uncali = -2.5 * np.log10(phot_table_ref_B['aperture_sum'])
    ref_V_uncali = -2.5 * np.log10(phot_table_ref_V['aperture_sum'])

    print("Uncalibrated Reference B Magnitudes:\n", ref_B_uncali)
    print("Uncalibrated Reference V Magnitudes:\n", ref_V_uncali)

    # Calibrate zero points (assuming you have correct reference_magnitudes)
    # For demonstration, we'll just use ref_mags_B for B and ref_mags_V for V
    _, b_B = calibrate_zero_point(ref_B_uncali, ref_mags_B)
    _, b_V = calibrate_zero_point(ref_V_uncali, ref_mags_V)
    print(f"B-band Zero Point: {b_B:.2f}")
    print(f"V-band Zero Point: {b_V:.2f}")

    # Apply zero-point corrections
    Bmag_cali = Bmag + b_B
    Vmag_cali = Vmag + b_V
    color_cali = Bmag_cali - Vmag_cali

    cmap = LinearSegmentedColormap.from_list('blue_orange', ['blue', 'darkorange'], N=256)
    # Plot Calibrated CMD
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(color_cali, Vmag_cali, marker='o', edgecolors='none', s=10, c=color_cali, cmap=cmap, alpha=0.7)
    plt.xlabel('B - V Mag (Calibrated)')
    plt.ylabel('V Mag (Calibrated)')
    plt.gca().invert_yaxis()
    plt.title('M53 HR Diagram (Calibrated)')
    plt.grid(True)

    # Add a colorbar to show the color scale for (B - V)
    cbar = plt.colorbar(sc)
    cbar.set_label('Calibrated (B - V)')

    plt.show()

        # After you've calculated:
    # Bmag, Vmag, color, Bmag_cali, Vmag_cali, color_cali
    # Create a custom colormap from blue to orange
    cmap = LinearSegmentedColormap.from_list('blue_orange', ['blue', 'darkorange'], N=256)

    # Create a single figure to plot both uncalibrated and calibrated data
    plt.figure(figsize=(8, 6))

    # Plot uncalibrated (V vs B-V)
    # Uncalibrated data in black
    plt.scatter(color, Vmag, marker='o', edgecolors='none', s=10, c='black', alpha=0.5, label='Uncalibrated')

    # Plot calibrated (V vs B-V)
    # Calibrated data colored by B-V
    sc = plt.scatter(color_cali, Vmag_cali, marker='o', edgecolors='none', s=10, c=color_cali, cmap=cmap, alpha=0.7, label='Calibrated')

    # Invert y-axis for magnitude plots
    plt.gca().invert_yaxis()

    # Label axes
    plt.xlabel('B - V')
    plt.ylabel('V magnitude')
    plt.title('M53: HR Diagram (Uncalibrated & Calibrated)')

    # Add a colorbar for the calibrated data
    cbar = plt.colorbar(sc)
    cbar.set_label('Calibrated B - V) Color Index')

    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
