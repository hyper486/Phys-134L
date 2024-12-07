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
import astroalign as aa

sep.set_sub_object_limit(100000)  # Increase the limit if needed
plt.style.use(hep.style.ROOT)


def ensure_native_float32(data):
    if data.dtype.byteorder not in ('=', '|'):
        data = data.byteswap().newbyteorder()
    if data.dtype != np.float32:
        data = data.astype(np.float32, copy=False)
    return data


# Alignment Function
def align_images(reference_image, target_image):
    # Ensure native byte order and float32 before SEP calls
    reference_image = ensure_native_float32(reference_image)
    target_image = ensure_native_float32(target_image)

    # SEP extraction for alignment stars
    bkg_ref = sep.Background(reference_image)
    data_ref = reference_image - bkg_ref
    objects_ref = sep.extract(data_ref, thresh=5, minarea=5, segmentation_map=False)

    bkg_target = sep.Background(target_image)
    data_target = target_image - bkg_target
    objects_target = sep.extract(data_target, thresh=5, minarea=5, segmentation_map=False)
    
    # If WCS is not available, use pixel coordinates only
    stars_ref = np.vstack((objects_ref['x'], objects_ref['y'])).T
    stars_target = np.vstack((objects_target['x'], objects_target['y'])).T
    
    # Align images using astroalign
    try:
        aligned_target, footprint = aa.register(target_image, reference_image)
        print("Images aligned successfully.")
    except Exception as e:
        print("Error aligning images:", e)
        # If alignment fails, return the original image
        aligned_target = target_image

    aligned_target = ensure_native_float32(aligned_target)
    return aligned_target


# Load Image Function
def load_image(imagefile):
    with fits.open(imagefile) as hdul:
        image = hdul[0].data
        header = hdul[0].header
    return image, header


# Plot Image Function
def plot_image(image, log_scale=True, origin='lower', cmap='gray', vmin=6, vmax=11, title=''):
    plt.figure(figsize=(10, 10))
    img_safe = image.copy()
    img_safe[img_safe < 0] = 0  # Avoid log of negative
    if log_scale:
        plt.imshow(np.log(img_safe + 1), origin=origin, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        plt.imshow(img_safe, origin=origin, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis('off')
    if title:
        plt.title(title)
    plt.show()


# Find Sources with DAOStarFinder
def find_sources(image, bkg_median, fwhm=1, threshold_sigma=5, brightest=200):
    mean, median, std = sigma_clipped_stats(image, sigma=3, cenfunc='mean')
    daofind = DAOStarFinder(fwhm=fwhm, threshold=std * threshold_sigma, brightest=brightest)
    sources = daofind(image - bkg_median)
    print("Detected Sources:\n", sources)
    return sources

# Plot Apertures Function
def plot_apertures(image, apertures, origin='lower', cmap='gray', vmin=4, vmax=11, title=''):
    plt.figure(figsize=(10, 10))
    img_safe = image.copy()
    img_safe[img_safe < 0] = 0
    plt.imshow(np.log(img_safe + 1), origin=origin, cmap=cmap, vmin=vmin, vmax=vmax)
    apertures.plot(color='lightcoral', lw=3, alpha=0.3)
    if title:
        plt.title(title)
    plt.show()


# Extract SEP Sources
def extract_sep_sources(image, threshold=3, min_area=19):
    objects = sep.extract(image, threshold, minarea=min_area, segmentation_map=False)
    print("SEP Extracted Objects:\n", objects)
    return objects


# Plot Ellipses around SEP Sources
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


# Perform Photometry
def perform_photometry(data_B, data_V):
    # Ensure arrays are native float32 before SEP
    data_B = ensure_native_float32(data_B)
    data_V = ensure_native_float32(data_V)

    bkg_B = sep.Background(data_B)
    data_sub_B = data_B - bkg_B
    objects_B = sep.extract(data_sub_B, 3.0, minarea=10, segmentation_map=False)

    positions_ = np.transpose((objects_B['x'], objects_B['y']))
    apertures = CircularAperture(positions_, r=5)
    phot_table_B = aperture_photometry(data_sub_B, apertures)
    aperture_sums_B = phot_table_B['aperture_sum']

    bkg_V = sep.Background(data_V)
    data_sub_V = data_V - bkg_V
    phot_table_V = aperture_photometry(data_sub_V, apertures)
    aperture_sums_V = phot_table_V['aperture_sum']

    valid = (aperture_sums_B > 0) & (aperture_sums_V > 0)
    positions = positions_[valid]
    aperture_sums_B = aperture_sums_B[valid]
    aperture_sums_V = aperture_sums_V[valid]

    return positions, aperture_sums_B, aperture_sums_V, data_sub_B, data_sub_V, bkg_V


# Calibrate Zero Point
def calibrate_zero_point(instrumental, reference):
    x = instrumental
    y = reference
    xbar = np.mean(x)
    ybar = np.mean(y)
    # slope=1 for simplicity, as in original code
    m = 1
    b = ybar - m * xbar
    return m, b

# Main
def main():
    # Load B and V images
    imagefile_B = "B.tif.fits"
    imagefile_V = "V.tif.fits"
    image_B, header_B = load_image(imagefile_B)
    image_V, header_V = load_image(imagefile_V)
    
    # Ensure correct format after loading
    image_B = ensure_native_float32(image_B)
    image_V = ensure_native_float32(image_V)

    print(f"B-band Image Shape: {image_B.shape}")
    print(f"V-band Image Shape: {image_V.shape}")
    
    # Plot B-band and V-band images before alignment
    plot_image(image_B, log_scale=True, vmin=4, vmax=11, title='B-band Image (Log Scale)')
    plot_image(image_V, log_scale=True, vmin=4, vmax=11, title='V-band Image (Log Scale)')
    
    # Align V-band image to B-band image
    aligned_V = align_images(image_B, image_V)

    # *** Remove the second alignment call (original code had a second call)
    # aligned_V, _ = aa.register(image_V, image_B)  # Remove this line to avoid double alignment

    # Ensure aligned_V format
    aligned_V = ensure_native_float32(aligned_V)

    # Plot V-band image after alignment
    plot_image(aligned_V, log_scale=True, vmin=4, vmax=11, title='V-band Image (Aligned, Log Scale)')
    
    # Background estimation for B-band
    sigma_clip_obj = SigmaClip(sigma=3, cenfunc='mean')
    mean_B, median_B, std_B = sigma_clipped_stats(image_B, sigma=3, cenfunc='mean')
    
    bkg_estimator = MedianBackground()
    bkg_B = Background2D(
        image_B,
        box_size=(10, 10),
        filter_size=(9, 9),
        sigma_clip=sigma_clip_obj,
        bkg_estimator=bkg_estimator
    )
    bkg_median_B = bkg_B.background_median
    
    # Background estimation for V-band
    mean_V, median_V, std_V = sigma_clipped_stats(aligned_V, sigma=3, cenfunc='mean')
    bkg_V_obj = Background2D(
        aligned_V,
        box_size=(10, 10),
        filter_size=(9, 9),
        sigma_clip=sigma_clip_obj,
        bkg_estimator=bkg_estimator
    )
    bkg_median_V = bkg_V_obj.background_median

    # Find sources in B-band
    sources_B = find_sources(image_B, bkg_median_B, fwhm=2.0, threshold_sigma=5, brightest=500)
    if sources_B is not None:
        positions_B = np.transpose((sources_B['xcentroid'], sources_B['ycentroid']))
        apertures_B = CircularAperture(positions_B, r=15)
        plot_apertures(image_B, apertures_B, vmin=4, vmax=11, title='B-band Sources with Apertures')
    else:
        print("No sources detected in B-band image.")
    
    # Find sources in V-band
    sources_V = find_sources(aligned_V, bkg_median_V, fwhm=2.0, threshold_sigma=5, brightest=500)
    if sources_V is not None:
        positions_V = np.transpose((sources_V['xcentroid'], sources_V['ycentroid']))
        apertures_V = CircularAperture(positions_V, r=15)
        plot_apertures(aligned_V, apertures_V, vmin=4, vmax=11, title='V-band Sources with Apertures')
    else:
        print("No sources detected in V-band image.")
    
    # Perform photometry
    positions, sums_B, sums_V, data_sub_B, data_sub_V, bkg_V = perform_photometry(image_B, aligned_V)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        Bmag = -2.5 * np.log10(sums_B)
        Vmag = -2.5 * np.log10(sums_V)
    
    valid_mag = np.isfinite(Bmag) & np.isfinite(Vmag)
    Bmag = Bmag[valid_mag]
    Vmag = Vmag[valid_mag]
    color = Bmag - Vmag
    
       
    # Reference Star Calibration
    RA = ['198.310127', '198.286042', '198.070813', '198.156493', '198.318738']
    DEC = ['18.033275', '18.024618', '18.319831', '18.341921', '18.365208']
    ref_mags_B = [9.808, 10.193, 10.87, 11.992, 12.720]
    ref_mags_V = [10.2, 11.993, 12.152, 12.603, 13.422]
    
    RA_deg = [float(ra) for ra in RA]
    DEC_deg = [float(dec) for dec in DEC]
    ref_coord = SkyCoord(ra=RA_deg*u.degree, dec=DEC_deg*u.degree, frame='icrs')
    
    wcs_B = WCS(header_B)
    ref_x, ref_y = skycoord_to_pixel(ref_coord, wcs_B)
    ref_positions = np.transpose((ref_x, ref_y))
    ref_apertures = CircularAperture(ref_positions, r=5)
    
    # Plot reference apertures on B-sub image
    plt.figure(figsize=(10, 10))
    median_B_sub, std_B_sub = np.median(data_sub_B), np.std(data_sub_B)
    plt.imshow(data_sub_B, origin='lower', cmap='gray', vmin=median_B_sub - std_B_sub, vmax=median_B_sub + 5*std_B_sub)
    ref_apertures.plot(color='powderblue', lw=12, alpha=1)
    plt.title('Reference Stars (M53)')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.show()

if __name__ == "__main__":
    main()
