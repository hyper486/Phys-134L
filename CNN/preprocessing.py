import os
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')  # Force using the TkAgg backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import sys
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.simbad import Simbad
import numpy as np
import warnings
from astropy.utils.exceptions import AstropyWarning

# Ignore Astropy warnings
warnings.simplefilter('ignore', AstropyWarning)

# Set global font to SimHei (Heiti) to support Chinese characters
plt.rcParams['font.family'] = ['SimHei']  # Or 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False  # Solve the problem of minus signs displayed as squares

class FITSImageProcessor:
    def __init__(self, parent_dir, output_dir):
        
        self.parent_dir = Path(parent_dir)
        self.output_dir = Path(output_dir)
        self.simbad = Simbad()
        self.simbad.add_votable_fields('ra', 'dec')
        # Define supported FITS file extensions
        self.fits_extensions = ['.fits', '.fit', '.fz', '.fits.gz', '.fit.fz', '.fits.fz']
        self.current_star = None
        self.target_ra = None
        self.target_dec = None

    def process_all_folders(self):
        # Check if the parent directory contains subfolders
        folders = [folder for folder in sorted(self.parent_dir.iterdir()) if folder.is_dir()]
        if folders:
            print(f"Found {len(folders)} subfolders in the parent directory '{self.parent_dir}'.")
            # Iterate through all subfolders in the parent directory
            for folder in folders:
                print(f"\nFound subfolder: {folder.name}")
                self.current_folder = folder
                self.current_star = folder.name

                # Query SIMBAD to get RA and Dec
                coords = self.get_ra_dec(self.current_star)
                if coords is None:
                    print(f"Could not retrieve coordinates for object {self.current_star}, skipping.")
                    continue
                self.target_ra, self.target_dec = coords
                print(f"Coordinates of {self.current_star}: RA = {self.target_ra}°, Dec = {self.target_dec}°")

                # Create output folder
                star_output_dir = self.output_dir / self.current_star
                star_output_dir.mkdir(parents=True, exist_ok=True)

                # Get all FITS files in the folder
                fits_files = sorted([f for f in folder.iterdir() if self.is_fits_file(f)])

                if not fits_files:
                    print(f"No FITS files found in folder '{self.current_star}', skipping.")
                    continue
                else:
                    print(f"Found {len(fits_files)} FITS files in folder '{self.current_star}'.")

                # Process each FITS file
                for fits_file in fits_files:
                    self.process_single_fits(fits_file, star_output_dir)
        else:
            print(f"No subfolders found in the parent directory '{self.parent_dir}'.")
            # Treat the parent directory itself as a single object folder
            self.current_folder = self.parent_dir
            self.current_star = self.parent_dir.name
            print(f"\nTreating the parent directory '{self.current_star}' as a single object.")

            # Query SIMBAD to get RA and Dec
            coords = self.get_ra_dec(self.current_star)
            if coords is None:
                print(f"Could not retrieve coordinates for object {self.current_star}, cannot proceed.")
                return
            self.target_ra, self.target_dec = coords
            print(f"Coordinates of {self.current_star}: RA = {self.target_ra}°, Dec = {self.target_dec}°")

            # Create output folder
            star_output_dir = self.output_dir / self.current_star
            star_output_dir.mkdir(parents=True, exist_ok=True)

            # Get all FITS files in the parent directory
            fits_files = sorted([f for f in self.parent_dir.iterdir() if self.is_fits_file(f)])

            if not fits_files:
                print(f"No FITS files found in the parent directory '{self.current_star}', cannot proceed.")
                return
            else:
                print(f"Found {len(fits_files)} FITS files in the parent directory '{self.current_star}'.")

            # Process each FITS file
            for fits_file in fits_files:
                self.process_single_fits(fits_file, star_output_dir)

        print("\nAll folders' images have been processed.")

    def is_fits_file(self, file_path):
        file_name = file_path.name.lower()
        for ext in self.fits_extensions:
            if file_name.endswith(ext):
                return True
        return False

    def get_ra_dec(self, object_name):
        try:
            result_table = self.simbad.query_object(object_name)
            if result_table is None:
                print(f"Object not found in SIMBAD: {object_name}")
                return None
            ra = result_table['RA'][0]
            dec = result_table['DEC'][0]
            coord = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
            return coord.ra.deg, coord.dec.deg
        except Exception as e:
            print(f"Error during query: {e}")
            return None

    def process_single_fits(self, fits_path, output_dir):
        print(f"\nProcessing image: {fits_path.name}")
        try:
            with fits.open(fits_path) as hdulist:
                # Print HDU information
                print("FITS file structure:")
                hdulist.info()

                # Get SCI HDU (assuming the name is 'SCI')
                if 'SCI' in hdulist:
                    sci_hdu = hdulist['SCI']
                else:
                    # If there's no 'SCI' extension, try using the first HDU with data
                    sci_hdu = None
                    for hdu in hdulist:
                        if hdu.is_image and hdu.data is not None:
                            sci_hdu = hdu
                            break
                    if sci_hdu is None:
                        print(f"No usable data extension found in file {fits_path.name}, skipping.")
                        return

                data = sci_hdu.data
                header = sci_hdu.header

                # Get WCS information
                wcs = WCS(header)
                if not wcs.is_celestial:
                    print(f"Invalid WCS information in file {fits_path.name}, skipping.")
                    return

                # Convert the object's RA/Dec to pixel coordinates
                sky_coord = SkyCoord(ra=self.target_ra*u.degree, dec=self.target_dec*u.degree, frame='icrs')
                x_pixel, y_pixel = wcs.world_to_pixel(sky_coord)
                print(f"Coordinates of {self.current_star} in pixels: x = {x_pixel:.2f}, y = {y_pixel:.2f}")

                # Get basic statistics of the data
                base_vmin = np.nanmin(data)
                base_vmax = np.nanmax(data)

                # Calculate cropping area coordinates
                fixed_width = 500
                fixed_height = 500

                x1 = int(x_pixel - fixed_width // 2)
                y1 = int(y_pixel - fixed_height // 2)
                x2 = x1 + fixed_width
                y2 = y1 + fixed_height

                # Ensure the cropping area does not exceed image boundaries
                x1 = max(x1, 0)
                y1 = max(y1, 0)
                x2 = min(x2, data.shape[1])
                y2 = min(y2, data.shape[0])

                # Update actual width and height
                actual_width = x2 - x1
                actual_height = y2 - y1

                print(f"Cropping area: x = {x1}, y = {y1}, width = {actual_width}, height = {actual_height}")

                # Draw the cropping area
                fig, ax = plt.subplots(figsize=(10, 8))
                plt.subplots_adjust(left=0.25, bottom=0.25)

                # Display the image
                im = ax.imshow(data, cmap='gray', origin='lower', vmin=base_vmin, vmax=base_vmax)
                ax.set_title(f"{self.current_star} - {fits_path.name}")
                ax.set_xlabel('X Pixel')
                ax.set_ylabel('Y Pixel')

                # Mark the target star
                ax.scatter(x_pixel, y_pixel, s=100, edgecolor='red', facecolor='none', linewidth=2, label=self.current_star)
                ax.legend(loc='upper right')

                # Draw the cropping area's rectangle
                rect = Rectangle((x1, y1), actual_width, actual_height,
                                 linewidth=2, edgecolor='green', facecolor='none')
                ax.add_patch(rect)

                # Add brightness adjustment slider
                axcolor = 'lightgoldenrodyellow'
                ax_brightness = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
                brightness_slider = Slider(
                    ax=ax_brightness,
                    label='Brightness Adjustment',
                    valmin=0.1,
                    valmax=10.0,
                    valinit=1.0,
                    valstep=0.1
                )

                def update_brightness(val):
                    brightness = brightness_slider.val
                    new_vmax = base_vmax / brightness
                    im.set_clim(vmin=base_vmin, vmax=new_vmax)
                    cbar.update_normal(im)
                    fig.canvas.draw_idle()

                brightness_slider.on_changed(update_brightness)

                # Add zoom slider
                ax_zoom = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
                zoom_slider = Slider(
                    ax=ax_zoom,
                    label='Zoom Factor',
                    valmin=1.0,
                    valmax=10.0,
                    valinit=1.0,
                    valstep=0.5
                )

                def update_zoom(val):
                    zoom = zoom_slider.val
                    # Calculate new display range
                    half_width = (data.shape[1] / 2) / zoom
                    half_height = (data.shape[0] / 2) / zoom
                    ax.set_xlim(x_pixel - half_width, x_pixel + half_width)
                    ax.set_ylim(y_pixel - half_height, y_pixel + half_height)
                    fig.canvas.draw_idle()

                zoom_slider.on_changed(update_zoom)

                # Display color bar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Intensity')

                plt.draw()
                plt.show()  # Display the image

                # After the image window is closed, continue
                print("Cropping area has been automatically set to center 500x500 pixels.")
                print("If you need to adjust the cropping area, please enter new offset values. Otherwise, press Enter to continue saving.")
                user_input = input("Do you need to adjust the cropping area? Enter 'y' to adjust, press Enter to skip: ").strip().lower()

                if user_input == 'y':
                    try:
                        dx = int(input("Please enter the x-axis offset (positive for right, negative for left): "))
                        dy = int(input("Please enter the y-axis offset (positive for up, negative for down): "))

                        # Calculate new cropping area
                        new_x1 = x1 + dx
                        new_y1 = y1 + dy
                        new_x2 = new_x1 + fixed_width
                        new_y2 = new_y1 + fixed_height

                        # Ensure the cropping area does not exceed image boundaries
                        new_x1 = max(new_x1, 0)
                        new_y1 = max(new_y1, 0)
                        new_x2 = min(new_x2, data.shape[1])
                        new_y2 = min(new_y2, data.shape[0])

                        # Update actual width and height
                        new_width = new_x2 - new_x1
                        new_height = new_y2 - new_y1

                        print(f"New cropping area: x = {new_x1}, y = {new_y1}, width = {new_width}, height = {new_height}")

                        # Draw the new cropping area
                        fig, ax = plt.subplots(figsize=(10, 8))
                        plt.subplots_adjust(left=0.25, bottom=0.25)
                        im = ax.imshow(data, cmap='gray', origin='lower', vmin=base_vmin, vmax=base_vmax)
                        ax.scatter(x_pixel, y_pixel, s=100, edgecolor='red', facecolor='none', linewidth=2)
                        rect_new = Rectangle((new_x1, new_y1), new_width, new_height,
                                             linewidth=2, edgecolor='blue', facecolor='none')
                        ax.add_patch(rect_new)
                        plt.show()

                        # Update cropping coordinates
                        crop_coords = {'x': new_x1, 'y': new_y1, 'width': new_width, 'height': new_height}

                        # Ask user whether to save the cropping area
                        self.ask_save(crop_coords, data, fits_path.name, output_dir)
                    except Exception as e:
                        print(f"Error while adjusting cropping area: {e}")
                        print("Saving with the default cropping area.")
                        self.ask_save({'x': x1, 'y': y1, 'width': actual_width, 'height': actual_height}, data, fits_path.name, output_dir)
                else:
                    # Use the default cropping area
                    self.ask_save({'x': x1, 'y': y1, 'width': actual_width, 'height': actual_height}, data, fits_path.name, output_dir)

        except Exception as e:
            print(f"Error processing image {fits_path.name}: {e}")

    def ask_save(self, crop_coords, data, original_name, output_dir):
        if crop_coords['width'] < 1 or crop_coords['height'] < 1:
            print("Invalid cropping area, skipping save.")
            return

        while True:
            user_input = input("Do you want to save the cropping area? Enter 's' to save, 'k' to skip: ").strip().lower()
            if user_input == 's':
                self.save_crop(data, (crop_coords['x'], crop_coords['y'],
                                      crop_coords['x'] + crop_coords['width'],
                                      crop_coords['y'] + crop_coords['height']),
                               original_name, output_dir)
                break
            elif user_input == 'k':
                print("Skipping saving this cropping area.")
                break
            else:
                print("Invalid input, please enter 's' or 'k'.")

    def save_crop(self, data, coords, original_name, output_dir):
        x1, y1, x2, y2 = coords
        cropped_data = data[y1:y2, x1:x2]
        # Convert cropped data to image
        try:
            # Handle data types, ensure data is within 0-65535 range
            if np.issubdtype(cropped_data.dtype, np.floating):
                # Normalize to 0-1
                cropped_data = (cropped_data - np.nanmin(cropped_data)) / (np.nanmax(cropped_data) - np.nanmin(cropped_data))
                cropped_data = np.clip(cropped_data, 0, 1)
                # Scale to 0-65535
                cropped_data = (cropped_data * 65535).astype(np.uint16)
            elif cropped_data.dtype == np.int16:
                # Assume data range is -32768 to 32767, map to 0-65535
                cropped_data = np.clip(cropped_data, -32768, 32767)
                cropped_data = ((cropped_data + 32768) / 65535 * 65535).astype(np.uint16)
            elif cropped_data.dtype == np.uint16:
                # Keep as is
                cropped_data = cropped_data.astype(np.uint16)
            else:
                # Adjust based on data type
                cropped_data = np.clip(cropped_data, 0, 65535)
                cropped_data = cropped_data.astype(np.uint16)

            # Create PIL image, use 'I;16' mode to save as 16-bit PNG
            cropped_image = Image.fromarray(cropped_data, mode='I;16')
        except Exception as e:
            print(f"Error converting cropped data to image: {e}")
            return

        # Construct save path, name as originalname_x_y.png
        save_name = f"{Path(original_name).stem}_{x1}_{y1}.png"
        save_path = output_dir / save_name
        try:
            cropped_image.save(save_path, format='PNG')
            print(f"Cropping area saved as: {save_path}")
        except Exception as e:
            print(f"Could not save cropping area, error: {e}")

def select_directory(title):
    root = tk.Tk()
    root.withdraw()
    selected_dir = filedialog.askdirectory(title=title)
    root.destroy()
    if not selected_dir:
        print("No directory selected, exiting program.")
        sys.exit()
    return selected_dir

def main():
    print("Please select the parent directory containing the object folders.")
    parent_dir = select_directory("Select Parent Directory")
    print(f"You selected the parent directory: {parent_dir}")

    print("Please select the output directory for saving cropped images.")
    output_dir = select_directory("Select Output Directory")
    print(f"You selected the output directory: {output_dir}")

    processor = FITSImageProcessor(parent_dir, output_dir)
    processor.process_all_folders()

if __name__ == "__main__":
    main()
