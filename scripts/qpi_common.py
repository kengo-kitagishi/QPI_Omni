"""
QPI Analysis - Common Functions and Constants

Define commonly used functions and constants
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.optimize import curve_fit
from qpi import QPIParameters, make_disk

# ==================== Default Parameters ====================

# Optical system constants
WAVELENGTH = 663e-9  # 663 nm
NA = 0.95
PIXELSIZE = 3.45e-6 / 40  # 8.625e-8 m

# Commonly used off-axis center coordinates
OFFAXIS_CENTER_DEFAULT = (1623, 1621)

# ==================== Visibility Function ====================

def visibility(array: np.ndarray, params: QPIParameters) -> np.ndarray:
    """
    Compute interference fringe visibility

    Parameters
    ----------
    array : np.ndarray
        Input image
    params : QPIParameters
        QPI parameters

    Returns
    -------
    np.ndarray
        Visibility map
    """
    img = np.array(array)
    radius = params.aperturesize // 2
    
    # Get 0th order (DC) image
    img_freq = np.fft.fftshift(np.fft.fft2(img))
    disk_0th = make_disk(params.img_center, radius, img.shape)
    img_freq_0th = img_freq * disk_0th
    img_0th = np.fft.ifft2(np.fft.ifftshift(img_freq_0th))
    img_0th_abs = np.abs(img_0th)
    
    # Get 1st order (interferometric) image
    disk_1th = make_disk(params.offaxis_center, radius, img.shape)
    img_freq_interfere = img_freq * disk_1th
    img_interfere = np.fft.ifft2(np.fft.ifftshift(img_freq_interfere))
    img_interfere_abs = np.abs(img_interfere)
    
    # Compute visibility
    img_interfere_vis = img_interfere_abs / img_0th_abs * 2
    
    return img_interfere_vis


# ==================== Gaussian Background Subtraction ====================

def gaussian(x, amp, mean, std):
    """Gaussian function"""
    return amp * np.exp(-((x - mean)**2) / (2 * std**2))


def gaussian_background_subtraction(phase_image, 
                                   hist_min=-1.1, hist_max=1.5, 
                                   n_bins=512, smooth_window=20,
                                   min_phase=-1.1):
    """
    Gaussian background subtraction for phase images

    Detect the histogram peak, fit a Gaussian, and shift to zero

    Parameters
    ----------
    phase_image : np.ndarray
        Phase image (rad)
    hist_min : float
        Histogram minimum value
    hist_max : float
        Histogram maximum value
    n_bins : int
        Number of histogram bins
    smooth_window : int
        Smoothing window size
    min_phase : float
        Lower bound for peak detection

    Returns
    -------
    corrected_image : np.ndarray
        Corrected phase image
    correction_value : float
        Correction value (rad)
    """
    # Create histogram
    bin_edges = np.linspace(hist_min, hist_max, n_bins + 1)
    hist_counts, _ = np.histogram(phase_image.flatten(), bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Smoothing
    smoothed_histo = uniform_filter1d(hist_counts, size=smooth_window, mode='nearest')
    smoothed_histo = uniform_filter1d(smoothed_histo, size=smooth_window, mode='nearest')
    
    # Peak detection
    valid_indices = np.where(bin_centers >= min_phase)[0]
    max_search_idx = int(len(bin_centers) * 0.95)
    search_indices = valid_indices[valid_indices < max_search_idx]
    
    search_range = smoothed_histo[search_indices]
    peak_idx_relative = np.argmax(search_range)
    peak_idx = search_indices[peak_idx_relative]
    peak_value = bin_centers[peak_idx]
    
    # Gaussian fit
    fit_width = 300
    start_idx = max(0, peak_idx - fit_width)
    end_idx = min(len(bin_centers), peak_idx + fit_width)
    
    x_data = bin_centers[start_idx:end_idx]
    y_data = smoothed_histo[start_idx:end_idx]
    
    # Initial estimates
    p0 = [np.max(y_data), peak_value, (bin_centers[1] - bin_centers[0]) * 20]
    
    popt, _ = curve_fit(gaussian, x_data, y_data, p0=p0, maxfev=5000)
    amp, mean, std = popt
    
    # Background correction (shift peak to 0)
    correction_value = 0.0 - mean
    corrected_image = phase_image + correction_value
    
    return corrected_image, correction_value


# ==================== Utility Functions ====================

def visualize_crop_region(image, crop_coords, title="Crop Region Visualization"):
    """
    Visualize the crop region on the original image

    Parameters
    ----------
    image : np.ndarray
        Original image
    crop_coords : tuple
        (y_start, y_end, x_start, x_end)
    title : str
        Title
    """
    y_s, y_e, x_s, x_e = crop_coords
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(image, cmap='gray')
    
    # Draw crop region as rectangle
    from matplotlib.patches import Rectangle
    rect = Rectangle((x_s, y_s), x_e - x_s, y_e - y_s, 
                     linewidth=3, edgecolor='red', facecolor='none',
                     label=f'Crop Region\n({y_s}:{y_e}, {x_s}:{x_e})')
    ax.add_patch(rect)
    
    # Add corner points
    ax.plot([x_s, x_e, x_e, x_s], [y_s, y_s, y_e, y_e], 
           'ro', markersize=8)
    
    # Display region size
    width = x_e - x_s
    height = y_e - y_s
    ax.text(x_s + width/2, y_s - 20, f'Width: {width}px', 
           ha='center', color='red', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(x_s - 20, y_s + height/2, f'Height: {height}px', 
           ha='right', va='center', color='red', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
           rotation=90)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12)
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    
    plt.tight_layout()
    return fig


def save_with_colormap(data, save_path, cmap="viridis", vmin=None, vmax=None,
                       colorbar_label="", title="", dpi=300):
    """
    Save as PNG image with colormap

    Parameters
    ----------
    data : np.ndarray
        Data to save
    save_path : str
        Output path
    cmap : str, optional
        Colormap name, by default "viridis"
    vmin : float, optional
        Minimum value, by default None
    vmax : float, optional
        Maximum value, by default None
    colorbar_label : str, optional
        Colorbar label, by default ""
    title : str, optional
        Title, by default ""
    dpi : int, optional
        Resolution, by default 300
    """
    plt.figure(figsize=(6, 6))
    im = plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, label=colorbar_label)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def to_uint8(img):
    """
    Convert float image to uint8 (for OpenCV)

    Parameters
    ----------
    img : np.ndarray
        Input image (float)

    Returns
    -------
    np.ndarray
        uint8 image
    """
    img_min = np.min(img)
    img_max = np.max(img)
    if img_max - img_min > 0:
        normalized = (img - img_min) / (img_max - img_min)
    else:
        normalized = img
    return (normalized * 255).astype(np.uint8)


def create_qpi_params(img_shape, offaxis_center=OFFAXIS_CENTER_DEFAULT,
                      wavelength=WAVELENGTH, NA=NA, pixelsize=PIXELSIZE):
    """
    Helper function to create a QPIParameters object

    Parameters
    ----------
    img_shape : tuple
        Image shape
    offaxis_center : tuple, optional
        Off-axis center coordinates, by default OFFAXIS_CENTER_DEFAULT
    wavelength : float, optional
        Wavelength, by default WAVELENGTH
    NA : float, optional
        Numerical aperture, by default NA
    pixelsize : float, optional
        Pixel size, by default PIXELSIZE

    Returns
    -------
    QPIParameters
        QPI parameters object
    """
    return QPIParameters(
        wavelength=wavelength,
        NA=NA,
        img_shape=img_shape,
        pixelsize=pixelsize,
        offaxis_center=offaxis_center
    )
















