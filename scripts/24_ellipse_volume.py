#!/usr/bin/env python3
"""
Density map workflow for time-series images.

Automatically matches each ROI to its corresponding image,
even when the number of images differs from the number of rows in Results.csv.
"""
# %% 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from PIL import Image
import os
import glob
import re
import tifffile
from scipy import ndimage

class TimeSeriesDensityMapper:
    """Generate refractive index (RI) maps from time-series images and Results.csv"""

    def __init__(self, results_csv, image_directory,
                 wavelength_nm=663, n_medium=1.333, pixel_size_um=0.348,
                 alpha_ri=0.0018, shape_type='ellipse', subpixel_sampling=5,
                 thickness_mode='continuous', voxel_z_um=0.3, discretize_method='round',
                 min_thickness_px=0.0, csv_suffix=None):
        """
        Parameters:
        -----------
        results_csv : str
            Path to ImageJ Results.csv
        image_directory : str
            Directory containing time-series images
        wavelength_nm : float
            Wavelength (nanometers). Default: 663nm
        n_medium : float
            Refractive index of the medium. Default: 1.333
        pixel_size_um : float
            Pixel size (micrometers). Default: 0.348 um
            For 507x507 reconstructed images: 0.08625 x (2048/507) = 0.348 um/pixel
        alpha_ri : float
            Specific refractive index increment [ml/mg]
            Default: 0.0018 ml/mg (typical value for proteins)
        shape_type : str
            ROI shape approximation method. Default: 'ellipse'
            'ellipse': Use Major/Minor/Angle (ellipse approximation)
            'feret': Use Feret/MinFeret/FeretAngle (Feret diameter approximation)
        subpixel_sampling : int
            Number of subpixel samples per pixel (NxN). Default: 5
            1: Pixel center only (fast but low accuracy at edges)
            5: 5x5 subpixels (recommended, good balance)
            10: 10x10 subpixels (high accuracy but slow)
        thickness_mode : str
            Thickness map mode. Default: 'continuous'
            'continuous': Continuous thickness values (in pixel units)
            'discrete': Discrete Z-stack slice counts
        voxel_z_um : float
            Voxel size in Z direction (micrometers). Default: 0.3 um
            Used to convert thickness to slice counts in discrete mode
        discretize_method : str
            Discretization method (only for discrete mode). Default: 'round'
            'round': Round to nearest integer
            'ceil': Round up
            'floor': Round down
            'pomegranate': Pomegranate-compatible threshold-based decision
        min_thickness_px : float
            Minimum thickness threshold (in pixel units). Default: 0.0
            Pixels with thickness below this value are ignored (set to 0)
            e.g.: 1.0 -> ignore thickness below 1 pixel
        csv_suffix : str, optional
            Suffix appended to the output folder name. Default: None
            If None, automatically extracted from CSV filename (e.g.: Results_enlarge.csv -> enlarge)
            For manual specification: 'enlarge', 'interpolate', 'custom_name', etc.

        Note:
        -----
        Phase-to-refractive-index conversion:
        phi = (2*pi/lambda) x (n_sample - n_medium) x thickness
        n_sample = n_medium + (phi x lambda) / (2*pi x thickness)

        Refractive-index-to-mass-concentration conversion:
        C [mg/ml] = (RI - RI_medium) / alpha

        Subpixel sampling:
        To improve accuracy at mask edges, each pixel is divided into NxN
        subpixels and thickness is averaged at subpixel centers.
        """
        self.results_csv = results_csv
        self.image_directory = image_directory
        self.wavelength_nm = wavelength_nm
        self.n_medium = n_medium
        self.pixel_size_um = pixel_size_um
        self.alpha_ri = alpha_ri
        self.shape_type = shape_type
        self.subpixel_sampling = subpixel_sampling
        self.thickness_mode = thickness_mode
        self.voxel_z_um = voxel_z_um
        self.discretize_method = discretize_method
        self.min_thickness_px = min_thickness_px
        
        # Determine CSV suffix (manual specification or auto-extraction)
        if csv_suffix is not None:
            self.csv_suffix = csv_suffix
        else:
            # Auto-extract from CSV filename (e.g.: Results_enlarge.csv -> enlarge)
            csv_filename = os.path.basename(results_csv)
            csv_name_without_ext = os.path.splitext(csv_filename)[0]  # Results_enlarge
            # Get the part after "Results_" (if any)
            if '_' in csv_name_without_ext:
                parts = csv_name_without_ext.split('_', 1)  # ['Results', 'enlarge']
                if len(parts) > 1 and parts[1]:
                    self.csv_suffix = parts[1]
                else:
                    self.csv_suffix = None
            else:
                self.csv_suffix = None
        
        # Unit conversion
        self.wavelength_um = wavelength_nm / 1000.0  # nm -> um
        
        # Load Results.csv
        print(f"Loading Results.csv: {results_csv}")
        self.df = pd.read_csv(results_csv)
        print(f"  Found {len(self.df)} ROIs")
        
        # Extract frame numbers
        self._extract_frame_numbers()

        # Scan image files
        self._scan_image_files()

        # Output directory (named according to parameters)
        if self.csv_suffix:
            base_dir_suffix = f"{self.shape_type}_subpixel{self.subpixel_sampling}_{self.csv_suffix}"
        else:
            base_dir_suffix = f"{self.shape_type}_subpixel{self.subpixel_sampling}"
        
        # Include additional info when thickness_mode is discrete
        if self.thickness_mode == 'discrete':
            self.dir_suffix = f"{base_dir_suffix}_discrete_{self.discretize_method}"
        else:
            self.dir_suffix = base_dir_suffix
        
        self.output_dir = f"timeseries_density_output_{self.dir_suffix}"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "density_tiff"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "visualizations"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "csv_data"), exist_ok=True)
        
        print(f"Output directory: {self.output_dir}")
        
    def _extract_frame_numbers(self):
        """Extract frame numbers from Results.csv"""
        print("\nExtracting frame numbers from Results.csv...")
        
        # Use Slice column (most reliable)
        if 'Slice' in self.df.columns:
            self.df['frame_number'] = self.df['Slice']
            print(f"  Using 'Slice' column")
        else:
            # Extract from Label
            print(f"  Extracting from 'Label' column...")
            def extract_frame(label):
                match = re.search(r'output_phase(\d+)', label)
                if match:
                    return int(match.group(1))
                return None
            
            self.df['frame_number'] = self.df['Label'].apply(extract_frame)
        
        # Frame number statistics
        unique_frames = self.df['frame_number'].dropna().unique()
        print(f"  Frame range: {unique_frames.min():.0f} to {unique_frames.max():.0f}")
        print(f"  Number of unique frames: {len(unique_frames)}")
        
    def _scan_image_files(self):
        """Scan image files and create frame number to path mapping"""
        print(f"\nScanning image files in: {self.image_directory}")
        
        # Search for image files
        patterns = [
            os.path.join(self.image_directory, "*.tif"),
            os.path.join(self.image_directory, "*.tiff"),
            os.path.join(self.image_directory, "*.png"),
        ]
        
        image_files = []
        for pattern in patterns:
            image_files.extend(glob.glob(pattern))
        
        if not image_files:
            raise ValueError(f"No images found in {self.image_directory}")
        
        print(f"  Found {len(image_files)} image files")
        
        # Extract frame numbers and create mapping
        self.frame_to_path = {}
        
        for img_path in image_files:
            filename = os.path.basename(img_path)
            
            # Extract part like "output_phase0001"
            match = re.search(r'output_phase(\d+)', filename)
            if match:
                frame_num = int(match.group(1))
                self.frame_to_path[frame_num] = img_path
        
        print(f"  Mapped {len(self.frame_to_path)} frames to image files")
        
        if len(self.frame_to_path) == 0:
            # If frame numbers not found, treat as sequential numbering
            print("  WARNING: Could not extract frame numbers from filenames")
            print("  Using sequential numbering instead...")
            sorted_files = sorted(image_files)
            for i, img_path in enumerate(sorted_files, start=1):
                self.frame_to_path[i] = img_path
        
        # Display frame number range
        frame_numbers = sorted(self.frame_to_path.keys())
        print(f"  Image frame range: {frame_numbers[0]} to {frame_numbers[-1]}")
        
    def load_image(self, frame_number):
        """Load the image for the specified frame number"""
        if frame_number not in self.frame_to_path:
            raise ValueError(f"Frame {frame_number} not found in image files")
        
        img_path = self.frame_to_path[frame_number]
        image = np.array(Image.open(img_path)).astype(np.float64)
        
        return image, img_path
    
    def create_rod_zstack_map(self, roi_params, image_shape, shape_type='ellipse', subpixel_sampling=5):
        """
        Generate z-stack count map for ROI (with subpixel sampling support).

        Parameters:
        -----------
        roi_params : dict
            ROI parameters
        image_shape : tuple
            Image size (height, width)
        shape_type : str
            'ellipse': Use Major/Minor/Angle (ellipse approximation)
            'feret': Use Feret/MinFeret/FeretAngle (Feret diameter approximation)
        subpixel_sampling : int
            Number of subpixel samples per pixel (NxN grid)
            1: Pixel center only (fast but low accuracy)
            5: 5x5 subpixels (recommended, good balance)
            10: 10x10 subpixels (high accuracy but slow)

        Returns:
        --------
        zstack_map : 2D numpy array
            Z-stack count map (average thickness within each pixel)

        Note:
        -----
        Subpixel sampling improves accuracy at mask edges.
        Each pixel is divided into NxN subpixels, thickness is calculated
        at each subpixel center, and the average is taken. This provides
        accurate mean thickness even for pixels partially overlapping the mask.
        """
        # Get parameters (switch according to shape_type)
        center_x = roi_params['X']
        center_y = roi_params['Y']
        
        if shape_type == 'feret':
            # Use Feret diameter
            length = roi_params.get('Feret', roi_params.get('Major'))
            width = roi_params.get('MinFeret', roi_params.get('Minor'))
            angle = roi_params.get('FeretAngle', roi_params.get('Angle'))
        else:  # 'ellipse' (default)
            # Use ellipse parameters
            length = roi_params['Major']
            width = roi_params['Minor']
            angle = roi_params['Angle']
        
        # Rod shape parameters
        r = width / 2.0
        h = length - 2 * r
        
        if h < 0:
            h = 0
        
        # ImageJ coordinate system compatibility
        angle_rad = np.deg2rad(-angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Z-stack count map
        img_height, img_width = image_shape
        zstack_map = np.zeros((img_height, img_width), dtype=np.float64)
        
        # Calculate subpixel offsets
        if subpixel_sampling > 1:
            # Center coordinate offsets for NxN subpixels
            offsets = np.linspace(0.5/subpixel_sampling, 1 - 0.5/subpixel_sampling, subpixel_sampling) - 0.5
        else:
            # No subpixel (pixel center only)
            offsets = np.array([0.0])
        
        # Calculate z-stack count for each pixel (subpixel sampling)
        for py in range(img_height):
            for px in range(img_width):
                thickness_sum = 0.0
                valid_subpixels = 0
                
                # Calculate thickness for each subpixel
                for dy_offset in offsets:
                    for dx_offset in offsets:
                        # Subpixel center coordinates
                        px_sub = px + 0.5 + dx_offset
                        py_sub = py + 0.5 + dy_offset
                        
                        dx = px_sub - center_x
                        dy = py_sub - center_y
                        
                        # Transform to local coordinate system
                        x_local = dx * cos_a + dy * sin_a
                        y_local = -dx * sin_a + dy * cos_a
                        
                        dist_from_axis = abs(y_local)
                        
                        if dist_from_axis > r:
                            continue
                        
                        # Thickness in z direction
                        z_half = np.sqrt(r**2 - y_local**2)
                        thickness = 2 * z_half
                        
                        # Position along major axis
                        if abs(x_local) <= h / 2.0:
                            # Cylinder section
                            thickness_sum += thickness
                            valid_subpixels += 1
                        else:
                            # Hemisphere section
                            if x_local > 0:
                                x_from_sphere_center = x_local - h / 2.0
                            else:
                                x_from_sphere_center = x_local + h / 2.0
                            
                            dist_sq = x_from_sphere_center**2 + y_local**2
                            
                            if dist_sq <= r**2:
                                z_half_sphere = np.sqrt(r**2 - dist_sq)
                                thickness_sphere = 2 * z_half_sphere
                                thickness_sum += thickness_sphere
                                valid_subpixels += 1
                
                # Calculate average thickness within pixel
                if valid_subpixels > 0:
                    zstack_map[py, px] = thickness_sum / valid_subpixels
        
        return zstack_map
    
    def _discretize_thickness(self, z_continuous_px):
        """
        Convert continuous Z-direction thickness (in pixel units) to discrete slice counts.

        Parameters
        ----------
        z_continuous_px : float or np.ndarray
            Continuous Z-direction thickness (in pixel units)

        Returns
        -------
        int or np.ndarray
            Discretized slice count
        """
        if self.voxel_z_um <= 0:
            # If voxel_z_um is invalid, return the original value
            if isinstance(z_continuous_px, np.ndarray):
                return z_continuous_px.astype(int)
            return int(z_continuous_px)
        
        # Convert thickness from pixel units to um units
        z_um = z_continuous_px * self.pixel_size_um
        
        if self.discretize_method == 'round':
            z_slices = np.round(z_um / self.voxel_z_um)
        elif self.discretize_method == 'ceil':
            z_slices = np.ceil(z_um / self.voxel_z_um)
        elif self.discretize_method == 'floor':
            z_slices = np.floor(z_um / self.voxel_z_um)
        elif self.discretize_method == 'pomegranate':
            # Pomegranate-style discretization
            # Minimum radius threshold (set to 2x pixel size here)
            min_radius_threshold_um = 2.0 * self.pixel_size_um
            
            # Branch processing for array vs scalar
            if isinstance(z_um, np.ndarray):
                num_z_voxels_float = z_um / self.voxel_z_um
                z_slices = np.zeros_like(num_z_voxels_float)
                
                # Process only elements exceeding threshold
                mask = z_um > min_radius_threshold_um
                z_slices[mask] = np.round(num_z_voxels_float[mask])
                
                # Ensure at least 1 slice (when exceeding threshold)
                small_mask = (num_z_voxels_float > 0) & (z_slices == 0) & mask
                z_slices[small_mask] = 1
            else:
                if z_um > min_radius_threshold_um:
                    num_z_voxels_float = z_um / self.voxel_z_um
                    z_slices = np.round(num_z_voxels_float)
                    if z_slices == 0 and num_z_voxels_float > 0:
                        z_slices = 1
                else:
                    z_slices = 0
        else:
            # Default is round
            z_slices = np.round(z_um / self.voxel_z_um)
        
        # Clip negative values to 0
        z_slices = np.maximum(0, z_slices)
        
        # Convert to integer
        if isinstance(z_slices, np.ndarray):
            return z_slices.astype(int)
        return int(z_slices)
    
    def process_roi(self, roi_index):
        """
        Process a specific ROI.

        Parameters:
        -----------
        roi_index : int
            ROI index in Results.csv

        Returns:
        --------
        results : dict
            Processing results
        """
        row = self.df.iloc[roi_index]
        frame_number = int(row['frame_number'])
        
        print(f"\n{'='*70}")
        print(f"Processing ROI {roi_index} (Frame {frame_number})")
        print(f"{'='*70}")
        
        # Load image
        try:
            image, img_path = self.load_image(frame_number)
            print(f"Loaded image: {os.path.basename(img_path)}")
            print(f"  Image shape: {image.shape}")
        except ValueError as e:
            print(f"  ERROR: {e}")
            return None
        
        # ROI parameters
        roi_params = {
            'X': row['X'],
            'Y': row['Y'],
            'Major': row['Major'],
            'Minor': row['Minor'],
            'Angle': row['Angle'],
        }
        
        # Add Feret diameter parameters (if available)
        if 'Feret' in row:
            roi_params['Feret'] = row['Feret']
        if 'MinFeret' in row:
            roi_params['MinFeret'] = row['MinFeret']
        if 'FeretAngle' in row:
            roi_params['FeretAngle'] = row['FeretAngle']
        
        print(f"  ROI parameters:")
        print(f"    Center: ({roi_params['X']:.2f}, {roi_params['Y']:.2f})")
        if self.shape_type == 'feret' and 'Feret' in roi_params:
            print(f"    Feret: {roi_params['Feret']:.2f} pixels")
            print(f"    MinFeret: {roi_params.get('MinFeret', 'N/A'):.2f} pixels" if 'MinFeret' in roi_params else "    MinFeret: N/A")
            print(f"    FeretAngle: {roi_params.get('FeretAngle', roi_params['FeretAngle']):.2f}°" if 'FeretAngle' in roi_params else f"    FeretAngle: {roi_params['Angle']:.2f}°")
        else:
            print(f"    Major: {roi_params['Major']:.2f} pixels")
            print(f"    Minor: {roi_params['Minor']:.2f} pixels")
            print(f"    Angle: {roi_params['Angle']:.2f}°")
        
        # Generate ROI identifier (for cache filename)
        roi_id = f"frame{frame_number:04d}"
        
        # Generate or load z-stack count map
        # Generate cache filename
        cache_dir = os.path.join(self.output_dir, 'thickness_cache')
        cache_path = os.path.join(cache_dir, f"roi_{roi_index:04d}_{roi_id}.npz")
        
        # Load from cache if in discrete mode and cache exists
        if self.thickness_mode == 'discrete' and os.path.exists(cache_path):
            print(f"  Loading cached thickness map from: {os.path.basename(cache_path)}")
            cached_data = np.load(cache_path)
            zstack_map_continuous = cached_data['thickness_map_continuous']
            print(f"    Loaded shape: {zstack_map_continuous.shape}")
        else:
            # Calculate from scratch
            print(f"  Generating z-stack map (shape_type: {self.shape_type}, subpixel: {self.subpixel_sampling}×{self.subpixel_sampling})...")
            print(f"    Thickness mode: {self.thickness_mode}")
            if self.thickness_mode == 'discrete':
                print(f"    Discretize method: {self.discretize_method}")
                print(f"    Voxel Z size: {self.voxel_z_um} µm")
            
            zstack_map_continuous = self.create_rod_zstack_map(roi_params, image.shape, 
                                                                 shape_type=self.shape_type,
                                                                 subpixel_sampling=self.subpixel_sampling)
            
            # Save to cache in continuous mode
            if self.thickness_mode == 'continuous':
                os.makedirs(cache_dir, exist_ok=True)
                np.savez_compressed(cache_path, 
                                    thickness_map_continuous=zstack_map_continuous,
                                    roi_id=roi_id,
                                    roi_index=roi_index)
                print(f"    Saved thickness cache: {os.path.basename(cache_path)}")
        
        # Determine zstack_map according to thickness_mode
        if self.thickness_mode == 'discrete':
            # Discretized slice counts
            zstack_map = self._discretize_thickness(zstack_map_continuous)
        else:
            # Continuous values (thickness in pixel units)
            zstack_map = zstack_map_continuous
        
        # Minimum thickness threshold filtering (evaluated in pixel units)
        if self.min_thickness_px > 0:
            # Direct comparison in continuous mode, converted comparison in discrete mode
            if self.thickness_mode == 'discrete':
                # Convert slice count to pixel units for threshold evaluation
                thickness_px_for_threshold = zstack_map * (self.voxel_z_um / self.pixel_size_um)
            else:
                thickness_px_for_threshold = zstack_map
            
            # Set values below threshold to 0
            pixels_before = np.count_nonzero(zstack_map > 0)
            zstack_map = np.where(thickness_px_for_threshold >= self.min_thickness_px, zstack_map, 0)
            pixels_after = np.count_nonzero(zstack_map > 0)
            
            if pixels_before > pixels_after:
                print(f"    Min thickness filter: {self.min_thickness_px:.2f} px")
                print(f"    Filtered pixels: {pixels_before - pixels_after} ({(pixels_before - pixels_after) / pixels_before * 100:.1f}%)")
        
        mask = zstack_map > 0
        if not np.any(mask):
            print("  WARNING: No pixels in ROI!")
            return None
        
        print(f"    Max z-stack: {zstack_map.max():.2f}")
        print(f"    Mean z-stack: {zstack_map[mask].mean():.2f}")
        print(f"    Non-zero pixels: {np.count_nonzero(mask)}")
        
        # Calculate refractive index (RI) from phase
        print(f"  Converting phase to refractive index...")
        print(f"    Wavelength: {self.wavelength_nm} nm")
        print(f"    Medium RI: {self.n_medium}")
        print(f"    Pixel size: {self.pixel_size_um} µm")
        
        # Convert thickness to um units (according to thickness_mode)
        if self.thickness_mode == 'discrete':
            # Discrete mode: slice count x Z-direction voxel size
            thickness_um = zstack_map * self.voxel_z_um
        else:
            # Continuous mode: thickness in pixel units x XY pixel size
            thickness_um = zstack_map * self.pixel_size_um

        # Initialize RI map (with medium refractive index)
        ri_map = np.full_like(image, self.n_medium, dtype=np.float64)
        
        # Phase-to-refractive-index conversion
        # n_sample = n_medium + (phi x lambda) / (2*pi x thickness)
        # Here, image is assumed to be proportional to phase phi
        # In practice, verify the image units and apply appropriate scaling
        ri_map[mask] = self.n_medium + (image[mask] * self.wavelength_um) / (2 * np.pi * thickness_um[mask])
        
        print(f"    RI range: [{ri_map[mask].min():.6f}, {ri_map[mask].max():.6f}]")
        print(f"    Mean RI: {ri_map[mask].mean():.6f}")
        print(f"    ΔRI from medium: {(ri_map[mask].mean() - self.n_medium):.6f}")
        
        # Calculate mass concentration map (mg/ml)
        # C = (RI - RI_medium) / α
        concentration_map = np.zeros_like(image, dtype=np.float64)
        concentration_map[mask] = (ri_map[mask] - self.n_medium) / self.alpha_ri
        
        print(f"  Calculating mass concentration...")
        print(f"    Concentration range: [{concentration_map[mask].min():.2f}, {concentration_map[mask].max():.2f}] mg/ml")
        print(f"    Mean concentration: {concentration_map[mask].mean():.2f} mg/ml")
        
        # Calculate volume
        # Volume = sum(thickness per pixel [um] x pixel area [um^2])
        pixel_area_um2 = self.pixel_size_um ** 2
        volume_um3 = np.sum(thickness_um[mask]) * pixel_area_um2
        
        print(f"  Calculating volume...")
        print(f"    Volume: {volume_um3:.2f} µm³")
        print(f"    Mean thickness: {thickness_um[mask].mean():.4f} µm")
        print(f"    Max thickness: {thickness_um[mask].max():.4f} µm")
        
        # Calculate total mass
        # Total mass [pg] = sum(concentration [mg/ml] x pixel_volume [um^3])
        # Unit conversion: 1 mg/ml = 1 pg/um^3
        pixel_volumes = thickness_um[mask] * pixel_area_um2  # volume per pixel [um^3]
        total_mass_pg = np.sum(concentration_map[mask] * pixel_volumes)  # [pg]
        
        print(f"  Calculating total mass...")
        print(f"    Total mass: {total_mass_pg:.2f} pg")
        print(f"    Mean concentration: {concentration_map[mask].mean():.2f} mg/ml")
        
        # Package results
        results = {
            'roi_index': roi_index,
            'frame_number': frame_number,
            'image_path': img_path,
            'image': image,
            'zstack_map': zstack_map,
            'ri_map': ri_map,  # refractive index map
            'concentration_map': concentration_map,  # mass concentration map (mg/ml)
            'roi_params': roi_params,
            'stats': {
                'zstack_max': float(zstack_map.max()),
                'zstack_mean': float(zstack_map[mask].mean()),
                'thickness_mean_um': float(thickness_um[mask].mean()),
                'thickness_max_um': float(thickness_um[mask].max()),
                'volume_um3': float(volume_um3),
                'total_mass_pg': float(total_mass_pg),
                'ri_mean': float(ri_map[mask].mean()),
                'concentration_mean': float(concentration_map[mask].mean()),
                'concentration_min': float(concentration_map[mask].min()),
                'concentration_max': float(concentration_map[mask].max()),
                'ri_min': float(ri_map[mask].min()),
                'ri_max': float(ri_map[mask].max()),
                'ri_delta': float(ri_map[mask].mean() - self.n_medium),  # difference from medium
                'num_pixels': int(np.count_nonzero(mask)),
            }
        }
        
        return results
    
    def save_results(self, results):
        """Save results"""
        if results is None:
            return
        
        roi_index = results['roi_index']
        frame_number = results['frame_number']
        roi_str = f"ROI_{roi_index:04d}_Frame_{frame_number:04d}"
        
        # Save TIFF
        ri_tiff = os.path.join(self.output_dir, "density_tiff", f"{roi_str}_ri.tif")
        concentration_tiff = os.path.join(self.output_dir, "density_tiff", f"{roi_str}_concentration.tif")
        zstack_tiff = os.path.join(self.output_dir, "density_tiff", f"{roi_str}_zstack.tif")
        phase_tiff = os.path.join(self.output_dir, "density_tiff", f"{roi_str}_phase.tif")
        
        tifffile.imwrite(ri_tiff, results['ri_map'].astype(np.float32))
        tifffile.imwrite(concentration_tiff, results['concentration_map'].astype(np.float32))
        tifffile.imwrite(zstack_tiff, results['zstack_map'].astype(np.float32))
        tifffile.imwrite(phase_tiff, results['image'].astype(np.float32))  # original phase image
        
        print(f"\nSaved: {os.path.basename(ri_tiff)}")
        print(f"Saved: {os.path.basename(concentration_tiff)}")
        print(f"Saved: {os.path.basename(zstack_tiff)}")
        print(f"Saved: {os.path.basename(phase_tiff)}")
        
        # Save CSV
        csv_path = os.path.join(self.output_dir, "csv_data", f"{roi_str}_pixel_data.csv")
        
        mask = results['zstack_map'] > 0
        y_coords, x_coords = np.where(mask)
        
        # Convert thickness to um units (according to thickness_mode)
        if self.thickness_mode == 'discrete':
            # Discrete mode: slice count x Z-direction voxel size
            thickness_um_map = results['zstack_map'][mask] * self.voxel_z_um
            z_column_name = 'Z_slice_count'
        else:
            # Continuous mode: thickness in pixel units x XY pixel size
            thickness_um_map = results['zstack_map'][mask] * self.pixel_size_um
            z_column_name = 'Z_thickness_pixel'
        
        pixel_data = pd.DataFrame({
            'X_pixel': x_coords,
            'Y_pixel': y_coords,
            z_column_name: results['zstack_map'][mask],
            'Thickness_um': thickness_um_map,
            'Phase_value': results['image'][mask],
            'Refractive_Index': results['ri_map'][mask],
            'Delta_RI': results['ri_map'][mask] - self.n_medium,
            'Concentration_mg_ml': results['concentration_map'][mask],
        })
        pixel_data.to_csv(csv_path, index=False)
        print(f"Saved: {os.path.basename(csv_path)}")
        
        # Parameters CSV
        params_path = os.path.join(self.output_dir, "csv_data", f"{roi_str}_parameters.csv")
        params_dict = {
            'roi_index': roi_index,
            'frame_number': frame_number,
            'image_path': results['image_path'],
            **results['roi_params'],
            **results['stats'],
            'wavelength_nm': self.wavelength_nm,
            'n_medium': self.n_medium,
            'pixel_size_um': self.pixel_size_um,
            'alpha_ri': self.alpha_ri,
        }
        params_df = pd.DataFrame([params_dict])
        params_df.to_csv(params_path, index=False)
        print(f"Saved: {os.path.basename(params_path)}")
        
        # Visualization
        self.create_visualization(results, roi_str)
    
    def create_visualization(self, results, roi_str):
        """Create visualization figure"""
        fig = plt.figure(figsize=(26, 12))
        gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
        
        image = results['image']
        zstack_map = results['zstack_map']
        ri_map = results['ri_map']  # refractive index map
        concentration_map = results['concentration_map']  # mass concentration map
        roi_params = results['roi_params']
        stats = results['stats']
        
        mask = zstack_map > 0
        
        # 1. Original image + ROI + mask contour
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(image, cmap='gray',vmin=-0.5,vmax=2.5)
        ax1.set_title(f'Original Image + Mask Contour\nFrame {results["frame_number"]}', 
                      fontsize=12, fontweight='bold')
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        plt.colorbar(im1, ax=ax1, label='Intensity')
        
        # Draw mask contour (region where zstack > 0)
        from skimage import measure
        binary_mask = (zstack_map > 0).astype(np.uint8)
        contours = measure.find_contours(binary_mask, 0.5)
        for contour in contours:
            ax1.plot(contour[:, 1], contour[:, 0], 'c-', linewidth=2, alpha=0.8)
        

        # 2. Z-stack map (thickness map)
        ax2 = fig.add_subplot(gs[0, 1])
        thickness_um = zstack_map * self.pixel_size_um
        im2 = ax2.imshow(thickness_um, cmap='viridis',vmin=0,vmax=12)
        ax2.set_title(f'Thickness Map\n(max={stats["zstack_max"]*self.pixel_size_um:.2f} µm)', 
                      fontsize=12, fontweight='bold')
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')
        plt.colorbar(im2, ax=ax2, label='Thickness (µm)')
        
        # 3. Refractive index map (RI map)
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(ri_map, cmap='jet',
                        vmin=1.3, vmax=1.39)
        ax3.set_title(f'Refractive Index Map\n(mean={stats["ri_mean"]:.6f})', 
                      fontsize=12, fontweight='bold')
        ax3.set_xlabel('X (pixels)')
        ax3.set_ylabel('Y (pixels)')
        cbar3 = plt.colorbar(im3, ax=ax3, label='RI')
        # Add line indicating medium RI
        cbar3.ax.axhline(y=self.n_medium, color='cyan', linestyle='--', linewidth=2, alpha=0.8)
        
        # 4. Mass concentration map (mg/ml)
        ax4 = fig.add_subplot(gs[0, 3])
        im4 = ax4.imshow(concentration_map, cmap='hot',
                        vmin=0, vmax=450)
        ax4.set_title(f'Protein Concentration Map\n(mean={stats["concentration_mean"]:.1f} mg/ml)', 
                      fontsize=12, fontweight='bold')
        ax4.set_xlabel('X (pixels)')
        ax4.set_ylabel('Y (pixels)')
        plt.colorbar(im4, ax=ax4, label='Concentration (mg/ml)')
        
        # 5. Delta-RI map (difference from medium)
        ax5 = fig.add_subplot(gs[1, 0])
        delta_ri = ri_map - self.n_medium
        im5 = ax5.imshow(delta_ri, cmap='plasma',vmin=-0.1,vmax=0.3)
        ax5.set_title(f'ΔRI Map (sample - medium)\n(mean ΔRI={stats["ri_delta"]:.6f})', 
                      fontsize=12, fontweight='bold')
        ax5.set_xlabel('X (pixels)')
        ax5.set_ylabel('Y (pixels)')
        plt.colorbar(im5, ax=ax5, label='ΔRI')
        
        # 6. RI vs Thickness
        ax6 = fig.add_subplot(gs[1, 1])
        if np.any(mask):
            thickness_masked = thickness_um[mask]
            ri_masked = ri_map[mask]
            ax6.scatter(thickness_masked, ri_masked, alpha=0.3, s=10, c='blue')
            
            # Reference line for medium RI
            ax6.axhline(y=self.n_medium, color='cyan', linestyle='--', linewidth=2,
                       label=f'Medium RI: {self.n_medium:.3f}')
            # Reference line for mean RI
            ax6.axhline(y=stats['ri_mean'], color='red', linestyle='--', linewidth=2,
                       label=f'Mean RI: {stats["ri_mean"]:.6f}')
            
            ax6.set_xlabel('Thickness (µm)', fontsize=11)
            ax6.set_ylabel('Refractive Index', fontsize=11)
            ax6.set_title('RI vs Thickness', fontsize=12, fontweight='bold')
            ax6.set_xlim(0, 6)      # Thickness range: 0-6 µm
            ax6.set_ylim(1.3, 1.6)  # RI range: 1.3-1.6
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # 7. RI distribution
        ax7 = fig.add_subplot(gs[1, 2])
        if np.any(mask):
            ri_masked = ri_map[mask].flatten()
            ax7.hist(ri_masked, bins=50, alpha=0.7, color='purple', edgecolor='black', range=(1.30, 1.60))
            ax7.axvline(x=self.n_medium, color='cyan', linestyle='--', linewidth=2,
                       label=f'Medium: {self.n_medium:.3f}')
            ax7.axvline(x=stats['ri_mean'], color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {stats["ri_mean"]:.6f}')
            ax7.set_xlabel('Refractive Index', fontsize=11)
            ax7.set_ylabel('Frequency', fontsize=11)
            ax7.set_title('RI Distribution', fontsize=12, fontweight='bold')
            ax7.set_xlim(1.30, 1.60)  # Fix X-axis range
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        
        # 8. Mass concentration distribution
        ax8 = fig.add_subplot(gs[1, 3])
        if np.any(mask):
            concentration_masked = concentration_map[mask].flatten()
            ax8.hist(concentration_masked, bins=50, alpha=0.7, color='orange', edgecolor='black', range=(0, 450))
            ax8.axvline(x=stats['concentration_mean'], color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {stats["concentration_mean"]:.1f} mg/ml')
            ax8.set_xlabel('Concentration (mg/ml)', fontsize=11)
            ax8.set_ylabel('Frequency', fontsize=11)
            ax8.set_title('Protein Concentration Distribution', fontsize=12, fontweight='bold')
            ax8.set_xlim(0, 600)  # Fix X-axis range
            ax8.legend()
            ax8.grid(True, alpha=0.3)
        
        fig.suptitle(f'{roi_str} - QPI Analysis (V={stats["volume_um3"]:.1f}µm³, RI={stats["ri_mean"]:.4f}, C={stats["concentration_mean"]:.1f}mg/ml)', 
                     fontsize=16, fontweight='bold')
        
        viz_path = os.path.join(self.output_dir, "visualizations", f"{roi_str}_visualization.png")
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {os.path.basename(viz_path)}")
    
    def process_all_rois(self, max_rois=None):
        """
        Process all ROIs.

        Parameters:
        -----------
        max_rois : int or None
            Maximum number of ROIs to process (None for all)
        """
        num_rois = len(self.df) if max_rois is None else min(max_rois, len(self.df))
        
        print(f"\n{'#'*80}")
        print(f"# Starting time-series RI analysis workflow")
        print(f"# Total ROIs: {num_rois}")
        print(f"# Wavelength: {self.wavelength_nm} nm")
        print(f"# Medium RI: {self.n_medium}")
        print(f"# Pixel size: {self.pixel_size_um} µm")
        print(f"{'#'*80}\n")
        
        all_results = []
        success_count = 0
        
        for i in range(num_rois):
            try:
                results = self.process_roi(i)
                
                if results is not None:
                    self.save_results(results)
                    all_results.append({
                        'roi_index': i,
                        'frame_number': results['frame_number'],
                        **results['roi_params'],
                        **results['stats'],
                    })
                    success_count += 1
                
            except Exception as e:
                print(f"\nERROR processing ROI {i}: {e}")
                import traceback
                traceback.print_exc()
        
        # Save summary
        if all_results:
            summary_path = os.path.join(self.output_dir, "all_rois_summary.csv")
            summary_df = pd.DataFrame(all_results)
            summary_df.to_csv(summary_path, index=False)
            print(f"\n{'='*80}")
            print(f"Saved summary: {summary_path}")
            print(f"{'='*80}")
        
        print(f"\n{'#'*80}")
        print(f"# Workflow completed!")
        print(f"# Successfully processed: {success_count}/{num_rois} ROIs")
        print(f"# Output directory: {self.output_dir}")
        print(f"{'#'*80}\n")


# ===== Main execution =====
if __name__ == "__main__":
    # Parameter settings
    # Use globals() for batch execution, default values for standalone execution
    RESULTS_CSV = r"C:\Users\QPI\Desktop\align_demo\from_outputphase\bg_corr\subtracted\inference_out\Results_enlarge.csv" if 'RESULTS_CSV' not in globals() else globals()['RESULTS_CSV']
    
    # Image directory (change according to your environment)
    # Windows path example: r"C:\Users\QPI\Desktop\align_demo\from_outputphase\bg_corr\subtracted"
    # Linux path example: "/home/user/images/subtracted"
    IMAGE_DIRECTORY = r"C:\Users\QPI\Desktop\align_demo\from_outputphase\bg_corr\subtracted" if 'IMAGE_DIRECTORY' not in globals() else globals()['IMAGE_DIRECTORY']
    
    # QPI experiment parameters (same values as 01_QPI_analysis.py)
    WAVELENGTH_NM = 658 if 'WAVELENGTH_NM' not in globals() else globals()['WAVELENGTH_NM']
                                 # Laser wavelength (nanometers)
                                 # Experimental value: 663nm (red laser)
    N_MEDIUM = 1.333 if 'N_MEDIUM' not in globals() else globals()['N_MEDIUM']
                                 # Medium refractive index
                                 # Water: 1.333, PBS: 1.334, DMEM: ~1.335
    PIXEL_SIZE_UM = 0.348 if 'PIXEL_SIZE_UM' not in globals() else globals()['PIXEL_SIZE_UM']
                                 # Pixel size (micrometers)
                                 # For 507x507 reconstructed images
                                 # Calculation: 0.08625 um x (2048/507) = 0.348 um/pixel
                                 # Note: Original hologram 2048x2048 has 0.08625 um/pixel
    ALPHA_RI = 0.00018 if 'ALPHA_RI' not in globals() else globals()['ALPHA_RI']
                                 # Specific refractive index increment [ml/mg]
                                 # Typical value for proteins: 0.0018 ml/mg
                                 # Reference: RI = RI_medium + alpha x C [mg/ml]

    SHAPE_TYPE = 'feret' if 'SHAPE_TYPE' not in globals() else globals()['SHAPE_TYPE']
                                 # ROI shape approximation method
                                 # 'ellipse': Major/Minor/Angle (ellipse approximation)
                                 # 'feret': Feret/MinFeret/FeretAngle (Feret diameter approximation)

    SUBPIXEL_SAMPLING = 5 if 'SUBPIXEL_SAMPLING' not in globals() else globals()['SUBPIXEL_SAMPLING']
                                 # Subpixel sampling count (NxN)
                                 # 1: Pixel center only (fast, low edge accuracy)
                                 # 5: 5x5 subpixels (recommended, good balance)
                                 # 10: 10x10 subpixels (high accuracy, slow)

    THICKNESS_MODE = 'continuous' if 'THICKNESS_MODE' not in globals() else globals()['THICKNESS_MODE']
                                 # Thickness map mode
                                 # 'continuous': Continuous thickness values (in pixel units)
                                 # 'discrete': Discrete Z-stack slice counts

    VOXEL_Z_UM = 0.3 if 'VOXEL_Z_UM' not in globals() else globals()['VOXEL_Z_UM']
                                 # Z-direction voxel size (micrometers)
                                 # Used to convert thickness to slice counts in discrete mode

    DISCRETIZE_METHOD = 'round' if 'DISCRETIZE_METHOD' not in globals() else globals()['DISCRETIZE_METHOD']
                                 # Discretization method (only for discrete mode)
                                 # 'round': Round to nearest integer
                                 # 'ceil': Round up
                                 # 'floor': Round down
                                 # 'pomegranate': Pomegranate-compatible threshold-based decision

    MIN_THICKNESS_PX = 0.0 if 'MIN_THICKNESS_PX' not in globals() else globals()['MIN_THICKNESS_PX']
                                 # Minimum thickness threshold (in pixel units)
                                 # 0.0: No threshold (include all)
                                 # 1.0: Ignore thickness below 1 pixel
                                 # 2.0: Ignore thickness below 2 pixels

    MAX_ROIS = None if 'MAX_ROIS' not in globals() else globals()['MAX_ROIS']  # Test run (None for all ROIs)

    # CSV suffix (for identifying output folder name)
    # None: Auto-extract from CSV filename (Results_enlarge.csv -> 'enlarge')
    # String: Manual specification (e.g.: 'custom_name')
    CSV_SUFFIX = None if 'CSV_SUFFIX' not in globals() else globals()['CSV_SUFFIX']
    
    print(f"\n{'='*70}")
    print(f"QPI Analysis Parameters:")
    print(f"  Wavelength: {WAVELENGTH_NM} nm")
    print(f"  Medium RI: {N_MEDIUM}")
    print(f"  Pixel size: {PIXEL_SIZE_UM} µm (for 507×507 reconstructed images)")
    print(f"  Alpha (RI increment): {ALPHA_RI} ml/mg")
    print(f"  Shape type: {SHAPE_TYPE}")
    print(f"  Subpixel sampling: {SUBPIXEL_SAMPLING}×{SUBPIXEL_SAMPLING}")
    print(f"  Thickness mode: {THICKNESS_MODE}")
    if THICKNESS_MODE == 'discrete':
        print(f"  Voxel Z size: {VOXEL_Z_UM} µm")
        print(f"  Discretize method: {DISCRETIZE_METHOD}")
    if MIN_THICKNESS_PX > 0:
        print(f"  Min thickness filter: {MIN_THICKNESS_PX} px")
    print(f"  Note: Concentration (mg/ml) = (RI - {N_MEDIUM}) / {ALPHA_RI}")
    print(f"{'='*70}\n")
    
    # Execute workflow
    mapper = TimeSeriesDensityMapper(
        results_csv=RESULTS_CSV,
        image_directory=IMAGE_DIRECTORY,
        wavelength_nm=WAVELENGTH_NM,
        n_medium=N_MEDIUM,
        pixel_size_um=PIXEL_SIZE_UM,
        alpha_ri=ALPHA_RI,
        shape_type=SHAPE_TYPE,
        subpixel_sampling=SUBPIXEL_SAMPLING,
        thickness_mode=THICKNESS_MODE,
        voxel_z_um=VOXEL_Z_UM,
        discretize_method=DISCRETIZE_METHOD,
        min_thickness_px=MIN_THICKNESS_PX,
        csv_suffix=CSV_SUFFIX
    )
    
    mapper.process_all_rois(max_rois=MAX_ROIS)
    
    # ===== Generate time-series plots =====
    print(f"\n{'#'*80}")
    print(f"# Generating time-series plots...")
    print(f"{'#'*80}\n")
    
    # Integrate functionality from 27_timeseries_plot.py
    from matplotlib.gridspec import GridSpec
    
    # Load all_rois_summary.csv
    summary_path = os.path.join(mapper.output_dir, "all_rois_summary.csv")
    
    if os.path.exists(summary_path):
        print(f"Loading summary data: {summary_path}")
        df_summary = pd.read_csv(summary_path)
        
        # Calculate time (frame_number / 12 = time [h])
        # Or use existing slice if available
        if 'Slice' in mapper.df.columns:
            df_summary['time_h'] = df_summary['frame_number'] / 12.0
        else:
            df_summary['time_h'] = df_summary['frame_number'] / 12.0
        
        print(f"  Time range: {df_summary['time_h'].min():.2f} - {df_summary['time_h'].max():.2f} h")
        print(f"  Number of data points: {len(df_summary)}")
        
        # Output directory for time-series plots
        plot_output_dir = f"timeseries_plots_{mapper.dir_suffix}"
        os.makedirs(plot_output_dir, exist_ok=True)
        
        # Create time bins
        time_bin_h = 1.0  # aggregate per 1 hour
        time_bins = np.arange(
            np.floor(df_summary['time_h'].min()),
            np.ceil(df_summary['time_h'].max()) + time_bin_h,
            time_bin_h
        )
        
        # Calculate statistics per time bin
        time_centers = []
        volume_means = []
        mass_means = []
        ri_means = []
        
        for i in range(len(time_bins) - 1):
            mask = (df_summary['time_h'] >= time_bins[i]) & (df_summary['time_h'] < time_bins[i+1])
            if np.any(mask):
                time_centers.append((time_bins[i] + time_bins[i+1]) / 2)
                
                if 'volume_um3' in df_summary.columns:
                    volume_means.append(df_summary.loc[mask, 'volume_um3'].mean())
                
                if 'total_mass_pg' in df_summary.columns:
                    mass_means.append(df_summary.loc[mask, 'total_mass_pg'].mean())
                
                if 'ri_mean' in df_summary.columns:
                    ri_means.append(df_summary.loc[mask, 'ri_mean'].mean())
        
        # Create plots
        fig = plt.figure(figsize=(14, 14))
        gs = GridSpec(3, 1, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Volume over time
        ax1 = fig.add_subplot(gs[0, 0])
        if 'volume_um3' in df_summary.columns:
            ax1.scatter(df_summary['time_h'], df_summary['volume_um3'], 
                       alpha=0.3, s=20, c='lightblue', label='Individual ROIs')
            if time_centers and volume_means:
                ax1.plot(time_centers, volume_means, '-', color='blue', 
                        linewidth=2.5, label='Mean', zorder=10)
            ax1.set_xlabel('Time [h]', fontsize=13, fontweight='bold')
            ax1.set_ylabel('Volume [µm³]', fontsize=13, fontweight='bold')
            ax1.set_title('Volume vs Time', fontsize=15, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
        
        # 2. Mean RI over time
        ax2 = fig.add_subplot(gs[1, 0])
        if 'ri_mean' in df_summary.columns:
            ax2.scatter(df_summary['time_h'], df_summary['ri_mean'], 
                       alpha=0.3, s=20, c='lightcoral', label='Individual ROIs')
            if time_centers and ri_means:
                ax2.plot(time_centers, ri_means, '-', color='red', 
                        linewidth=2.5, label='Mean', zorder=10)
            ax2.axhline(y=N_MEDIUM, color='cyan', linestyle='--', linewidth=2, 
                       alpha=0.7, label=f'Medium ({N_MEDIUM})')
            ax2.set_xlabel('Time [h]', fontsize=13, fontweight='bold')
            ax2.set_ylabel('Refractive Index (RI)', fontsize=13, fontweight='bold')
            ax2.set_title('Mean RI vs Time', fontsize=15, fontweight='bold')
            ax2.set_ylim(1.36, 1.40)
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
        
        # 3. Total mass over time
        ax3 = fig.add_subplot(gs[2, 0])
        if 'total_mass_pg' in df_summary.columns:
            ax3.scatter(df_summary['time_h'], df_summary['total_mass_pg'], 
                       alpha=0.3, s=20, c='lightyellow', label='Individual ROIs')
            if time_centers and mass_means:
                ax3.plot(time_centers, mass_means, '-', color='orange', 
                        linewidth=2.5, label='Mean', zorder=10)
            ax3.set_xlabel('Time [h]', fontsize=13, fontweight='bold')
            ax3.set_ylabel('Total Mass [pg]', fontsize=13, fontweight='bold')
            ax3.set_title('Total Mass vs Time', fontsize=15, fontweight='bold')
            ax3.legend(fontsize=10)
            ax3.grid(True, alpha=0.3)
        
        fig.suptitle(f'Time-Series Analysis: {SHAPE_TYPE}, {SUBPIXEL_SAMPLING}×{SUBPIXEL_SAMPLING} subpixel', 
                    fontsize=18, fontweight='bold', y=0.995)
        
        # Save
        plot_path = os.path.join(plot_output_dir, 'timeseries_volume_ri_mass.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nTime-series plot saved: {plot_path}")
        print(f"Time-series plot directory: {plot_output_dir}")
    else:
        print(f"WARNING: Summary file not found: {summary_path}")
        print("Skipping time-series plot generation.")
    
    print(f"\n{'#'*80}")
    print(f"# All workflows completed!")
    print(f"# Data output: {mapper.output_dir}")
    print(f"# Plots output: {plot_output_dir if os.path.exists(summary_path) else 'N/A'}")
    print(f"{'#'*80}\n")
    
    # Create completion flag file (for resume functionality)
    flag_file = os.path.join(mapper.output_dir, '.completed')
    with open(flag_file, 'w') as f:
        f.write(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Shape type: {SHAPE_TYPE}\n")
        f.write(f"Subpixel: {SUBPIXEL_SAMPLING}\n")
        f.write(f"Thickness mode: {THICKNESS_MODE}\n")
        if THICKNESS_MODE == 'discrete':
            f.write(f"Discretize method: {DISCRETIZE_METHOD}\n")
    print(f"✅ Completion flag created: {flag_file}")

# %%
