#!/usr/bin/env python3
"""
Apply rotational symmetry volume estimation to ROI sets
Algorithm from eLife 2021 (Odermatt et al.)

RoiSet.zip -> volume calculation assuming rotational symmetry -> time-series data
"""
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from figure_logger import setup_autosave
setup_autosave()
import zipfile
import struct
import tifffile
import os
from scipy import ndimage
from skimage import morphology, measure
from skimage.draw import polygon
import re
from collections import defaultdict
import cv2

# Font settings
rcParams['font.sans-serif'] = ['Arial']

class RotationalSymmetryROIAnalyzer:
    """Apply rotational symmetry volume estimation to ROI sets"""
    
    def __init__(self, roi_zip_path, pixel_size_um=0.08625, 
                 section_interval_um=0.25, image_width=512, image_height=512,
                 max_iterations=3, convergence_tolerance=0.5):
        """
        Parameters
        ----------
        roi_zip_path : str
            Path to ImageJ ROI set (.zip)
        pixel_size_um : float
            Pixel size (um)
        section_interval_um : float
            Interval between cross-section lines (um), 250nm = 0.25 um in the paper
        image_width : int
            Image width (pixels)
        image_height : int
            Image height (pixels)
        max_iterations : int
            Maximum number of iterations for centerline update
        convergence_tolerance : float
            Convergence threshold (pixels)
        """
        self.roi_zip_path = roi_zip_path
        self.pixel_size_um = pixel_size_um
        self.section_interval_um = section_interval_um
        self.section_interval_px = section_interval_um / pixel_size_um
        self.image_width = image_width
        self.image_height = image_height
        self.max_iterations = max_iterations
        self.convergence_tolerance = convergence_tolerance
        
        print(f"=== Rotational Symmetry ROI Analyzer ===")
        print(f"ROI Set: {roi_zip_path}")
        print(f"Pixel size: {pixel_size_um} um")
        print(f"Section interval: {section_interval_um} um ({self.section_interval_px:.2f} pixels)")
        print(f"Image Size: {image_width} x {image_height}")
        
        # Load ROI set
        self.load_roi_set()
    
    def load_roi_set(self):
        """Load and organize ROI set"""
        print(f"\n=== Loading ROI Set ===")
        
        with zipfile.ZipFile(self.roi_zip_path, 'r') as zf:
            roi_names = zf.namelist()
            print(f"  Total ROIs: {len(roi_names)}")
            
            self.rois_by_time = defaultdict(list)
            self.roi_data = []
            
            for idx, roi_name in enumerate(roi_names):
                if idx % 100 == 0:
                    print(f"    Processing: {idx}/{len(roi_names)}")
                
                try:
                    roi_bytes = zf.read(roi_name)
                    roi_info = self.parse_roi_basic(roi_bytes, roi_name)
                    
                    if roi_info is not None:
                        self.roi_data.append(roi_info)
                        frame_num = self.extract_frame_number(roi_name)
                        self.rois_by_time[frame_num].append(roi_info)
                        
                except Exception as e:
                    continue
            
            print(f"  Successfully parsed: {len(self.roi_data)} ROIs")
            print(f"  Time points: {len(self.rois_by_time)}")
    
    def extract_frame_number(self, roi_name):
        """Extract frame number from ROI name"""
        match = re.match(r'(\d+)-', roi_name)
        if match:
            return int(match.group(1))
        return 0
    
    def parse_roi_basic(self, roi_bytes, roi_name):
        """Parse basic information of ImageJ ROI"""
        if len(roi_bytes) < 64:
            return None
        
        iout = roi_bytes[0:4]
        if iout != b'Iout':
            return None
        
        version = struct.unpack('>h', roi_bytes[4:6])[0]
        roi_type = roi_bytes[6]
        
        top = struct.unpack('>h', roi_bytes[8:10])[0]
        left = struct.unpack('>h', roi_bytes[10:12])[0]
        bottom = struct.unpack('>h', roi_bytes[12:14])[0]
        right = struct.unpack('>h', roi_bytes[14:16])[0]
        n_coordinates = struct.unpack('>h', roi_bytes[16:18])[0]
        
        width = right - left
        height = bottom - top
        
        roi_info = {
            'name': roi_name,
            'type': roi_type,
            'left': left,
            'top': top,
            'right': right,
            'bottom': bottom,
            'width': width,
            'height': height,
            'n_coordinates': n_coordinates,
            'bytes': roi_bytes
        }
        
        return roi_info
    
    def roi_to_mask(self, roi_info):
        """Convert ROI to binary mask"""
        roi_bytes = roi_info['bytes']
        roi_type = roi_info['type']
        
        mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        
        try:
            if roi_type in [0, 7, 8]:  # Polygon, Freehand, Traced
                n_coords = roi_info['n_coordinates']
                if n_coords > 0 and len(roi_bytes) >= 64 + n_coords * 4:
                    x_coords = []
                    y_coords = []
                    
                    for i in range(n_coords):
                        x_offset = 64 + i * 2
                        y_offset = 64 + n_coords * 2 + i * 2
                        
                        if y_offset + 2 <= len(roi_bytes):
                            x = struct.unpack('>h', roi_bytes[x_offset:x_offset+2])[0]
                            y = struct.unpack('>h', roi_bytes[y_offset:y_offset+2])[0]
                            
                            x_coords.append(x + roi_info['left'])
                            y_coords.append(y + roi_info['top'])
                    
                    if len(x_coords) > 2:
                        rr, cc = polygon(y_coords, x_coords, shape=mask.shape)
                        mask[rr, cc] = 255
                        
            elif roi_type == 1:  # Rectangle
                y1, y2 = roi_info['top'], roi_info['bottom']
                x1, x2 = roi_info['left'], roi_info['right']
                mask[y1:y2, x1:x2] = 255
                
            elif roi_type == 2:  # Oval/Ellipse
                from skimage.draw import ellipse
                cy = (roi_info['top'] + roi_info['bottom']) / 2
                cx = (roi_info['left'] + roi_info['right']) / 2
                ry = roi_info['height'] / 2
                rx = roi_info['width'] / 2
                
                rr, cc = ellipse(cy, cx, ry, rx, shape=mask.shape)
                mask[rr, cc] = 255
                
        except Exception as e:
            return None
        
        return mask > 0
    
    def compute_volume_rotational(self, mask, return_visualization_data=False, return_thickness_map=True):
        """
        Compute volume assuming rotational symmetry (iterative update version)
        Algorithm from Odermatt et al. (2021) eLife + thickness map calculation
        """
        # Extract contours
        contours = measure.find_contours(mask, 0.5)
        
        if len(contours) == 0:
            return None
        
        # Largest contour
        contour = max(contours, key=lambda x: len(x))
        contour = contour[:, ::-1]  # Y,X -> X,Y
        
        # Determine long axis using minimum bounding rectangle
        try:
            rect = cv2.minAreaRect(contour.astype(np.float32))
            center, size, angle_deg = rect
            angle_rad = np.deg2rad(angle_deg)
            
            # Long axis
            width, height = size
            if width > height:
                length = width
                axis_angle = angle_rad
            else:
                length = height
                axis_angle = angle_rad + np.pi/2
            
            dx = length/2 * np.cos(axis_angle)
            dy = length/2 * np.sin(axis_angle)
            
            axis_start = np.array([center[0] - dx, center[1] - dy])
            axis_end = np.array([center[0] + dx, center[1] + dy])
            
        except Exception as e:
            return None
        
        # Calculate number of cross-sections
        axis_length = np.linalg.norm(axis_end - axis_start)
        n_sections = int(axis_length / self.section_interval_px)
        
        if n_sections < 2:
            return None
        
        # === Iteratively update centerline and cross-section lines ===
        # Initialization
        t = np.linspace(0, 1, n_sections)
        centerline_points = axis_start[np.newaxis, :] + t[:, np.newaxis] * (axis_end - axis_start)[np.newaxis, :]
        section_angles = np.ones(n_sections) * (axis_angle + np.pi/2)
        
        n_sections_initial = n_sections
        n_sections_removed_total = 0
        converged_iteration = -1
        
        for iteration in range(self.max_iterations):
            new_centerline = []
            new_angles = []
            new_radii = []
            new_section_lines = []
            
            # Current number of sections (may change due to removal)
            current_n_sections = len(centerline_points)
            
            for i in range(current_n_sections):
                # Current cross-section line angle and position
                if isinstance(centerline_points, np.ndarray):
                    current_center = centerline_points[i]
                else:
                    current_center = np.array(centerline_points[i])
                
                if isinstance(section_angles, np.ndarray):
                    current_angle = section_angles[i]
                else:
                    current_angle = section_angles[i]
                
                # Cross-section line
                line_length = 500
                dx_line = line_length * np.cos(current_angle)
                dy_line = line_length * np.sin(current_angle)
                
                line_start = current_center - np.array([dx_line, dy_line])
                line_end = current_center + np.array([dx_line, dy_line])
                
                # Find intersection points with contour
                intersections = []
                
                for j in range(len(contour)):
                    p1 = contour[j]
                    p2 = contour[(j+1) % len(contour)]
                    
                    # Line segment intersection test
                    intersection = self.line_segment_intersection(
                        line_start, line_end, p1, p2
                    )
                    
                    if intersection is not None:
                        intersections.append(intersection)
                
                if len(intersections) >= 2:
                    intersections = np.array(intersections)
                    distances = np.linalg.norm(intersections - current_center, axis=1)
                    sorted_idx = np.argsort(distances)
                    
                    p1 = intersections[sorted_idx[-1]]
                    p2 = intersections[sorted_idx[-2]]
                    
                    # Midpoint (updated centerline point)
                    midpoint = (p1 + p2) / 2
                    
                    radius = np.linalg.norm(p1 - p2) / 2
                    new_centerline.append(midpoint)
                    new_radii.append(radius)
                    new_section_lines.append((p1, p2))
                    
                    # Calculate angle for next iteration (perpendicular to local centerline slope)
                    if i > 0 and i < n_sections - 1 and len(new_centerline) > 1:
                        tangent = new_centerline[-1] - new_centerline[-2]
                        local_angle = np.arctan2(tangent[1], tangent[0])
                        perpendicular_angle = local_angle + np.pi/2
                        new_angles.append(perpendicular_angle)
                    else:
                        new_angles.append(current_angle)
                else:
                    # Keep previous values if no intersection found
                    new_centerline.append(current_center)
                    new_radii.append(0)
                    new_angles.append(current_angle)
            
            # Convergence check
            if iteration > 0 and len(new_centerline) > 0 and len(centerline_points) > 0:
                # Skip convergence check if size changed (sections removed by intersection check)
                if len(new_centerline) == len(centerline_points):
                    shifts = [np.linalg.norm(new_centerline[i] - centerline_points[i]) 
                             for i in range(len(new_centerline))]
                    mean_shift = np.mean(shifts)
                    
                    if mean_shift < self.convergence_tolerance:
                        converged_iteration = iteration + 1
                        centerline_points = new_centerline
                        radii = new_radii
                        section_lines = new_section_lines
                        break
            
            # Remove crossing section lines
            # "Sectioning lines that crossed a neighboring line were removed."
            valid_indices = []
            for i in range(len(new_section_lines)):
                is_valid = True
                
                # Check intersection with neighboring section lines
                if i > 0:
                    # Check if crossing with previous section line
                    if self._check_line_intersection(new_section_lines[i-1], new_section_lines[i]):
                        is_valid = False
                
                if i < len(new_section_lines) - 1 and is_valid:
                    # Check if crossing with next section line
                    if self._check_line_intersection(new_section_lines[i], new_section_lines[i+1]):
                        is_valid = False
                
                if is_valid:
                    valid_indices.append(i)
            
            # Record removal count
            n_removed_this_iter = len(new_section_lines) - len(valid_indices)
            if n_removed_this_iter > 0:
                n_sections_removed_total += n_removed_this_iter
            
            # Keep only valid section lines
            if len(valid_indices) > 0:
                new_centerline = [new_centerline[i] for i in valid_indices]
                new_radii = [new_radii[i] for i in valid_indices]
                new_section_lines = [new_section_lines[i] for i in valid_indices]
                new_angles = [new_angles[i] for i in valid_indices]
            
            # Update
            centerline_points = new_centerline
            section_angles = new_angles
            radii = new_radii
            section_lines = new_section_lines
        
        if len(radii) < 2:
            return None
        
        radii = np.array(radii)
        
        # Volume calculation (sum of cylinders)
        total_volume_px3 = 0
        
        for r in radii:
            h = self.section_interval_px
            volume = np.pi * r**2 * h
            total_volume_px3 += volume
        
        # Pixels -> um
        volume_um3 = total_volume_px3 * (self.pixel_size_um ** 3)
        
        # Surface area
        surface_area_px2 = 0
        for r in radii:
            h = self.section_interval_px
            area = 2 * np.pi * r * h
            surface_area_px2 += area
        
        # End caps
        if len(radii) > 0:
            surface_area_px2 += np.pi * radii[0]**2
            surface_area_px2 += np.pi * radii[-1]**2
        
        surface_area_um2 = surface_area_px2 * (self.pixel_size_um ** 2)
        
        # === Calculate thickness map (number of Z-occupied slices at each XY pixel) ===
        thickness_map = np.zeros((self.image_height, self.image_width), dtype=np.float32)
        
        if return_thickness_map and len(centerline_points) > 0 and len(radii) > 0:
            centerline_array = np.array(centerline_points)
            
            # Process at each centerline point
            for i, (center, radius) in enumerate(zip(centerline_array, radii)):
                if radius > 0:
                    # Calculate thickness from maximum radius at this position
                    # Assuming rotational symmetry: Z-height of sphere with radius R = 2R
                    # Discretized by section_interval
                    z_height_um = 2 * radius * self.pixel_size_um
                    z_slices = z_height_um / self.pixel_size_um  # Number of slices (calculated in pixel units)
                    
                    # Assign thickness to pixels within this radius
                    y, x = int(center[1]), int(center[0])
                    r_int = int(radius) + 1
                    
                    for dy in range(-r_int, r_int+1):
                        for dx in range(-r_int, r_int+1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < self.image_height and 0 <= nx < self.image_width:
                                dist_from_center = np.sqrt(dx**2 + dy**2)
                                if dist_from_center <= radius:
                                    # Sphere cross-section: Z-height = 2*sqrt(R^2 - r^2)
                                    z_at_r = 2 * np.sqrt(max(0, radius**2 - dist_from_center**2))
                                    # Keep maximum value (when multiple sections overlap)
                                    thickness_map[ny, nx] = max(thickness_map[ny, nx], z_at_r)
        
        result = {
            'volume_um3': volume_um3,
            'surface_area_um2': surface_area_um2,
            'n_sections': len(radii),
            'n_sections_initial': n_sections_initial,
            'n_sections_removed': n_sections_removed_total,
            'converged_iteration': converged_iteration,
            'mean_radius_um': np.mean(radii) * self.pixel_size_um if len(radii) > 0 else 0,
            'max_radius_um': np.max(radii) * self.pixel_size_um if len(radii) > 0 else 0,
            'length_um': axis_length * self.pixel_size_um,
            'area_2d': np.sum(mask),
            'thickness_map': thickness_map
        }
        
        # Add visualization data
        if return_visualization_data:
            result['centerline_points'] = np.array(centerline_points) if len(centerline_points) > 0 else None
            result['section_lines'] = section_lines
            result['contour'] = contour
            result['axis_start'] = axis_start
            result['axis_end'] = axis_end
            result['radii'] = radii
            result['n_sections_initial'] = n_sections_initial
            result['n_sections_removed'] = n_sections_removed_total
            result['converged_iteration'] = converged_iteration
        
        return result
    
    def line_segment_intersection(self, p1, p2, p3, p4):
        """Calculate intersection point of two line segments"""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 1e-10:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return np.array([x, y])
        
        return None
    
    def _check_line_intersection(self, line1, line2):
        """
        Check if two section lines intersect
        Paper: "Sectioning lines that crossed a neighboring line were removed."

        Parameters
        ----------
        line1 : tuple of (p1, p2)
            Endpoints of the first section line
        line2 : tuple of (p1, p2)
            Endpoints of the second section line

        Returns
        -------
        bool
            True if the lines intersect
        """
        p1_line1, p2_line1 = line1
        p1_line2, p2_line2 = line2
        
        # Line segment intersection test
        intersection = self.line_segment_intersection(
            p1_line1, p2_line1, p1_line2, p2_line2
        )
        
        return intersection is not None
    
    def analyze_timeseries(self, max_frames=None, save_visualizations=False, save_thickness_maps=True):
        """Analyze volume over time-series"""
        print(f"\n=== Analyzing Time-series with Rotational Symmetry ===")
        
        time_points = sorted(self.rois_by_time.keys())
        
        if max_frames is not None:
            time_points = time_points[:max_frames]
            print(f"  Processing first {max_frames} frames")
        
        print(f"  Time points to process: {len(time_points)}")
        
        results = []
        self.visualization_data = []  # Store visualization data
        self.thickness_maps = []  # Store thickness maps
        
        for t_idx, t in enumerate(time_points):
            print(f"\n  Frame {t_idx+1}/{len(time_points)} (t={t})")
            
            rois_at_t = self.rois_by_time[t]
            print(f"    ROIs at this time: {len(rois_at_t)}")
            
            for cell_idx, roi_info in enumerate(rois_at_t):
                if cell_idx % 10 == 0:
                    print(f"      Cell {cell_idx+1}/{len(rois_at_t)}")
                
                mask = self.roi_to_mask(roi_info)
                
                if mask is None or np.sum(mask) == 0:
                    continue
                
                vol_result = self.compute_volume_rotational(mask, return_visualization_data=save_visualizations)
                
                if vol_result is not None:
                    vol_result['time_point'] = t
                    vol_result['time_index'] = t_idx
                    vol_result['cell_index'] = cell_idx
                    vol_result['roi_name'] = roi_info['name']
                    
                    # Store visualization data separately
                    if save_visualizations and 'centerline_points' in vol_result:
                        vis_data = {
                            'time_index': t_idx,
                            'cell_index': cell_idx,
                            'roi_name': roi_info['name'],
                            'mask': mask,
                            'centerline_points': vol_result['centerline_points'],
                            'section_lines': vol_result['section_lines'],
                            'contour': vol_result['contour'],
                            'axis_start': vol_result['axis_start'],
                            'axis_end': vol_result['axis_end'],
                            'radii': vol_result['radii'],
                            'n_sections_initial': vol_result.get('n_sections_initial', 0),
                            'n_sections_removed': vol_result.get('n_sections_removed', 0),
                            'converged_iteration': vol_result.get('converged_iteration', -1)
                        }
                        self.visualization_data.append(vis_data)
                    
                    # Store thickness map separately
                    if save_thickness_maps and 'thickness_map' in vol_result:
                        thickness_info = {
                            'time_index': t_idx,
                            'time_point': t,
                            'cell_index': cell_idx,
                            'roi_name': roi_info['name'],
                            'thickness_map': vol_result['thickness_map']
                        }
                        self.thickness_maps.append(thickness_info)
                    
                    # Exclude visualization data and thickness maps from CSV data
                    result_for_csv = {k: v for k, v in vol_result.items() 
                                     if k not in ['centerline_points', 'section_lines', 'contour', 
                                                 'axis_start', 'axis_end', 'radii', 'thickness_map']}
                    results.append(result_for_csv)
                    
                    if cell_idx < 3:  # Display only first few
                        max_thickness = np.max(vol_result['thickness_map'])
                        print(f"        [OK] {roi_info['name']}: Volume={vol_result['volume_um3']:.2f} um^3, "
                              f"Max thickness={max_thickness:.1f}px, "
                              f"Sections={vol_result['n_sections']}/{vol_result['n_sections_initial']} "
                              f"(removed={vol_result['n_sections_removed']}), "
                              f"Converged@iter{vol_result['converged_iteration']}")
        
        self.results_df = pd.DataFrame(results)
        
        print(f"\n  Total processed: {len(self.results_df)} cells")
        
        if len(self.results_df) > 0:
            print(f"  Volume range: {self.results_df['volume_um3'].min():.2f} - {self.results_df['volume_um3'].max():.2f} um^3")
        
        return self.results_df
    
    def save_results(self, output_dir='rotational_volume_output'):
        """Save results"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n=== Saving Results to {output_dir} ===")
        
        # CSV
        csv_path = os.path.join(output_dir, 'rotational_volume_timeseries.csv')
        self.results_df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")
        
        # Summary
        summary_path = os.path.join(output_dir, 'rotational_volume_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("=== Rotational Symmetry Volume Analysis ===\n\n")
            f.write(f"Algorithm: Odermatt et al. (2021) eLife 10:e64901\n")
            f.write(f"Section interval: {self.section_interval_um} um\n\n")
            
            f.write(f"Total cells: {len(self.results_df)}\n")
            f.write(f"Time points: {len(self.results_df['time_index'].unique())}\n\n")
            
            f.write("Volume Statistics (um^3):\n")
            f.write(f"  Mean: {self.results_df['volume_um3'].mean():.2f}\n")
            f.write(f"  Median: {self.results_df['volume_um3'].median():.2f}\n")
            f.write(f"  Std: {self.results_df['volume_um3'].std():.2f}\n")
            f.write(f"  Min: {self.results_df['volume_um3'].min():.2f}\n")
            f.write(f"  Max: {self.results_df['volume_um3'].max():.2f}\n\n")
            
            f.write("Surface Area Statistics (um^2):\n")
            f.write(f"  Mean: {self.results_df['surface_area_um2'].mean():.2f}\n")
            f.write(f"  Std: {self.results_df['surface_area_um2'].std():.2f}\n\n")
            
            f.write(self.results_df.describe().to_string())
        
        print(f"  Saved: {summary_path}")
        
        # Save thickness maps
        if hasattr(self, 'thickness_maps') and len(self.thickness_maps) > 0:
            thickness_dir = os.path.join(output_dir, 'thickness_maps')
            os.makedirs(thickness_dir, exist_ok=True)
            
            print(f"\n  Saving thickness maps ({len(self.thickness_maps)} maps)...")
            
            for idx, thick_info in enumerate(self.thickness_maps):
                if idx % 100 == 0:
                    print(f"    Progress: {idx}/{len(self.thickness_maps)}")
                
                roi_name = thick_info['roi_name'].replace('.roi', '')
                thick_path = os.path.join(thickness_dir, f"{roi_name}_thickness.tif")
                
                tifffile.imwrite(thick_path, thick_info['thickness_map'])
            
            print(f"  Saved: {len(self.thickness_maps)} thickness maps to {thickness_dir}/")
            
            # Integrated stack
            if len(self.thickness_maps) > 0:
                stack_list = [tm['thickness_map'] for tm in self.thickness_maps]
                stack_array = np.stack(stack_list, axis=0)
                
                stack_path = os.path.join(output_dir, 'thickness_stack_all_frames.tif')
                tifffile.imwrite(stack_path, stack_array.astype(np.float32),
                                metadata={'axes': 'TYX'})
                print(f"  Saved: {stack_path} (shape: {stack_array.shape})")
    
    def compute_ri_from_phase_images(self, phase_image_dir, wavelength_nm=663, n_medium=1.333):
        """
        Compute RI (Refractive Index) from phase images and thickness maps
        Same processing as 24_ellipse_volume.py

        Parameters
        ----------
        phase_image_dir : str
            Directory containing phase images
        wavelength_nm : float
            Wavelength (nanometers)
        n_medium : float
            Medium refractive index

        Returns
        -------
        ri_results : list of dict
            RI calculation results for each frame
        """
        if not hasattr(self, 'thickness_maps') or len(self.thickness_maps) == 0:
            print("Error: No thickness maps available. Run analyze_timeseries() first.")
            return None
        
        print(f"\n=== Computing RI from Phase Images ===")
        print(f"Phase image directory: {phase_image_dir}")
        print(f"Wavelength: {wavelength_nm} nm")
        print(f"Medium RI: {n_medium}")
        
        wavelength_um = wavelength_nm / 1000.0
        
        # Search for phase images and extract numbers from filenames
        import glob
        phase_files_all = sorted(glob.glob(os.path.join(phase_image_dir, "*.tif")))
        
        if len(phase_files_all) == 0:
            print(f"Error: No .tif files found in {phase_image_dir}")
            return None
        
        # Create dictionary by extracting numbers from filenames
        # e.g.: "subtracted_by_maskmean_float320085_bg_corr_aligned.tif" -> 85
        phase_file_dict = {}
        for phase_file in phase_files_all:
            basename = os.path.basename(phase_file)
            # Extract number (last numeric part)
            match = re.search(r'(\d+)(?:_bg_corr_aligned)?\.tif$', basename)
            if match:
                frame_num = int(match.group(1))
                phase_file_dict[frame_num] = phase_file
        
        print(f"Found {len(phase_file_dict)} phase images with frame numbers {min(phase_file_dict.keys())}-{max(phase_file_dict.keys())}")
        
        ri_results = []
        not_found_count = 0
        
        for thick_info in self.thickness_maps:
            time_point = thick_info.get('time_point', thick_info['time_index'])  # Frame number extracted from ROI name
            roi_name = thick_info['roi_name']
            thickness_map = thick_info['thickness_map']
            
            # Extract frame number from ROI name
            # e.g.: "0085-0024-0136.roi" -> 85
            match = re.match(r'(\d+)-', roi_name)
            if match:
                frame_num = int(match.group(1))
            else:
                frame_num = time_point
            
            if frame_num in phase_file_dict:
                phase_img = tifffile.imread(phase_file_dict[frame_num])
                
                # Convert thickness map (pixel count) to actual thickness (um)
                thickness_um = thickness_map * self.pixel_size_um
                
                # Size check
                if phase_img.shape != thickness_map.shape:
                    print(f"  Warning: Size mismatch for {roi_name}")
                    continue
                
                # Avoid division by zero
                thickness_um_safe = np.where(thickness_um > 0, thickness_um, np.nan)
                
                # RI calculation: n_sample = n_medium + (phi * lambda) / (2*pi * thickness)
                n_sample = n_medium + (phase_img * wavelength_um) / (2 * np.pi * thickness_um_safe)
                
                # Within mask only
                mask = thickness_map > 0
                
                if np.sum(mask) > 0:
                    mean_ri = np.nanmean(n_sample[mask])
                    median_ri = np.nanmedian(n_sample[mask])
                    std_ri = np.nanstd(n_sample[mask])
                    total_ri = np.nansum(n_sample[mask] - n_medium)
                    
                    ri_result = {
                        'time_index': thick_info['time_index'],
                        'time_point': time_point,
                        'frame_num': frame_num,
                        'roi_name': roi_name,
                        'mean_ri': mean_ri,
                        'median_ri': median_ri,
                        'std_ri': std_ri,
                        'total_ri': total_ri,
                        'n_pixels': np.sum(mask),
                        'ri_map': n_sample
                    }
                    
                    ri_results.append(ri_result)
                    
                    if len(ri_results) <= 3:
                        print(f"  [Frame {frame_num}] {roi_name}: Mean RI = {mean_ri:.4f}, Total RI = {total_ri:.2f}")
            else:
                not_found_count += 1
                if not_found_count <= 3:
                    print(f"  Warning: Phase image not found for frame {frame_num} ({roi_name})")
        
        if not_found_count > 0:
            print(f"\n  Warning: {not_found_count} phase images not found")
        
        self.ri_results = ri_results
        print(f"\n  Total RI calculations: {len(ri_results)}")
        
        return ri_results
    
    def save_ri_results(self, output_dir='rotational_volume_output'):
        """Save RI calculation results"""
        if not hasattr(self, 'ri_results') or len(self.ri_results) == 0:
            print("No RI results to save")
            return
        
        ri_dir = os.path.join(output_dir, 'ri_maps')
        os.makedirs(ri_dir, exist_ok=True)
        
        print(f"\n=== Saving RI Results to {ri_dir} ===")
        
        # Save RI statistics to CSV
        ri_stats = []
        for ri_res in self.ri_results:
            stats = {k: v for k, v in ri_res.items() if k != 'ri_map'}
            ri_stats.append(stats)
        
        ri_df = pd.DataFrame(ri_stats)
        csv_path = os.path.join(output_dir, 'ri_statistics.csv')
        ri_df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")
        
        # Save individual RI maps
        print(f"  Saving {len(self.ri_results)} RI maps...")
        for idx, ri_res in enumerate(self.ri_results):
            if idx % 100 == 0:
                print(f"    Progress: {idx}/{len(self.ri_results)}")
            
            roi_name = ri_res['roi_name'].replace('.roi', '')
            ri_path = os.path.join(ri_dir, f"{roi_name}_ri_map.tif")
            
            tifffile.imwrite(ri_path, ri_res['ri_map'].astype(np.float32))
        
        print(f"  Saved: {len(self.ri_results)} RI maps")
        
        # Summary
        summary_path = os.path.join(output_dir, 'ri_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("=== RI Time-series Summary ===\n\n")
            f.write(f"Total measurements: {len(ri_df)}\n")
            f.write(f"Time points: {len(ri_df['time_index'].unique())}\n\n")
            
            f.write("Mean RI Statistics:\n")
            f.write(f"  Mean: {ri_df['mean_ri'].mean():.4f}\n")
            f.write(f"  Std: {ri_df['mean_ri'].std():.4f}\n")
            f.write(f"  Min: {ri_df['mean_ri'].min():.4f}\n")
            f.write(f"  Max: {ri_df['mean_ri'].max():.4f}\n\n")
            
            f.write("Total RI Statistics:\n")
            f.write(f"  Mean: {ri_df['total_ri'].mean():.2f}\n")
            f.write(f"  Std: {ri_df['total_ri'].std():.2f}\n")
            f.write(f"  Min: {ri_df['total_ri'].min():.2f}\n")
            f.write(f"  Max: {ri_df['total_ri'].max():.2f}\n\n")
            
            f.write(ri_df.describe().to_string())
        
        print(f"  Saved: {summary_path}")
    
    def save_visualizations(self, output_dir='rotational_volume_output', format='png'):
        """Save visualizations of section lines and centerline"""
        if not hasattr(self, 'visualization_data') or len(self.visualization_data) == 0:
            print("No visualization data available")
            return
        
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        print(f"\n=== Saving Visualizations ({format.upper()}) ===")
        print(f"  Total visualizations: {len(self.visualization_data)}")
        
        for idx, vis_data in enumerate(self.visualization_data):
            if idx % 10 == 0:
                print(f"    Progress: {idx}/{len(self.visualization_data)}")
            
            roi_name = vis_data['roi_name'].replace('.roi', '')
            
            # Create image
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            
            # Mask as background
            ax.imshow(vis_data['mask'], cmap='gray', alpha=0.3)
            
            # Contour
            contour = vis_data['contour']
            ax.plot(contour[:, 0], contour[:, 1], 'b-', linewidth=2, label='Contour', alpha=0.7)
            
            # Long axis
            axis_start = vis_data['axis_start']
            axis_end = vis_data['axis_end']
            ax.plot([axis_start[0], axis_end[0]], [axis_start[1], axis_end[1]], 
                   'r-', linewidth=3, label='Long axis', alpha=0.8)
            
            # Centerline
            if vis_data['centerline_points'] is not None and len(vis_data['centerline_points']) > 0:
                centerline = vis_data['centerline_points']
                ax.plot(centerline[:, 0], centerline[:, 1], 'g-', 
                       linewidth=3, label='Centerline', marker='o', markersize=4)
            
            # Section lines
            section_lines = vis_data['section_lines']
            for i, (p1, p2) in enumerate(section_lines):
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                       'c-', linewidth=1, alpha=0.5)

                # Display radius as circle
                if vis_data['radii'] is not None and i < len(vis_data['radii']):
                    midpoint = (p1 + p2) / 2
                    radius = vis_data['radii'][i]
                    circle = plt.Circle((midpoint[0], midpoint[1]), radius, 
                                       fill=False, color='yellow', linewidth=1, alpha=0.3)
                    ax.add_patch(circle)
            
            # Add detailed information to title
            title_text = f"{roi_name}\n"
            title_text += f"Sections: {len(section_lines)}"
            
            # Add removal information if available
            if 'n_sections_initial' in vis_data and 'n_sections_removed' in vis_data:
                n_init = vis_data['n_sections_initial']
                n_removed = vis_data['n_sections_removed']
                if n_removed > 0:
                    title_text += f" (initial: {n_init}, removed: {n_removed})"
            
            # Add convergence information if available
            if 'converged_iteration' in vis_data and vis_data['converged_iteration'] > 0:
                title_text += f"\nConverged at iteration {vis_data['converged_iteration']}"
            
            ax.set_title(title_text, fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', fontsize=8)
            ax.axis('equal')
            
            # Set appropriate range
            margin = 20
            ax.set_xlim(np.min(contour[:, 0]) - margin, np.max(contour[:, 0]) + margin)
            ax.set_ylim(np.min(contour[:, 1]) - margin, np.max(contour[:, 1]) + margin)
            
            # Save
            if format == 'png':
                save_path = os.path.join(vis_dir, f"{roi_name}_visualization.png")
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            elif format == 'tif' or format == 'tiff':
                save_path = os.path.join(vis_dir, f"{roi_name}_visualization.tif")
                plt.savefig(save_path, dpi=150, bbox_inches='tight', format='tiff')
            
            plt.close(fig)
        
        print(f"  Saved: {len(self.visualization_data)} visualizations to {vis_dir}/")
        
        # Create integrated TIFF stack (optional)
        if format in ['tif', 'tiff']:
            print(f"\n  Creating integrated TIFF stack...")
            self._create_visualization_stack(vis_dir)
    
    def _create_visualization_stack(self, vis_dir):
        """Create a TIFF stack combining all visualization images"""
        try:
            # Load all visualization images
            images = []
            
            for vis_data in self.visualization_data:
                roi_name = vis_data['roi_name'].replace('.roi', '')
                img_path = os.path.join(vis_dir, f"{roi_name}_visualization.tif")
                
                if os.path.exists(img_path):
                    img = tifffile.imread(img_path)
                    # For RGB, keep only first 3 channels
                    if img.ndim == 3 and img.shape[2] >= 3:
                        img = img[:, :, :3]
                    images.append(img)
            
            if len(images) > 0:
                # Convert to stack
                stack = np.stack(images, axis=0)
                
                # Save
                stack_path = os.path.join(vis_dir, 'visualization_stack_all_frames.tif')
                tifffile.imwrite(stack_path, stack, metadata={'axes': 'TYXC'})
                print(f"  Saved: {stack_path} (shape: {stack.shape})")
        except Exception as e:
            print(f"  Warning: Failed to create stack: {e}")
    
    def plot_results(self, save_path='rotational_volume_plot.png'):
        """Plot results"""
        if not hasattr(self, 'results_df') or len(self.results_df) == 0:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Volume time-series
        ax = axes[0, 0]
        for cell_idx in self.results_df['cell_index'].unique()[:10]:  # First 10
            cell_data = self.results_df[self.results_df['cell_index'] == cell_idx]
            ax.plot(cell_data['time_index'], cell_data['volume_um3'], 
                   alpha=0.7, linewidth=2, marker='o')
        
        ax.set_xlabel('Time (frame)', fontsize=12)
        ax.set_ylabel('Volume (um^3)', fontsize=12)
        ax.set_title('Volume Time-series (Rotational Symmetry)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 2. Mean volume
        ax = axes[0, 1]
        mean_vol = self.results_df.groupby('time_index')['volume_um3'].mean()
        std_vol = self.results_df.groupby('time_index')['volume_um3'].std()
        
        ax.plot(mean_vol.index, mean_vol, 'b-', linewidth=2, label='Mean')
        ax.fill_between(mean_vol.index, mean_vol - std_vol, mean_vol + std_vol,
                        alpha=0.3, color='blue', label='±1 SD')
        
        ax.set_xlabel('Time (frame)', fontsize=12)
        ax.set_ylabel('Volume (um^3)', fontsize=12)
        ax.set_title('Mean Volume Time-series', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Surface area vs volume
        ax = axes[1, 0]
        ax.scatter(self.results_df['volume_um3'], self.results_df['surface_area_um2'],
                  alpha=0.3, s=20)
        
        ax.set_xlabel('Volume (um^3)', fontsize=12)
        ax.set_ylabel('Surface Area (um^2)', fontsize=12)
        ax.set_title('Surface Area vs Volume', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 4. Length and radius distribution
        ax = axes[1, 1]
        ax.scatter(self.results_df['length_um'], self.results_df['mean_radius_um'],
                  alpha=0.3, s=20, c=self.results_df['volume_um3'], cmap='viridis')
        
        ax.set_xlabel('Cell Length (um)', fontsize=12)
        ax.set_ylabel('Mean Radius (um)', fontsize=12)
        ax.set_title('Cell Shape Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Volume (um^3)', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n=== Plot saved: {save_path} ===")
        plt.show()


def main():
    """Main execution"""
    roi_zip_path = r"C:\Users\QPI\Desktop\align_demo\from_outputphase\bg_corr\subtracted\inference_out\Roiset_enlarge.zip"
    if not os.path.exists(roi_zip_path):
        print(f"Error: ROI set not found at {roi_zip_path}")
        return
    
    print(f"Found ROI set: {roi_zip_path}")
    
    # Create analyzer (with iterative update)
    analyzer = RotationalSymmetryROIAnalyzer(
        roi_zip_path=roi_zip_path,
        pixel_size_um=0.348,
        section_interval_um=0.25,  # 250 nm
        image_width=512,
        image_height=512,
        max_iterations=2,  # Maximum 3 iterative updates
        convergence_tolerance=0.5  # Converge when below 0.5 pixels
    )
    
    # Run analysis (also save visualization data and thickness maps)
    print(f"\n{'='*60}")
    print(f"SETTINGS:")
    print(f"  Max iterations: {analyzer.max_iterations}")
    print(f"  Convergence tolerance: {analyzer.convergence_tolerance} pixels")
    print(f"  Section interval: {analyzer.section_interval_um} um")
    print(f"{'='*60}")
    
    results_df = analyzer.analyze_timeseries(
        max_frames=100, 
        save_visualizations=True,
        save_thickness_maps=True
    )
    
    # Save results
    analyzer.save_results('rotational_volume_output')
    
    # Save visualizations (PNG format)
    analyzer.save_visualizations('rotational_volume_output', format='png')
    
    # Plot
    analyzer.plot_results('rotational_volume_plot.png')
    
    # RI calculation (using phase images)
    # Relative path from scripts directory
    phase_dir = os.path.join(os.path.dirname(__file__), "..", "data", "align_demo", "bg_corr_aligned", "aligned")
    phase_dir = os.path.abspath(phase_dir)
    
    if os.path.exists(phase_dir):
        print(f"\n=== RI Calculation ===")
        print(f"Using phase images from: {phase_dir}")
        analyzer.compute_ri_from_phase_images(phase_dir, wavelength_nm=663, n_medium=1.333)
        analyzer.save_ri_results('rotational_volume_output')
    else:
        print(f"\n  Warning: Phase image directory not found: {phase_dir}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nOutput files:")
    print("  - rotational_volume_output/rotational_volume_timeseries.csv")
    print("  - rotational_volume_output/rotational_volume_summary.txt")
    print("  - rotational_volume_output/thickness_maps/ (z-stack thickness)")
    print("  - rotational_volume_output/thickness_stack_all_frames.tif")
    print("  - rotational_volume_output/visualizations/ (centerline & cross-sections)")
    
    if hasattr(analyzer, 'ri_results') and len(analyzer.ri_results) > 0:
        print("  - rotational_volume_output/ri_statistics.csv")
        print("  - rotational_volume_output/ri_summary.txt")
        print("  - rotational_volume_output/ri_maps/ (RI maps)")
    
    print("  - rotational_volume_plot.png")
    print("\nAlgorithm features:")
    print(f"  - Iterative centerline update: {analyzer.max_iterations} iterations")
    print(f"  - Convergence tolerance: {analyzer.convergence_tolerance} px")
    print(f"  - Z-stack thickness maps generated for RI calculation")
    print(f"  - RI calculation from phase images: {len(analyzer.ri_results) if hasattr(analyzer, 'ri_results') else 0} cells")
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()

# %%

