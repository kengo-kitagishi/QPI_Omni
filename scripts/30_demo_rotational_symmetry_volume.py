#!/usr/bin/env python3
"""
Rotational Symmetry Volume Estimation
Implementation of the algorithm from eLife 2021 (Odermatt et al.)

Method to compute cell volume assuming rotational symmetry from cell contour:
1. Determine the long axis of the cell
2. Place cross-section lines perpendicular to the long axis at regular intervals
3. Compute intersection points of each cross-section line with the contour
4. Update centerline (passing through midpoints of intersections)
5. Update cross-section line angles (perpendicular to centerline)
6. Sum volumes of each cross-section assuming rotational symmetry

Reference:
Odermatt et al. (2021) eLife 10:e64901
https://elifesciences.org/articles/64901
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from figure_logger import setup_autosave
setup_autosave()
from skimage import morphology, measure
from skimage.draw import polygon
import tifffile
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import cv2

class RotationalSymmetryVolumeEstimator:
    """Volume estimation assuming rotational symmetry"""
    
    def __init__(self, pixel_size_um=0.08625, section_interval_um=0.25):
        """
        Parameters
        ----------
        pixel_size_um : float
            Pixel size (um)
        section_interval_um : float
            Cross-section line interval (um)
            250nm = 0.25 um in the paper
        """
        self.pixel_size_um = pixel_size_um
        self.section_interval_um = section_interval_um
        self.section_interval_px = section_interval_um / pixel_size_um
        
        print(f"=== Rotational Symmetry Volume Estimator ===")
        print(f"Pixel size: {pixel_size_um} um")
        print(f"Section interval: {section_interval_um} um ({self.section_interval_px:.2f} pixels)")
    
    def fit_minimum_bounding_rectangle(self, contour):
        """
        Compute minimum bounding rectangle (for determining long axis)

        Returns
        -------
        center : tuple
            Center coordinates (x, y)
        size : tuple
            Size (width, height)
        angle : float
            Angle (radians)
        """
        # Using OpenCV
        rect = cv2.minAreaRect(contour.astype(np.float32))
        center, size, angle_deg = rect
        angle_rad = np.deg2rad(angle_deg)
        
        return center, size, angle_rad
    
    def get_long_axis(self, contour):
        """
        Determine the long axis of the cell

        Returns
        -------
        axis_start : ndarray
            Start point of long axis (x, y)
        axis_end : ndarray
            End point of long axis (x, y)
        angle : float
            Angle of long axis (radians)
        """
        center, size, angle = self.fit_minimum_bounding_rectangle(contour)
        
        # Use the longer side as the long axis
        width, height = size
        if width > height:
            length = width
            axis_angle = angle
        else:
            length = height
            axis_angle = angle + np.pi/2
        
        # Start and end points of the long axis
        dx = length/2 * np.cos(axis_angle)
        dy = length/2 * np.sin(axis_angle)
        
        axis_start = np.array([center[0] - dx, center[1] - dy])
        axis_end = np.array([center[0] + dx, center[1] + dy])
        
        return axis_start, axis_end, axis_angle
    
    def compute_perpendicular_sections(self, axis_start, axis_end, n_sections):
        """
        Compute positions of cross-section lines perpendicular to the long axis

        Returns
        -------
        section_centers : ndarray
            Center positions of each cross-section (n_sections, 2)
        section_angle : float
            Angle of cross-section lines (perpendicular to long axis)
        """
        # Place points at equal intervals along the long axis
        t = np.linspace(0, 1, n_sections)
        section_centers = axis_start[np.newaxis, :] + t[:, np.newaxis] * (axis_end - axis_start)[np.newaxis, :]
        
        # Angle of the long axis
        axis_vec = axis_end - axis_start
        axis_angle = np.arctan2(axis_vec[1], axis_vec[0])
        
        # Angle of cross-section lines (perpendicular to long axis)
        section_angle = axis_angle + np.pi/2
        
        return section_centers, section_angle
    
    def find_contour_intersections(self, section_center, section_angle, contour, line_length=500):
        """
        Find intersection points between cross-section line and contour

        Returns
        -------
        intersections : list of ndarray
            List of intersection coordinates
        """
        # Define cross-section line (long enough)
        dx = line_length * np.cos(section_angle)
        dy = line_length * np.sin(section_angle)
        
        line_start = section_center - np.array([dx, dy])
        line_end = section_center + np.array([dx, dy])
        
        # Find intersections with the contour
        intersections = []
        
        for i in range(len(contour)):
            p1 = contour[i]
            p2 = contour[(i+1) % len(contour)]
            
            # Line segment intersection check
            intersection = self.line_segment_intersection(
                line_start, line_end, p1, p2
            )
            
            if intersection is not None:
                intersections.append(intersection)
        
        return intersections
    
    def line_segment_intersection(self, p1, p2, p3, p4):
        """
        Compute intersection point of two line segments

        Parameters
        ----------
        p1, p2 : ndarray
            Endpoints of line segment 1
        p3, p4 : ndarray
            Endpoints of line segment 2

        Returns
        -------
        intersection : ndarray or None
            Intersection coordinates, None if no intersection
        """
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
    
    def compute_cell_volume(self, mask, return_details=False):
        """
        Compute volume from a cell mask

        Parameters
        ----------
        mask : ndarray
            2D binary mask
        return_details : bool
            Whether to return detailed information

        Returns
        -------
        volume_um3 : float
            Volume (um^3)
        details : dict (optional)
            Detailed information
        """
        # Extract contour
        contours = measure.find_contours(mask, 0.5)
        
        if len(contours) == 0:
            return None
        
        # Use the largest contour
        contour = max(contours, key=lambda x: len(x))
        
        # Convert from Y, X order to X, Y order
        contour = contour[:, ::-1]
        
        # Determine long axis
        axis_start, axis_end, axis_angle = self.get_long_axis(contour)
        
        # Length of long axis
        axis_length = np.linalg.norm(axis_end - axis_start)
        
        # Compute number of cross-sections
        n_sections = int(axis_length / self.section_interval_px)
        
        if n_sections < 2:
            return None
        
        # Initialize cross-section positions
        section_centers, section_angle = self.compute_perpendicular_sections(
            axis_start, axis_end, n_sections
        )
        
        # Find contour intersections at each cross-section and update centerline
        centerline = []
        radii = []
        valid_sections = []
        
        for i in range(n_sections):
            intersections = self.find_contour_intersections(
                section_centers[i], section_angle, contour
            )
            
            if len(intersections) >= 2:
                # Sort intersections by distance
                intersections = np.array(intersections)
                distances = np.linalg.norm(intersections - section_centers[i], axis=1)
                sorted_idx = np.argsort(distances)
                
                # Use the two farthest points (both sides of the cell)
                p1 = intersections[sorted_idx[-1]]
                p2 = intersections[sorted_idx[-2]]
                
                # Compute midpoint
                midpoint = (p1 + p2) / 2
                
                # Compute radius (assuming rotational symmetry)
                radius = np.linalg.norm(p1 - p2) / 2
                
                centerline.append(midpoint)
                radii.append(radius)
                valid_sections.append(i)
        
        if len(centerline) < 2:
            return None
        
        centerline = np.array(centerline)
        radii = np.array(radii)
        
        # Compute volume (sum of cylinders)
        total_volume_px3 = 0
        
        for i in range(len(radii)):
            # Compute each cross-section as a cylinder
            r = radii[i]
            h = self.section_interval_px
            
            # Cylinder volume: V = pi * r^2 * h
            volume = np.pi * r**2 * h
            total_volume_px3 += volume
        
        # Pixel to um conversion
        volume_um3 = total_volume_px3 * (self.pixel_size_um ** 3)
        
        # Compute surface area (optional)
        surface_area_px2 = 0
        for i in range(len(radii)):
            r = radii[i]
            h = self.section_interval_px
            
            # Cylinder lateral area: A = 2*pi * r * h
            area = 2 * np.pi * r * h
            surface_area_px2 += area
        
        # End caps (approximated as spherical caps)
        if len(radii) > 0:
            # Start cap
            r_start = radii[0]
            cap_area_start = np.pi * r_start**2
            surface_area_px2 += cap_area_start
            
            # End cap
            r_end = radii[-1]
            cap_area_end = np.pi * r_end**2
            surface_area_px2 += cap_area_end
        
        surface_area_um2 = surface_area_px2 * (self.pixel_size_um ** 2)
        
        if return_details:
            details = {
                'centerline': centerline,
                'radii': radii,
                'axis_start': axis_start,
                'axis_end': axis_end,
                'axis_angle': axis_angle,
                'n_sections': len(radii),
                'volume_um3': volume_um3,
                'surface_area_um2': surface_area_um2,
                'contour': contour
            }
            return volume_um3, details
        
        return volume_um3
    
    def visualize_segmentation(self, mask, details, save_path='segmentation_visualization.png'):
        """
        Visualize segmentation results
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Original mask + contour
        ax = axes[0]
        ax.imshow(mask, cmap='gray', alpha=0.5)
        
        # Contour
        contour = details['contour']
        ax.plot(contour[:, 0], contour[:, 1], 'b-', linewidth=2, label='Contour')
        
        # Long axis
        axis_start = details['axis_start']
        axis_end = details['axis_end']
        ax.plot([axis_start[0], axis_end[0]], [axis_start[1], axis_end[1]], 
               'r-', linewidth=2, label='Long axis')
        
        # Centerline
        centerline = details['centerline']
        ax.plot(centerline[:, 0], centerline[:, 1], 'g-', linewidth=2, label='Centerline')
        
        ax.set_title('Cell Outline and Centerline', fontsize=14, fontweight='bold')
        ax.legend()
        ax.axis('equal')
        ax.set_xlim(np.min(contour[:, 0]) - 10, np.max(contour[:, 0]) + 10)
        ax.set_ylim(np.min(contour[:, 1]) - 10, np.max(contour[:, 1]) + 10)
        
        # Cross-sections and radii
        ax = axes[1]
        ax.imshow(mask, cmap='gray', alpha=0.5)
        
        # Draw cross-section lines
        centerline = details['centerline']
        radii = details['radii']
        section_angle = details['axis_angle'] + np.pi/2
        
        for i in range(len(centerline)):
            center = centerline[i]
            radius = radii[i]
            
            # Cross-section line
            dx = radius * np.cos(section_angle)
            dy = radius * np.sin(section_angle)
            
            ax.plot([center[0] - dx, center[0] + dx], 
                   [center[1] - dy, center[1] + dy], 
                   'c-', linewidth=1, alpha=0.5)
            
            # Draw circle (representing rotational symmetry)
            circle = plt.Circle((center[0], center[1]), radius, 
                               fill=False, color='yellow', linewidth=1, alpha=0.5)
            ax.add_patch(circle)
        
        ax.plot(centerline[:, 0], centerline[:, 1], 'g-', linewidth=2, label='Centerline')
        
        ax.set_title(f'Cross-sections (n={len(radii)})\nVolume={details["volume_um3"]:.2f} um^3', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.axis('equal')
        ax.set_xlim(np.min(contour[:, 0]) - 10, np.max(contour[:, 0]) + 10)
        ax.set_ylim(np.min(contour[:, 1]) - 10, np.max(contour[:, 1]) + 10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved: {save_path}")
        plt.show()


def demo_with_test_cell():
    """Demo with a test cell"""
    print("\n" + "="*60)
    print("DEMO: Rotational Symmetry Volume Estimation")
    print("="*60)
    
    # Create estimator
    estimator = RotationalSymmetryVolumeEstimator(
        pixel_size_um=0.08625,
        section_interval_um=0.25  # 250 nm
    )
    
    # Create elongated ellipse for testing (fission yeast-like shape)
    from skimage.draw import ellipse
    
    img = np.zeros((300, 300), dtype=np.uint8)
    
    # Vertically elongated ellipse
    rr, cc = ellipse(150, 150, 100, 30, rotation=np.deg2rad(15))
    img[rr, cc] = 255
    
    mask = img > 0
    
    print(f"\n=== Test Cell ===")
    print(f"Mask shape: {mask.shape}")
    print(f"Foreground pixels: {np.sum(mask)}")
    
    # Compute volume
    volume, details = estimator.compute_cell_volume(mask, return_details=True)
    
    print(f"\n=== Results ===")
    print(f"Volume: {volume:.2f} um^3")
    print(f"Surface area: {details['surface_area_um2']:.2f} um^2")
    print(f"Number of sections: {details['n_sections']}")
    print(f"Mean radius: {np.mean(details['radii']) * estimator.pixel_size_um:.3f} um")
    
    # Visualize
    estimator.visualize_segmentation(mask, details, 'rotational_symmetry_demo.png')
    
    return estimator, details


def compare_with_pomegranate():
    """Comparison with the Pomegranate algorithm"""
    print("\n" + "="*60)
    print("COMPARISON: Rotational Symmetry vs Pomegranate")
    print("="*60)
    
    # Create both estimators
    from timeseries_volume_from_roiset import TimeSeriesVolumeTracker
    
    rot_estimator = RotationalSymmetryVolumeEstimator(
        pixel_size_um=0.08625,
        section_interval_um=0.25
    )
    
    pom_tracker = TimeSeriesVolumeTracker(
        roi_zip_path="RoiSet.zip",
        voxel_xy=0.08625,
        voxel_z=0.08625,
        image_width=512,
        image_height=512
    )
    
    # Test ellipse
    from skimage.draw import ellipse
    
    results_comparison = []
    
    # Test with various sizes and shapes
    test_cases = [
        {'r_major': 60, 'r_minor': 20, 'name': 'Elongated (3:1)'},
        {'r_major': 50, 'r_minor': 30, 'name': 'Moderate (5:3)'},
        {'r_major': 40, 'r_minor': 40, 'name': 'Circular (1:1)'},
    ]
    
    for case in test_cases:
        img = np.zeros((200, 200), dtype=np.uint8)
        rr, cc = ellipse(100, 100, case['r_major'], case['r_minor'])
        img[rr, cc] = 255
        mask = img > 0
        
        # Rotational symmetry method
        vol_rot = rot_estimator.compute_cell_volume(mask)
        
        # Pomegranate method (simplified version)
        # Create ROI info
        roi_info = {
            'name': case['name'],
            'type': 2,  # Ellipse
            'left': 100 - case['r_minor'],
            'top': 100 - case['r_major'],
            'right': 100 + case['r_minor'],
            'bottom': 100 + case['r_major'],
            'width': 2 * case['r_minor'],
            'height': 2 * case['r_major'],
            'n_coordinates': 0,
            'bytes': b''
        }
        
        # Use mask directly
        distance_map = ndimage.distance_transform_edt(mask)
        max_dist = np.max(distance_map)
        
        # Simplified Pomegranate volume estimation
        elongation_factor = pom_tracker.elongation_factor
        z_slices = int(2 * (np.ceil(max_dist * elongation_factor) + 2))
        
        # Approximation: ellipsoid volume
        vol_pom_approx = (4/3) * np.pi * case['r_major'] * case['r_minor'] * max_dist * (pom_tracker.voxel_xy**3)
        
        results_comparison.append({
            'name': case['name'],
            'vol_rotational': vol_rot,
            'vol_pomegranate': vol_pom_approx,
            'ratio': vol_rot / vol_pom_approx if vol_pom_approx > 0 else np.nan
        })
        
        print(f"\n{case['name']}:")
        print(f"  Rotational Symmetry: {vol_rot:.2f} um^3")
        print(f"  Pomegranate (approx): {vol_pom_approx:.2f} um^3")
        print(f"  Ratio: {vol_rot / vol_pom_approx:.3f}")
    
    return results_comparison


if __name__ == "__main__":
    # Run demo
    estimator, details = demo_with_test_cell()

    # Run comparison
    # comparison = compare_with_pomegranate()
    
    print("\n" + "="*60)
    print("DEMO COMPLETE!")
    print("="*60)
    print("\nThis algorithm:")
    print("  - Assumes rotational symmetry")
    print("  - Better for elongated cells (fission yeast)")
    print("  - Based on Odermatt et al. (2021) eLife")
    print("\nPomegranate algorithm:")
    print("  - Uses Distance Transform + Skeleton")
    print("  - Spherical cross-sections")
    print("  - Better for irregular shapes")


# %%
