#!/usr/bin/env python3
"""
2D to 3D Reconstruction - Pomegranate Algorithm Analysis
Detailed analysis and implementation of the Pomegranate 3D reconstruction algorithm

Principle:
1. Distance Transform: Distance from each pixel to the boundary = local radius
2. Medial Axis Transform: Skeleton + Distance Map
3. Spherical Cross-Section: r(z) = sqrt(R^2 - z^2)
4. 3D Reconstruction: Expand spheres from each medial axis pixel
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import morphology, measure
import tifffile
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
import os
from figure_logger import setup_autosave
setup_autosave()

class TwoD_to_ThreeD_Reconstructor:
    """2D to 3D reconstruction using the Pomegranate algorithm"""
    
    def __init__(self, voxel_xy=0.1, voxel_z=0.3, radius_enlarge=1.0):
        """
        Parameters
        ----------
        voxel_xy : float
            Pixel size in XY direction (um)
        voxel_z : float
            Step size in Z direction (um)
        radius_enlarge : float
            Radius enlargement amount (pixels)
        """
        self.voxel_xy = voxel_xy
        self.voxel_z = voxel_z
        self.radius_enlarge = radius_enlarge
        self.elongation_factor = voxel_xy / voxel_z
        
        print(f"=== 2D to 3D Reconstructor ===")
        print(f"Voxel XY: {voxel_xy} um")
        print(f"Voxel Z: {voxel_z} um")
        print(f"Elongation Factor: {self.elongation_factor:.4f}")
        print(f"Radius Enlargement: {radius_enlarge} pixels")
    
    def load_2d_image(self, image_path):
        """Load a 2D image and binarize it"""
        print(f"\n=== Loading Image ===")
        print(f"Path: {image_path}")
        
        # Load image
        img = tifffile.imread(image_path)
        
        # For 2D images
        if img.ndim == 2:
            self.image_2d = img
        # For 3D images, use max projection
        elif img.ndim == 3:
            print(f"  Input is 3D ({img.shape}), using max projection")
            self.image_2d = np.max(img, axis=0)
        else:
            raise ValueError(f"Unsupported image dimensions: {img.ndim}")
        
        # Binarize (Otsu)
        from skimage.filters import threshold_otsu
        threshold = threshold_otsu(self.image_2d)
        self.binary_2d = self.image_2d > threshold
        
        print(f"  Shape: {self.binary_2d.shape}")
        print(f"  Foreground pixels: {np.sum(self.binary_2d)}")
        
        return self.binary_2d
    
    def create_from_roi_mask(self, binary_mask):
        """Create directly from an ROI mask"""
        self.binary_2d = binary_mask.astype(bool)
        print(f"=== Binary Mask Input ===")
        print(f"  Shape: {self.binary_2d.shape}")
        print(f"  Foreground pixels: {np.sum(self.binary_2d)}")
        return self.binary_2d
    
    def compute_distance_map(self):
        """Compute the Distance Transform"""
        print(f"\n=== Step 1: Distance Transform ===")
        
        # Distance Transform (EDT: Euclidean Distance Transform)
        self.distance_map = ndimage.distance_transform_edt(self.binary_2d)
        
        # Statistics
        max_dist = np.max(self.distance_map)
        mean_dist = np.mean(self.distance_map[self.binary_2d])
        
        print(f"  Max Distance: {max_dist:.2f} pixels ({max_dist * self.voxel_xy:.3f} um)")
        print(f"  Mean Distance: {mean_dist:.2f} pixels ({mean_dist * self.voxel_xy:.3f} um)")
        
        # Estimate the number of Z slices
        self.z_slices = int(2 * (np.ceil(max_dist * self.elongation_factor) + 2))
        self.mid_slice = self.z_slices // 2
        
        print(f"  Estimated Z Slices: {self.z_slices}")
        print(f"  Mid Slice: {self.mid_slice}")
        
        return self.distance_map
    
    def compute_skeleton(self):
        """Compute the Skeleton (skeletonization)"""
        print(f"\n=== Step 2: Skeletonization ===")
        
        # Skeletonize (Zhang's algorithm)
        self.skeleton = morphology.skeletonize(self.binary_2d)
        
        skeleton_pixels = np.sum(self.skeleton)
        print(f"  Skeleton pixels: {skeleton_pixels}")
        print(f"  Reduction ratio: {skeleton_pixels / np.sum(self.binary_2d):.4f}")
        
        return self.skeleton
    
    def compute_medial_axis(self):
        """Compute the Medial Axis Transform"""
        print(f"\n=== Step 3: Medial Axis Transform ===")
        
        # Medial Axis = Skeleton AND Distance Map
        self.medial_axis = self.skeleton * self.distance_map
        
        medial_pixels = np.sum(self.skeleton)
        total_radius = np.sum(self.medial_axis)
        avg_radius = total_radius / medial_pixels if medial_pixels > 0 else 0
        
        print(f"  Medial axis pixels: {medial_pixels}")
        print(f"  Average radius: {avg_radius:.2f} pixels ({avg_radius * self.voxel_xy:.3f} um)")
        
        return self.medial_axis
    
    def spherical_cross_section_radius(self, r0, z_distance):
        """
        Compute the cross-section radius of a sphere

        Sphere equation: x^2 + y^2 + z^2 = R^2
        Cross-section at z-plane: x^2 + y^2 = R^2 - z^2
        Cross-section radius: r(z) = sqrt(R^2 - z^2)

        Parameters
        ----------
        r0 : float
            Base radius (pixels)
        z_distance : float
            Z distance from center (pixels, corrected for elongation factor)

        Returns
        -------
        float
            Cross-section radius (pixels), 0 if negative
        """
        r_squared = r0**2 - z_distance**2
        return np.sqrt(r_squared) if r_squared > 0 else 0
    
    def reconstruct_3d(self):
        """Perform 3D reconstruction"""
        print(f"\n=== Step 4: 3D Reconstruction ===")
        
        # Initialize 3D stack
        height, width = self.binary_2d.shape
        self.stack_3d = np.zeros((self.z_slices, height, width), dtype=np.uint8)
        
        # Get medial axis coordinates
        medial_coords = np.argwhere(self.skeleton)
        total_voxels = len(medial_coords)
        
        print(f"  Processing {total_voxels} medial axis voxels...")
        
        # Process each medial axis pixel
        processed_count = 0
        for idx, (y, x) in enumerate(medial_coords):
            if idx % 100 == 0:
                print(f"    Progress: {idx}/{total_voxels} ({100*idx/total_voxels:.1f}%)")
            
            # Get base radius
            r0 = self.medial_axis[y, x] + self.radius_enlarge
            
            # Draw spherical cross-sections for each Z slice
            for z in range(self.z_slices):
                # Compute Z distance (with elongation factor correction)
                z_distance = (self.mid_slice - z) / self.elongation_factor
                
                # Compute the cross-section radius of the sphere
                segment_radius = self.spherical_cross_section_radius(r0, z_distance)
                
                # Threshold check
                if segment_radius > 2 * self.voxel_xy:
                    # Draw circle
                    try:
                        from skimage.draw import disk as draw_disk
                        rr, cc = draw_disk((y, x), segment_radius, shape=(height, width))
                        
                        # Only pixels within bounds
                        self.stack_3d[z, rr, cc] = 255
                        
                        processed_count += 1
                    except Exception as e:
                        # Skip if out of bounds
                        pass
        
        print(f"  Successfully processed: {processed_count} voxel-slice pairs")
        print(f"  3D Stack shape: {self.stack_3d.shape}")
        print(f"  Total foreground voxels: {np.sum(self.stack_3d > 0)}")
        
        return self.stack_3d
    
    def run_full_pipeline(self, input_source, is_file=True):
        """Run the full pipeline"""
        print("\n" + "="*50)
        print("FULL PIPELINE EXECUTION")
        print("="*50)
        
        # Load input
        if is_file:
            self.load_2d_image(input_source)
        else:
            self.create_from_roi_mask(input_source)
        
        # Execute pipeline
        self.compute_distance_map()
        self.compute_skeleton()
        self.compute_medial_axis()
        self.reconstruct_3d()
        
        print("\n" + "="*50)
        print("RECONSTRUCTION COMPLETE")
        print("="*50)
        
        return self.stack_3d
    
    def save_results(self, output_dir="reconstruction_output"):
        """Save results"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n=== Saving Results to {output_dir} ===")
        
        # 3D Stack
        output_path = os.path.join(output_dir, "3D_reconstruction.tif")
        tifffile.imwrite(output_path, self.stack_3d, 
                        resolution=(1/self.voxel_xy, 1/self.voxel_xy),
                        metadata={'spacing': self.voxel_z, 'unit': 'um'})
        print(f"  Saved: {output_path}")
        
        # 2D Binary
        binary_path = os.path.join(output_dir, "binary_2d.tif")
        tifffile.imwrite(binary_path, (self.binary_2d * 255).astype(np.uint8))
        print(f"  Saved: {binary_path}")
        
        # Distance Map
        dist_path = os.path.join(output_dir, "distance_map.tif")
        tifffile.imwrite(dist_path, self.distance_map.astype(np.float32))
        print(f"  Saved: {dist_path}")
        
        # Skeleton
        skel_path = os.path.join(output_dir, "skeleton.tif")
        tifffile.imwrite(skel_path, (self.skeleton * 255).astype(np.uint8))
        print(f"  Saved: {skel_path}")
        
        # Medial Axis
        medial_path = os.path.join(output_dir, "medial_axis.tif")
        tifffile.imwrite(medial_path, self.medial_axis.astype(np.float32))
        print(f"  Saved: {medial_path}")
    
    def visualize_algorithm(self, save_path="algorithm_visualization.png"):
        """Visualize each step of the algorithm"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Original Binary
        ax = axes[0, 0]
        ax.imshow(self.binary_2d, cmap='gray')
        ax.set_title('Step 0: Binary Input', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # 2. Distance Map
        ax = axes[0, 1]
        im = ax.imshow(self.distance_map, cmap='hot')
        ax.set_title('Step 1: Distance Transform\n(Local Radius Map)', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Distance (pixels)')
        ax.axis('off')
        
        # 3. Skeleton
        ax = axes[0, 2]
        ax.imshow(self.skeleton, cmap='gray')
        ax.set_title('Step 2: Skeleton\n(Center Axis Extraction)', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # 4. Medial Axis Transform
        ax = axes[1, 0]
        im = ax.imshow(self.medial_axis, cmap='hot')
        ax.set_title('Step 3: Medial Axis Transform\n(Center Axis + Radius Info)', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Radius (pixels)')
        ax.axis('off')
        
        # 5. 3D Reconstruction (Mid Slice)
        ax = axes[1, 1]
        ax.imshow(self.stack_3d[self.mid_slice], cmap='gray')
        ax.set_title(f'Step 4: 3D Reconstruction\n(Mid Slice: {self.mid_slice}/{self.z_slices})', 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # 6. Cross-section Function
        ax = axes[1, 2]
        r0_values = np.linspace(5, 20, 3)
        z_range = np.linspace(-25, 25, 100)
        
        for r0 in r0_values:
            radii = [self.spherical_cross_section_radius(r0, z) for z in z_range]
            ax.plot(z_range, radii, linewidth=2, label=f'R₀ = {r0:.1f} px')
        
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Z Distance from Center (pixels)', fontsize=12)
        ax.set_ylabel('Cross-section Radius (pixels)', fontsize=12)
        ax.set_title('Spherical Cross-Section Function\nr(z) = √(R₀² - z²)', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n=== Visualization saved: {save_path} ===")
        plt.show()
    
    def create_test_ellipse(self, width=200, height=200,
                          semi_major=60, semi_minor=40, angle=30):
        """Generate an ellipse image for testing"""
        from skimage.draw import ellipse
        
        img = np.zeros((height, width), dtype=np.uint8)
        rr, cc = ellipse(height//2, width//2, semi_major, semi_minor, 
                        rotation=np.deg2rad(angle))
        img[rr, cc] = 255
        
        self.binary_2d = img > 0
        
        print(f"=== Test Ellipse Created ===")
        print(f"  Size: {width}x{height}")
        print(f"  Semi-major axis: {semi_major} px")
        print(f"  Semi-minor axis: {semi_minor} px")
        print(f"  Angle: {angle}°")
        
        return self.binary_2d


def demo_with_test_ellipse():
    """Run demo with a test ellipse"""
    print("\n" + "="*60)
    print("DEMO: Pomegranate Algorithm with Test Ellipse")
    print("="*60)
    
    # Create Reconstructor
    reconstructor = TwoD_to_ThreeD_Reconstructor(
        voxel_xy=0.1,   # 0.1 um/pixel
        voxel_z=0.3,    # 0.3 um/slice
        radius_enlarge=1.0
    )
    
    # Create test ellipse
    test_ellipse = reconstructor.create_test_ellipse(
        width=300, height=300,
        semi_major=80, semi_minor=50, angle=30
    )
    
    # Execute full pipeline
    stack_3d = reconstructor.run_full_pipeline(test_ellipse, is_file=False)
    
    # Save results
    reconstructor.save_results("demo_output")
    
    # Visualize
    reconstructor.visualize_algorithm("demo_algorithm_steps.png")
    
    return reconstructor


if __name__ == "__main__":
    # Run demo
    reconstructor = demo_with_test_ellipse()
    
    print("\n" + "="*60)
    print("DEMO COMPLETE!")
    print("="*60)
    print("\nOutput files:")
    print("  - demo_output/3D_reconstruction.tif")
    print("  - demo_output/distance_map.tif")
    print("  - demo_output/medial_axis.tif")
    print("  - demo_algorithm_steps.png")


# %%
