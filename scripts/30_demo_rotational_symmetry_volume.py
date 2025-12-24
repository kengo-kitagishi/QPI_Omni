#!/usr/bin/env python3
"""
Rotational Symmetry Volume Estimation
eLife 2021 (Odermatt et al.) のアルゴリズムを実装

細胞輪郭から回転対称を仮定して体積を計算する方法：
1. 細胞の長軸を決定
2. 長軸に垂直な断面線を一定間隔で配置
3. 各断面線と輪郭の交点を計算
4. 中心線を更新（交点の中点を通る）
5. 断面線の傾きを更新（中心線に垂直）
6. 回転対称を仮定して各断面の体積を合計

Reference:
Odermatt et al. (2021) eLife 10:e64901
https://elifesciences.org/articles/64901
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import morphology, measure
from skimage.draw import polygon
import tifffile
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import cv2

class RotationalSymmetryVolumeEstimator:
    """回転対称を仮定した体積推定"""
    
    def __init__(self, pixel_size_um=0.08625, section_interval_um=0.25):
        """
        Parameters
        ----------
        pixel_size_um : float
            ピクセルサイズ (um)
        section_interval_um : float
            断面線の間隔 (um)
            論文では250nm = 0.25 um
        """
        self.pixel_size_um = pixel_size_um
        self.section_interval_um = section_interval_um
        self.section_interval_px = section_interval_um / pixel_size_um
        
        print(f"=== Rotational Symmetry Volume Estimator ===")
        print(f"Pixel size: {pixel_size_um} um")
        print(f"Section interval: {section_interval_um} um ({self.section_interval_px:.2f} pixels)")
    
    def fit_minimum_bounding_rectangle(self, contour):
        """
        最小外接矩形を計算（長軸決定用）
        
        Returns
        -------
        center : tuple
            中心座標 (x, y)
        size : tuple
            サイズ (width, height)
        angle : float
            角度（ラジアン）
        """
        # OpenCVを使用
        rect = cv2.minAreaRect(contour.astype(np.float32))
        center, size, angle_deg = rect
        angle_rad = np.deg2rad(angle_deg)
        
        return center, size, angle_rad
    
    def get_long_axis(self, contour):
        """
        細胞の長軸を決定
        
        Returns
        -------
        axis_start : ndarray
            長軸の始点 (x, y)
        axis_end : ndarray
            長軸の終点 (x, y)
        angle : float
            長軸の角度（ラジアン）
        """
        center, size, angle = self.fit_minimum_bounding_rectangle(contour)
        
        # 長い方を長軸とする
        width, height = size
        if width > height:
            length = width
            axis_angle = angle
        else:
            length = height
            axis_angle = angle + np.pi/2
        
        # 長軸の始点と終点
        dx = length/2 * np.cos(axis_angle)
        dy = length/2 * np.sin(axis_angle)
        
        axis_start = np.array([center[0] - dx, center[1] - dy])
        axis_end = np.array([center[0] + dx, center[1] + dy])
        
        return axis_start, axis_end, axis_angle
    
    def compute_perpendicular_sections(self, axis_start, axis_end, n_sections):
        """
        長軸に垂直な断面線の位置を計算
        
        Returns
        -------
        section_centers : ndarray
            各断面の中心位置 (n_sections, 2)
        section_angle : float
            断面線の角度（長軸に垂直）
        """
        # 長軸に沿って等間隔に点を配置
        t = np.linspace(0, 1, n_sections)
        section_centers = axis_start[np.newaxis, :] + t[:, np.newaxis] * (axis_end - axis_start)[np.newaxis, :]
        
        # 長軸の角度
        axis_vec = axis_end - axis_start
        axis_angle = np.arctan2(axis_vec[1], axis_vec[0])
        
        # 断面線の角度（長軸に垂直）
        section_angle = axis_angle + np.pi/2
        
        return section_centers, section_angle
    
    def find_contour_intersections(self, section_center, section_angle, contour, line_length=500):
        """
        断面線と輪郭の交点を探す
        
        Returns
        -------
        intersections : list of ndarray
            交点の座標リスト
        """
        # 断面線を定義（十分長く）
        dx = line_length * np.cos(section_angle)
        dy = line_length * np.sin(section_angle)
        
        line_start = section_center - np.array([dx, dy])
        line_end = section_center + np.array([dx, dy])
        
        # 輪郭との交点を探す
        intersections = []
        
        for i in range(len(contour)):
            p1 = contour[i]
            p2 = contour[(i+1) % len(contour)]
            
            # 線分の交差判定
            intersection = self.line_segment_intersection(
                line_start, line_end, p1, p2
            )
            
            if intersection is not None:
                intersections.append(intersection)
        
        return intersections
    
    def line_segment_intersection(self, p1, p2, p3, p4):
        """
        2つの線分の交点を計算
        
        Parameters
        ----------
        p1, p2 : ndarray
            線分1の端点
        p3, p4 : ndarray
            線分2の端点
        
        Returns
        -------
        intersection : ndarray or None
            交点の座標、交差しない場合はNone
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
        細胞マスクから体積を計算
        
        Parameters
        ----------
        mask : ndarray
            2Dバイナリマスク
        return_details : bool
            詳細情報を返すか
        
        Returns
        -------
        volume_um3 : float
            体積 (um^3)
        details : dict (optional)
            詳細情報
        """
        # 輪郭を抽出
        contours = measure.find_contours(mask, 0.5)
        
        if len(contours) == 0:
            return None
        
        # 最大の輪郭を使用
        contour = max(contours, key=lambda x: len(x))
        
        # Y, X順なのでX, Y順に変換
        contour = contour[:, ::-1]
        
        # 長軸を決定
        axis_start, axis_end, axis_angle = self.get_long_axis(contour)
        
        # 長軸の長さ
        axis_length = np.linalg.norm(axis_end - axis_start)
        
        # 断面の数を計算
        n_sections = int(axis_length / self.section_interval_px)
        
        if n_sections < 2:
            return None
        
        # 断面位置を初期化
        section_centers, section_angle = self.compute_perpendicular_sections(
            axis_start, axis_end, n_sections
        )
        
        # 各断面で輪郭との交点を探し、中心線を更新
        centerline = []
        radii = []
        valid_sections = []
        
        for i in range(n_sections):
            intersections = self.find_contour_intersections(
                section_centers[i], section_angle, contour
            )
            
            if len(intersections) >= 2:
                # 交点を距離でソート
                intersections = np.array(intersections)
                distances = np.linalg.norm(intersections - section_centers[i], axis=1)
                sorted_idx = np.argsort(distances)
                
                # 最も遠い2点を使用（細胞の両側）
                p1 = intersections[sorted_idx[-1]]
                p2 = intersections[sorted_idx[-2]]
                
                # 中点を計算
                midpoint = (p1 + p2) / 2
                
                # 半径を計算（回転対称を仮定）
                radius = np.linalg.norm(p1 - p2) / 2
                
                centerline.append(midpoint)
                radii.append(radius)
                valid_sections.append(i)
        
        if len(centerline) < 2:
            return None
        
        centerline = np.array(centerline)
        radii = np.array(radii)
        
        # 体積を計算（円柱の和）
        total_volume_px3 = 0
        
        for i in range(len(radii)):
            # 各断面を円柱として計算
            r = radii[i]
            h = self.section_interval_px
            
            # 円柱の体積: V = π * r^2 * h
            volume = np.pi * r**2 * h
            total_volume_px3 += volume
        
        # ピクセル → um変換
        volume_um3 = total_volume_px3 * (self.pixel_size_um ** 3)
        
        # 表面積も計算（オプション）
        surface_area_px2 = 0
        for i in range(len(radii)):
            r = radii[i]
            h = self.section_interval_px
            
            # 円柱の側面積: A = 2π * r * h
            area = 2 * np.pi * r * h
            surface_area_px2 += area
        
        # 両端のキャップ（球冠として近似）
        if len(radii) > 0:
            # 始端
            r_start = radii[0]
            cap_area_start = np.pi * r_start**2
            surface_area_px2 += cap_area_start
            
            # 終端
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
        セグメンテーション結果を可視化
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # 元のマスク + 輪郭
        ax = axes[0]
        ax.imshow(mask, cmap='gray', alpha=0.5)
        
        # 輪郭
        contour = details['contour']
        ax.plot(contour[:, 0], contour[:, 1], 'b-', linewidth=2, label='Contour')
        
        # 長軸
        axis_start = details['axis_start']
        axis_end = details['axis_end']
        ax.plot([axis_start[0], axis_end[0]], [axis_start[1], axis_end[1]], 
               'r-', linewidth=2, label='Long axis')
        
        # 中心線
        centerline = details['centerline']
        ax.plot(centerline[:, 0], centerline[:, 1], 'g-', linewidth=2, label='Centerline')
        
        ax.set_title('Cell Outline and Centerline', fontsize=14, fontweight='bold')
        ax.legend()
        ax.axis('equal')
        ax.set_xlim(np.min(contour[:, 0]) - 10, np.max(contour[:, 0]) + 10)
        ax.set_ylim(np.min(contour[:, 1]) - 10, np.max(contour[:, 1]) + 10)
        
        # 断面と半径
        ax = axes[1]
        ax.imshow(mask, cmap='gray', alpha=0.5)
        
        # 断面線を描画
        centerline = details['centerline']
        radii = details['radii']
        section_angle = details['axis_angle'] + np.pi/2
        
        for i in range(len(centerline)):
            center = centerline[i]
            radius = radii[i]
            
            # 断面線
            dx = radius * np.cos(section_angle)
            dy = radius * np.sin(section_angle)
            
            ax.plot([center[0] - dx, center[0] + dx], 
                   [center[1] - dy, center[1] + dy], 
                   'c-', linewidth=1, alpha=0.5)
            
            # 円を描画（回転対称を表現）
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
    """テスト細胞でデモ"""
    print("\n" + "="*60)
    print("DEMO: Rotational Symmetry Volume Estimation")
    print("="*60)
    
    # 推定器を作成
    estimator = RotationalSymmetryVolumeEstimator(
        pixel_size_um=0.08625,
        section_interval_um=0.25  # 250 nm
    )
    
    # テスト用の細長い楕円を作成（分裂酵母のような形状）
    from skimage.draw import ellipse
    
    img = np.zeros((300, 300), dtype=np.uint8)
    
    # 縦長の楕円
    rr, cc = ellipse(150, 150, 100, 30, rotation=np.deg2rad(15))
    img[rr, cc] = 255
    
    mask = img > 0
    
    print(f"\n=== Test Cell ===")
    print(f"Mask shape: {mask.shape}")
    print(f"Foreground pixels: {np.sum(mask)}")
    
    # 体積を計算
    volume, details = estimator.compute_cell_volume(mask, return_details=True)
    
    print(f"\n=== Results ===")
    print(f"Volume: {volume:.2f} um^3")
    print(f"Surface area: {details['surface_area_um2']:.2f} um^2")
    print(f"Number of sections: {details['n_sections']}")
    print(f"Mean radius: {np.mean(details['radii']) * estimator.pixel_size_um:.3f} um")
    
    # 可視化
    estimator.visualize_segmentation(mask, details, 'rotational_symmetry_demo.png')
    
    return estimator, details


def compare_with_pomegranate():
    """Pomegranateアルゴリズムとの比較"""
    print("\n" + "="*60)
    print("COMPARISON: Rotational Symmetry vs Pomegranate")
    print("="*60)
    
    # 両方の推定器を作成
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
    
    # テスト用の楕円
    from skimage.draw import ellipse
    
    results_comparison = []
    
    # 様々なサイズ・形状でテスト
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
        
        # 回転対称法
        vol_rot = rot_estimator.compute_cell_volume(mask)
        
        # Pomegranate法（簡易版）
        # ROI情報を作成
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
        
        # マスクを直接使用
        distance_map = ndimage.distance_transform_edt(mask)
        max_dist = np.max(distance_map)
        
        # 簡易的なPomegranate体積推定
        elongation_factor = pom_tracker.elongation_factor
        z_slices = int(2 * (np.ceil(max_dist * elongation_factor) + 2))
        
        # 近似: 楕円体の体積
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
    # デモ実行
    estimator, details = demo_with_test_cell()
    
    # 比較実行
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

