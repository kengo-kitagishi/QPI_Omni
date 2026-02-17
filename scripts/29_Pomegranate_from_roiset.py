#!/usr/bin/env python3
"""
Time-series Volume Tracking from ROI Set (Pomegranate互換)
ImageJ ROIセットから時系列の体積変化を追跡

本家Pomegranateアルゴリズムに準拠:
- Baybay et al. (2020) Pomegranate: Nuclear and whole-cell segmentation for fission yeast
- GitHub: https://github.com/erodb/Pomegranate

アルゴリズム:
1. Distance Transform → 局所半径
2. Skeleton → 中心線抽出
3. Medial Axis Transform → Skeleton × Distance Map
4. Spherical Expansion → Z方向に球体を展開
   - 各Zスライスで: r(z) = √(R² - z²)
   - 閾値判定: if r(z) > 2*pixel_size_xy: 描画

離散化方法:
- 'pomegranate': 閾値ベース（本家準拠）
  → 各Zスライスで半径が閾値以上なら描画
- 'round': 四捨五入方式
  → 厚みを計算してround()で変換

処理フロー:
各ROIを2Dマスクに変換 → 3D再構成 → 体積計算 → 厚みマップ生成
"""
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import zipfile
import struct
import tifffile
import os
from pathlib import Path
from scipy import ndimage
from skimage import morphology, measure
from skimage.draw import polygon
import re
from collections import defaultdict

# 日本語フォント設定（エラー回避）
rcParams['font.sans-serif'] = ['Arial']

class TimeSeriesVolumeTracker:
    """ROIセットから時系列体積を追跡"""
    
    def __init__(self, roi_zip_path, voxel_xy=0.1, voxel_z=0.3, 
                 radius_enlarge=1.0, image_width=512, image_height=512,
                 min_radius_threshold_px=2, discretization_method='pomegranate'):
        """
        Parameters
        ----------
        roi_zip_path : str
            ImageJ ROIセット（.zip）のパス
        voxel_xy : float
            XY方向のピクセルサイズ (um)
        voxel_z : float
            Z方向のステップサイズ (um)
        radius_enlarge : float
            半径の拡張量 (pixels) - Pomegranate本家は+1
        image_width : int
            画像の幅（pixels）
        image_height : int
            画像の高さ（pixels）
        min_radius_threshold_px : float
            最小半径閾値（ピクセル）- 本家Pomegranateは2
        discretization_method : str
            離散化方法：'pomegranate'（閾値ベース）または'round'（四捨五入）
        """
        self.roi_zip_path = roi_zip_path
        self.voxel_xy = voxel_xy
        self.voxel_z = voxel_z
        self.radius_enlarge = radius_enlarge
        self.elongation_factor = voxel_xy / voxel_z
        self.image_width = image_width
        self.image_height = image_height
        self.min_radius_threshold_px = min_radius_threshold_px
        self.discretization_method = discretization_method
        
        print(f"=== Time-series Volume Tracker (Pomegranate) ===")
        print(f"ROI Set: {roi_zip_path}")
        print(f"Voxel XY: {voxel_xy} um")
        print(f"Voxel Z: {voxel_z} um")
        print(f"Elongation Factor: {self.elongation_factor:.4f}")
        print(f"Image Size: {image_width} x {image_height}")
        print(f"Radius enlarge: {radius_enlarge} px (Pomegranate本家: +1)")
        print(f"Min radius threshold: {min_radius_threshold_px} px (Pomegranate本家: 2)")
        print(f"Discretization method: {discretization_method}")
        
        # ROIセットを読み込み
        self.load_roi_set()
        
    def load_roi_set(self):
        """ROIセットを読み込んで整理"""
        print(f"\n=== Loading ROI Set ===")
        
        with zipfile.ZipFile(self.roi_zip_path, 'r') as zf:
            roi_names = zf.namelist()
            print(f"  Total ROIs: {len(roi_names)}")
            
            # ROIを解析
            self.rois_by_time = defaultdict(list)
            self.roi_data = []
            
            for idx, roi_name in enumerate(roi_names):
                if idx % 100 == 0:
                    print(f"    Processing: {idx}/{len(roi_names)}")
                
                # ROI名からメタデータを抽出
                # 例: "2438-0026-0048.roi" → フレーム、セルID、その他
                try:
                    roi_bytes = zf.read(roi_name)
                    roi_info = self.parse_roi_basic(roi_bytes, roi_name)
                    
                    if roi_info is not None:
                        self.roi_data.append(roi_info)
                        
                        # 時間でグループ化（ROI名の最初の数字をフレーム番号と仮定）
                        frame_num = self.extract_frame_number(roi_name)
                        self.rois_by_time[frame_num].append(roi_info)
                        
                except Exception as e:
                    print(f"    Warning: Failed to parse {roi_name}: {e}")
                    continue
            
            print(f"  Successfully parsed: {len(self.roi_data)} ROIs")
            print(f"  Time points: {len(self.rois_by_time)}")
            
    def extract_frame_number(self, roi_name):
        """ROI名からフレーム番号を抽出"""
        # 例: "2438-0026-0048.roi" → 2438
        match = re.match(r'(\d+)-', roi_name)
        if match:
            return int(match.group(1))
        return 0
    
    def parse_roi_basic(self, roi_bytes, roi_name):
        """ImageJ ROIの基本情報を解析"""
        if len(roi_bytes) < 64:
            return None
        
        # ImageJ ROI format header
        # https://imagej.nih.gov/ij/developer/source/ij/io/RoiDecoder.java.html
        
        iout = roi_bytes[0:4]
        if iout != b'Iout':
            return None
        
        # Version
        version = struct.unpack('>h', roi_bytes[4:6])[0]
        
        # ROI type
        roi_type = roi_bytes[6]
        
        # Coordinates
        top = struct.unpack('>h', roi_bytes[8:10])[0]
        left = struct.unpack('>h', roi_bytes[10:12])[0]
        bottom = struct.unpack('>h', roi_bytes[12:14])[0]
        right = struct.unpack('>h', roi_bytes[14:16])[0]
        
        # Number of coordinates (for polygon)
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
        """ROIをバイナリマスクに変換"""
        roi_bytes = roi_info['bytes']
        roi_type = roi_info['type']
        
        # マスクを初期化
        mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        
        try:
            if roi_type in [0, 7, 8]:  # Polygon, Freehand, Traced
                n_coords = roi_info['n_coordinates']
                if n_coords > 0 and len(roi_bytes) >= 64 + n_coords * 4:
                    # 座標を読み込み
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
                        # Polygonを描画
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
            print(f"    Warning: Failed to create mask for {roi_info['name']}: {e}")
            return None
        
        return mask > 0
    
    def compute_volume_from_roi(self, roi_info, return_stack=False, return_thickness_map=False):
        """単一ROIから体積を計算"""
        # ROIをマスクに変換
        mask = self.roi_to_mask(roi_info)
        
        if mask is None or np.sum(mask) == 0:
            return None
        
        # Distance Transform
        distance_map = ndimage.distance_transform_edt(mask)
        max_dist = np.max(distance_map)
        
        if max_dist == 0:
            return None
        
        # Z方向のスライス数を推定
        z_slices = int(2 * (np.ceil(max_dist * self.elongation_factor) + 2))
        mid_slice = z_slices // 2
        
        # Skeleton
        skeleton = morphology.skeletonize(mask)
        
        # Medial Axis
        medial_axis = skeleton * distance_map
        
        # 3D再構成
        stack_3d = np.zeros((z_slices, self.image_height, self.image_width), dtype=np.uint8)
        
        medial_coords = np.argwhere(skeleton)
        
        for y, x in medial_coords:
            # 本家Pomegranate: rinput (局所半径) + radius_enlarge
            r0 = medial_axis[y, x] + self.radius_enlarge
            
            for z in range(z_slices):
                # 本家Pomegranate: zinput = (mid - k) / efactor
                z_distance = (mid_slice - z) / self.elongation_factor
                r_squared = r0**2 - z_distance**2
                
                if r_squared > 0:
                    # 本家Pomegranate: segmentRadius = √(R² - z²) + 1 ← +1は既にr0に含まれる
                    segment_radius = np.sqrt(r_squared)
                    
                    # 本家Pomegranate閾値判定: segmentRadius > (2 * nvx)
                    if segment_radius > self.min_radius_threshold_px:
                        try:
                            from skimage.draw import disk as draw_disk
                            rr, cc = draw_disk((y, x), segment_radius, 
                                             shape=(self.image_height, self.image_width))
                            stack_3d[z, rr, cc] = 255
                        except:
                            pass
        
        # 厚みマップを作成（各XY位置でのZ方向の占有スライス数）
        thickness_map = np.sum(stack_3d > 0, axis=0).astype(np.float32)
        
        # 体積計算
        voxel_volume = self.voxel_xy * self.voxel_xy * self.voxel_z  # um^3
        total_voxels = np.sum(stack_3d > 0)
        volume_um3 = total_voxels * voxel_volume
        
        result = {
            'roi_name': roi_info['name'],
            'area_2d': np.sum(mask),
            'max_radius': max_dist,
            'z_slices': z_slices,
            'total_voxels': total_voxels,
            'volume_um3': volume_um3,
            'thickness_map': thickness_map  # 常に含める
        }
        
        if return_stack:
            result['stack_3d'] = stack_3d
            result['mask_2d'] = mask
        
        return result
    
    def track_volume_timeseries(self, max_frames=None, save_stacks=False, save_thickness_maps=True):
        """時系列で体積を追跡"""
        print(f"\n=== Tracking Volume Time-series ===")
        
        # 時間順にソート
        time_points = sorted(self.rois_by_time.keys())
        
        if max_frames is not None:
            time_points = time_points[:max_frames]
            print(f"  Processing first {max_frames} frames")
        
        print(f"  Time points to process: {len(time_points)}")
        
        results = []
        self.thickness_maps = []  # 厚みマップを保存
        
        for t_idx, t in enumerate(time_points):
            print(f"\n  Frame {t_idx+1}/{len(time_points)} (t={t})")
            
            rois_at_t = self.rois_by_time[t]
            print(f"    ROIs at this time: {len(rois_at_t)}")
            
            for cell_idx, roi_info in enumerate(rois_at_t):
                if cell_idx % 10 == 0:
                    print(f"      Cell {cell_idx+1}/{len(rois_at_t)}")
                
                print(f"        ROI: {roi_info['name']}, type={roi_info['type']}, coords={roi_info['n_coordinates']}")
                
                vol_result = self.compute_volume_from_roi(roi_info, return_stack=save_stacks)
                
                if vol_result is not None:
                    vol_result['time_point'] = t
                    vol_result['time_index'] = t_idx
                    vol_result['cell_index'] = cell_idx
                    
                    # 厚みマップを保存
                    if save_thickness_maps:
                        thickness_info = {
                            'time_point': t,
                            'time_index': t_idx,
                            'cell_index': cell_idx,
                            'roi_name': roi_info['name'],
                            'thickness_map': vol_result['thickness_map']
                        }
                        self.thickness_maps.append(thickness_info)
                    
                    # thickness_mapはDataFrameに含めない（大きすぎるため）
                    vol_result_for_df = {k: v for k, v in vol_result.items() if k != 'thickness_map'}
                    results.append(vol_result_for_df)
                    
                    print(f"        [OK] Volume: {vol_result['volume_um3']:.2f} um^3, Max thickness: {np.max(vol_result['thickness_map']):.1f} slices")
                else:
                    print(f"        [FAIL] Failed to compute volume")
        
        # DataFrameに変換
        self.results_df = pd.DataFrame(results)
        
        print(f"\n  Total processed: {len(self.results_df)} cells")
        
        if len(self.results_df) > 0:
            print(f"  Volume range: {self.results_df['volume_um3'].min():.2f} - {self.results_df['volume_um3'].max():.2f} um^3")
        else:
            print("  Warning: No cells processed successfully")
        
        return self.results_df
    
    def plot_volume_timeseries(self, save_path='volume_timeseries.png'):
        """体積の時系列をプロット"""
        if not hasattr(self, 'results_df'):
            print("Error: Run track_volume_timeseries() first")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 全体の体積変化
        ax = axes[0, 0]
        for cell_idx in self.results_df['cell_index'].unique():
            cell_data = self.results_df[self.results_df['cell_index'] == cell_idx]
            ax.plot(cell_data['time_index'], cell_data['volume_um3'], 
                   alpha=0.5, linewidth=1)
        
        ax.set_xlabel('Time (frame)', fontsize=12)
        ax.set_ylabel('Volume (um^3)', fontsize=12)
        ax.set_title('Volume Time-series (All Cells)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 2. 平均体積変化
        ax = axes[0, 1]
        mean_vol = self.results_df.groupby('time_index')['volume_um3'].mean()
        std_vol = self.results_df.groupby('time_index')['volume_um3'].std()
        
        time_idx = mean_vol.index
        ax.plot(time_idx, mean_vol, 'b-', linewidth=2, label='Mean')
        ax.fill_between(time_idx, mean_vol - std_vol, mean_vol + std_vol, 
                        alpha=0.3, color='blue', label='±1 SD')
        
        ax.set_xlabel('Time (frame)', fontsize=12)
        ax.set_ylabel('Volume (um^3)', fontsize=12)
        ax.set_title('Mean Volume Time-series', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 3. 体積分布の変化
        ax = axes[1, 0]
        time_points = sorted(self.results_df['time_index'].unique())
        
        # 各時間点での分布をバイオリンプロット
        data_for_violin = [self.results_df[self.results_df['time_index'] == t]['volume_um3'].values 
                          for t in time_points[::max(1, len(time_points)//20)]]
        
        positions = list(range(len(data_for_violin)))
        parts = ax.violinplot(data_for_violin, positions=positions, 
                             showmeans=True, showmedians=True)
        
        ax.set_xlabel('Time (subsampled)', fontsize=12)
        ax.set_ylabel('Volume (um^3)', fontsize=12)
        ax.set_title('Volume Distribution Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. 2D面積 vs 3D体積
        ax = axes[1, 1]
        ax.scatter(self.results_df['area_2d'], self.results_df['volume_um3'], 
                  alpha=0.3, s=10)
        
        ax.set_xlabel('2D Area (pixels)', fontsize=12)
        ax.set_ylabel('3D Volume (um^3)', fontsize=12)
        ax.set_title('2D Area vs 3D Volume', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 回帰線
        from scipy.stats import linregress
        slope, intercept, r_value, _, _ = linregress(
            self.results_df['area_2d'], self.results_df['volume_um3'])
        x_range = np.array([self.results_df['area_2d'].min(), 
                           self.results_df['area_2d'].max()])
        ax.plot(x_range, slope * x_range + intercept, 'r--', 
               linewidth=2, label=f'R² = {r_value**2:.3f}')
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n=== Saved plot: {save_path} ===")
        plt.show()
    
    def save_results(self, output_dir='timeseries_volume_output'):
        """結果を保存"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n=== Saving Results to {output_dir} ===")
        
        # CSV
        csv_path = os.path.join(output_dir, 'volume_timeseries.csv')
        self.results_df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")
        
        # 厚みマップをTIFFスタックとして保存
        if hasattr(self, 'thickness_maps') and len(self.thickness_maps) > 0:
            thickness_dir = os.path.join(output_dir, 'thickness_maps')
            os.makedirs(thickness_dir, exist_ok=True)
            
            print(f"\n  Saving thickness maps ({len(self.thickness_maps)} maps)...")
            
            # 個別のTIFFファイルとして保存
            for idx, thick_info in enumerate(self.thickness_maps):
                if idx % 100 == 0:
                    print(f"    Progress: {idx}/{len(self.thickness_maps)}")
                
                roi_name = thick_info['roi_name'].replace('.roi', '')
                thick_path = os.path.join(thickness_dir, f"{roi_name}_thickness.tif")
                
                # 厚みマップを保存（浮動小数点）
                tifffile.imwrite(thick_path, thick_info['thickness_map'])
            
            print(f"  Saved: {len(self.thickness_maps)} thickness maps to {thickness_dir}/")
            
            # 統合スタック（全時間点）を作成
            if len(self.thickness_maps) > 0:
                stack_list = [tm['thickness_map'] for tm in self.thickness_maps]
                stack_array = np.stack(stack_list, axis=0)
                
                stack_path = os.path.join(output_dir, 'thickness_stack_all_frames.tif')
                tifffile.imwrite(stack_path, stack_array.astype(np.float32),
                                metadata={'axes': 'TYX'})
                print(f"  Saved: {stack_path} (shape: {stack_array.shape})")
        
        # 統計サマリー
        summary_path = os.path.join(output_dir, 'volume_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("=== Volume Time-series Summary ===\n\n")
            f.write(f"Total cells tracked: {len(self.results_df)}\n")
            f.write(f"Time points: {len(self.results_df['time_index'].unique())}\n")
            f.write(f"Cells per frame: {self.results_df.groupby('time_index').size().mean():.1f} ± {self.results_df.groupby('time_index').size().std():.1f}\n\n")
            
            f.write("Volume Statistics (um^3):\n")
            f.write(f"  Mean: {self.results_df['volume_um3'].mean():.2f}\n")
            f.write(f"  Median: {self.results_df['volume_um3'].median():.2f}\n")
            f.write(f"  Std: {self.results_df['volume_um3'].std():.2f}\n")
            f.write(f"  Min: {self.results_df['volume_um3'].min():.2f}\n")
            f.write(f"  Max: {self.results_df['volume_um3'].max():.2f}\n\n")
            
            f.write(self.results_df.describe().to_string())
        
        print(f"  Saved: {summary_path}")
    
    def compute_ri_from_phase_images(self, phase_image_dir, wavelength_nm=663, n_medium=1.333):
        """
        位相差画像と厚みマップからRI (Refractive Index) を計算
        
        Parameters
        ----------
        phase_image_dir : str
            位相差画像が入ったディレクトリ
        wavelength_nm : float
            波長（ナノメートル）
        n_medium : float
            培地の屈折率
        
        Returns
        -------
        ri_results : list of dict
            各フレームのRI計算結果
        """
        if not hasattr(self, 'thickness_maps') or len(self.thickness_maps) == 0:
            print("Error: No thickness maps available. Run track_volume_timeseries() first.")
            return None
        
        print(f"\n=== Computing RI from Phase Images ===")
        print(f"Phase image directory: {phase_image_dir}")
        print(f"Wavelength: {wavelength_nm} nm")
        print(f"Medium RI: {n_medium}")
        
        wavelength_um = wavelength_nm / 1000.0
        
        # 位相差画像を検索
        import glob
        phase_files = sorted(glob.glob(os.path.join(phase_image_dir, "*.tif")))
        
        if len(phase_files) == 0:
            print(f"Error: No .tif files found in {phase_image_dir}")
            return None
        
        print(f"Found {len(phase_files)} phase images")
        
        ri_results = []
        
        for thick_info in self.thickness_maps:
            t_idx = thick_info['time_index']
            roi_name = thick_info['roi_name']
            thickness_map = thick_info['thickness_map']
            
            # 対応する位相差画像を読み込み
            if t_idx < len(phase_files):
                phase_img = tifffile.imread(phase_files[t_idx])
                
                # 位相差 → 屈折率
                # φ = (2π/λ) × (n_sample - n_medium) × thickness
                # n_sample = n_medium + (φ × λ) / (2π × thickness)
                
                # 厚みマップ（スライス数）を実際の厚み（um）に変換
                thickness_um = thickness_map * self.voxel_z
                
                # 位相差画像とマスクのサイズを確認
                if phase_img.shape != thickness_map.shape:
                    print(f"  Warning: Size mismatch for {roi_name}")
                    print(f"    Phase: {phase_img.shape}, Thickness: {thickness_map.shape}")
                    continue
                
                # ゼロ除算を避ける
                thickness_um_safe = np.where(thickness_um > 0, thickness_um, np.nan)
                
                # RI計算
                # phase_imgは既に位相差（ラジアン）と仮定
                n_sample = n_medium + (phase_img * wavelength_um) / (2 * np.pi * thickness_um_safe)
                
                # マスク内のみ
                mask = thickness_map > 0
                
                if np.sum(mask) > 0:
                    # 統計量
                    mean_ri = np.nanmean(n_sample[mask])
                    median_ri = np.nanmedian(n_sample[mask])
                    std_ri = np.nanstd(n_sample[mask])
                    
                    # Total RI (積分)
                    total_ri = np.nansum(n_sample[mask] - n_medium)
                    
                    ri_result = {
                        'time_index': t_idx,
                        'roi_name': roi_name,
                        'mean_ri': mean_ri,
                        'median_ri': median_ri,
                        'std_ri': std_ri,
                        'total_ri': total_ri,
                        'n_pixels': np.sum(mask),
                        'ri_map': n_sample  # RI分布マップ
                    }
                    
                    ri_results.append(ri_result)
                    
                    print(f"  [{t_idx}] {roi_name}: Mean RI = {mean_ri:.4f}, Total RI = {total_ri:.2f}")
        
        self.ri_results = ri_results
        
        print(f"\n  Total RI calculations: {len(ri_results)}")
        
        return ri_results
    
    def save_ri_results(self, output_dir='Pomegranate_volume_output'):
        """RI計算結果を保存"""
        if not hasattr(self, 'ri_results') or len(self.ri_results) == 0:
            print("No RI results to save")
            return
        
        ri_dir = os.path.join(output_dir, 'ri_maps')
        os.makedirs(ri_dir, exist_ok=True)
        
        print(f"\n=== Saving RI Results to {ri_dir} ===")
        
        # RI統計をCSVに保存
        ri_stats = []
        for ri_res in self.ri_results:
            stats = {k: v for k, v in ri_res.items() if k != 'ri_map'}
            ri_stats.append(stats)
        
        ri_df = pd.DataFrame(ri_stats)
        csv_path = os.path.join(output_dir, 'ri_statistics.csv')
        ri_df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")
        
        # RIマップを個別に保存
        print(f"  Saving {len(self.ri_results)} RI maps...")
        for idx, ri_res in enumerate(self.ri_results):
            if idx % 100 == 0:
                print(f"    Progress: {idx}/{len(self.ri_results)}")
            
            roi_name = ri_res['roi_name'].replace('.roi', '')
            ri_path = os.path.join(ri_dir, f"{roi_name}_ri_map.tif")
            
            tifffile.imwrite(ri_path, ri_res['ri_map'].astype(np.float32))
        
        print(f"  Saved: {len(self.ri_results)} RI maps")
        
        # サマリー
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


def demo_with_roiset(roi_zip_path, max_frames=5):
    """ROIセットでデモ実行（Pomegranate本家準拠）"""
    print("\n" + "="*60)
    print("DEMO: Time-series Volume Tracking from ROI Set (Pomegranate)")
    print("="*60)
    
    # Trackerを作成（本家Pomegranate準拠のパラメータ）
    tracker = TimeSeriesVolumeTracker(
        roi_zip_path=roi_zip_path,
        voxel_xy=0.348,  # 24_elip_volume.pyと同じ
        voxel_z=0.174,
        radius_enlarge=1.0,  # 本家Pomegranateは+1
        image_width=512,
        image_height=512,
        min_radius_threshold_px=2,  # 本家Pomegranateは2ピクセル
        discretization_method='pomegranate'  # 閾値ベース判定
    )
    
    # 体積を追跡
    results_df = tracker.track_volume_timeseries(max_frames=max_frames)
    
    # プロット
    tracker.plot_volume_timeseries('Pomegranate_volume_plot.png')
    
    # 保存
    tracker.save_results()
    
    return tracker


if __name__ == "__main__":
    # ROIセットのパス
    roi_zip_path = r"C:\Users\QPI\Desktop\align_demo\from_outputphase\bg_corr\subtracted\inference_out\Roiset_enlarge_interpolate.zip"
    
    if os.path.exists(roi_zip_path):
        print(f"Found ROI set: {roi_zip_path}")
        
        # デモ実行（最初の5フレーム）
        tracker = demo_with_roiset(roi_zip_path, max_frames=2000)
        
        print("\n" + "="*60)
        print("DEMO COMPLETE!")
        print("="*60)
        print("\nOutput files:")
        print("  - timeseries_volume_output/volume_timeseries.csv")
        print("  - timeseries_volume_output/volume_summary.txt")
        print("  - timeseries_volume_plot.png")
    else:
        print(f"Error: ROI set not found at {roi_zip_path}")
        print("Please provide the correct path.")


# %%
