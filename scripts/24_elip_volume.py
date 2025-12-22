#!/usr/bin/env python3
"""
時系列画像対応の密度マップワークフロー

画像枚数とResults.csvの行数が異なる場合でも、
各ROIを対応する画像に自動マッチングして処理。
"""

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
    """時系列画像とResults.csvから密度マップを生成"""
    
    def __init__(self, results_csv, image_directory, correction_factor=0.02):
        """
        Parameters:
        -----------
        results_csv : str
            ImageJ Results.csvのパス
        image_directory : str
            時系列画像が入ったディレクトリ
        correction_factor : float
            培地補正係数（デフォルト: 0.02）
        """
        self.results_csv = results_csv
        self.image_directory = image_directory
        self.correction_factor = correction_factor
        
        # Results.csvを読み込み
        print(f"Loading Results.csv: {results_csv}")
        self.df = pd.read_csv(results_csv)
        print(f"  Found {len(self.df)} ROIs")
        
        # フレーム番号を抽出
        self._extract_frame_numbers()
        
        # 画像ファイルを検索
        self._scan_image_files()
        
        # 出力ディレクトリ
        self.output_dir = "timeseries_density_output"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "density_tiff"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "visualizations"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "csv_data"), exist_ok=True)
        
    def _extract_frame_numbers(self):
        """Results.csvからフレーム番号を抽出"""
        print("\nExtracting frame numbers from Results.csv...")
        
        # Sliceカラムを使用（最も信頼性が高い）
        if 'Slice' in self.df.columns:
            self.df['frame_number'] = self.df['Slice']
            print(f"  Using 'Slice' column")
        else:
            # Labelから抽出
            print(f"  Extracting from 'Label' column...")
            def extract_frame(label):
                match = re.search(r'output_phase(\d+)', label)
                if match:
                    return int(match.group(1))
                return None
            
            self.df['frame_number'] = self.df['Label'].apply(extract_frame)
        
        # フレーム番号の統計
        unique_frames = self.df['frame_number'].dropna().unique()
        print(f"  Frame range: {unique_frames.min():.0f} to {unique_frames.max():.0f}")
        print(f"  Number of unique frames: {len(unique_frames)}")
        
    def _scan_image_files(self):
        """画像ファイルをスキャンしてフレーム番号とパスをマッピング"""
        print(f"\nScanning image files in: {self.image_directory}")
        
        # 画像ファイルを検索
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
        
        # フレーム番号を抽出してマッピング
        self.frame_to_path = {}
        
        for img_path in image_files:
            filename = os.path.basename(img_path)
            
            # "output_phase0001" のような部分を抽出
            match = re.search(r'output_phase(\d+)', filename)
            if match:
                frame_num = int(match.group(1))
                self.frame_to_path[frame_num] = img_path
        
        print(f"  Mapped {len(self.frame_to_path)} frames to image files")
        
        if len(self.frame_to_path) == 0:
            # フレーム番号が見つからない場合、連番として扱う
            print("  WARNING: Could not extract frame numbers from filenames")
            print("  Using sequential numbering instead...")
            sorted_files = sorted(image_files)
            for i, img_path in enumerate(sorted_files, start=1):
                self.frame_to_path[i] = img_path
        
        # フレーム番号の範囲を表示
        frame_numbers = sorted(self.frame_to_path.keys())
        print(f"  Image frame range: {frame_numbers[0]} to {frame_numbers[-1]}")
        
    def load_image(self, frame_number):
        """指定されたフレーム番号の画像を読み込み"""
        if frame_number not in self.frame_to_path:
            raise ValueError(f"Frame {frame_number} not found in image files")
        
        img_path = self.frame_to_path[frame_number]
        image = np.array(Image.open(img_path)).astype(np.float64)
        
        return image, img_path
    
    def create_rod_zstack_map(self, roi_params, image_shape):
        """
        ROI用のz-stackカウントマップを生成
        
        Parameters:
        -----------
        roi_params : dict
            ROIのパラメータ
        image_shape : tuple
            画像のサイズ (height, width)
        
        Returns:
        --------
        zstack_map : 2D numpy array
            z-stackカウントマップ
        """
        # パラメータ取得
        center_x = roi_params['X']
        center_y = roi_params['Y']
        length = roi_params['Major']
        width = roi_params['Minor']
        angle = roi_params['Angle']
        
        # ロッド形状パラメータ
        r = width / 2.0
        h = length - 2 * r
        
        if h < 0:
            h = 0
        
        # ImageJ座標系対応
        angle_rad = np.deg2rad(-angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # z-stackカウントマップ
        img_height, img_width = image_shape
        zstack_map = np.zeros((img_height, img_width), dtype=np.float64)
        
        # 各ピクセルについてz-stack数を計算
        for py in range(img_height):
            for px in range(img_width):
                dx = px - center_x
                dy = py - center_y
                
                # ローカル座標系に変換
                x_local = dx * cos_a + dy * sin_a
                y_local = -dx * sin_a + dy * cos_a
                
                dist_from_axis = abs(y_local)
                
                if dist_from_axis > r:
                    continue
                
                # z方向の厚み
                z_half = np.sqrt(r**2 - y_local**2)
                thickness = 2 * z_half
                
                # 長軸方向の位置
                if abs(x_local) <= h / 2.0:
                    # 円柱部分
                    zstack_map[py, px] = thickness
                else:
                    # 半球部分
                    if x_local > 0:
                        x_from_sphere_center = x_local - h / 2.0
                    else:
                        x_from_sphere_center = x_local + h / 2.0
                    
                    dist_sq = x_from_sphere_center**2 + y_local**2
                    
                    if dist_sq <= r**2:
                        z_half_sphere = np.sqrt(r**2 - dist_sq)
                        thickness_sphere = 2 * z_half_sphere
                        zstack_map[py, px] = thickness_sphere
        
        return zstack_map
    
    def process_roi(self, roi_index):
        """
        特定のROIを処理
        
        Parameters:
        -----------
        roi_index : int
            Results.csv内のROIインデックス
        
        Returns:
        --------
        results : dict
            処理結果
        """
        row = self.df.iloc[roi_index]
        frame_number = int(row['frame_number'])
        
        print(f"\n{'='*70}")
        print(f"Processing ROI {roi_index} (Frame {frame_number})")
        print(f"{'='*70}")
        
        # 画像を読み込み
        try:
            image, img_path = self.load_image(frame_number)
            print(f"Loaded image: {os.path.basename(img_path)}")
            print(f"  Image shape: {image.shape}")
        except ValueError as e:
            print(f"  ERROR: {e}")
            return None
        
        # ROIパラメータ
        roi_params = {
            'X': row['X'],
            'Y': row['Y'],
            'Major': row['Major'],
            'Minor': row['Minor'],
            'Angle': row['Angle'],
        }
        
        print(f"  ROI parameters:")
        print(f"    Center: ({roi_params['X']:.2f}, {roi_params['Y']:.2f})")
        print(f"    Length: {roi_params['Major']:.2f} pixels")
        print(f"    Width: {roi_params['Minor']:.2f} pixels")
        print(f"    Angle: {roi_params['Angle']:.2f}°")
        
        # z-stackカウントマップを生成
        print("  Generating z-stack map...")
        zstack_map = self.create_rod_zstack_map(roi_params, image.shape)
        
        mask = zstack_map > 0
        if not np.any(mask):
            print("  WARNING: No pixels in ROI!")
            return None
        
        print(f"    Max z-stack: {zstack_map.max():.2f}")
        print(f"    Mean z-stack: {zstack_map[mask].mean():.2f}")
        print(f"    Non-zero pixels: {np.count_nonzero(mask)}")
        
        # 密度マップを計算
        print(f"  Calculating density map (correction={self.correction_factor})...")
        density_map = np.zeros_like(image, dtype=np.float64)
        
        correction_term = zstack_map[mask] * self.correction_factor
        density_map[mask] = (image[mask] + correction_term) / zstack_map[mask]
        
        print(f"    Density range: [{density_map[mask].min():.4f}, {density_map[mask].max():.4f}]")
        print(f"    Mean density: {density_map[mask].mean():.4f}")
        
        # 結果をパッケージング
        results = {
            'roi_index': roi_index,
            'frame_number': frame_number,
            'image_path': img_path,
            'image': image,
            'zstack_map': zstack_map,
            'density_map': density_map,
            'roi_params': roi_params,
            'stats': {
                'zstack_max': float(zstack_map.max()),
                'zstack_mean': float(zstack_map[mask].mean()),
                'density_mean': float(density_map[mask].mean()),
                'density_min': float(density_map[mask].min()),
                'density_max': float(density_map[mask].max()),
                'num_pixels': int(np.count_nonzero(mask)),
            }
        }
        
        return results
    
    def save_results(self, results):
        """結果を保存"""
        if results is None:
            return
        
        roi_index = results['roi_index']
        frame_number = results['frame_number']
        roi_str = f"ROI_{roi_index:04d}_Frame_{frame_number:04d}"
        
        # TIFF保存
        density_tiff = os.path.join(self.output_dir, "density_tiff", f"{roi_str}_density.tif")
        zstack_tiff = os.path.join(self.output_dir, "density_tiff", f"{roi_str}_zstack.tif")
        
        tifffile.imwrite(density_tiff, results['density_map'].astype(np.float32))
        tifffile.imwrite(zstack_tiff, results['zstack_map'].astype(np.float32))
        
        print(f"\nSaved: {os.path.basename(density_tiff)}")
        print(f"Saved: {os.path.basename(zstack_tiff)}")
        
        # CSV保存
        csv_path = os.path.join(self.output_dir, "csv_data", f"{roi_str}_pixel_data.csv")
        
        mask = results['zstack_map'] > 0
        y_coords, x_coords = np.where(mask)
        
        pixel_data = pd.DataFrame({
            'X_pixel': x_coords,
            'Y_pixel': y_coords,
            'Z_stack_count': results['zstack_map'][mask],
            'Original_value': results['image'][mask],
            'Correction': results['zstack_map'][mask] * self.correction_factor,
            'Density': results['density_map'][mask],
        })
        pixel_data.to_csv(csv_path, index=False)
        print(f"Saved: {os.path.basename(csv_path)}")
        
        # パラメータCSV
        params_path = os.path.join(self.output_dir, "csv_data", f"{roi_str}_parameters.csv")
        params_dict = {
            'roi_index': roi_index,
            'frame_number': frame_number,
            'image_path': results['image_path'],
            **results['roi_params'],
            **results['stats'],
            'correction_factor': self.correction_factor,
        }
        params_df = pd.DataFrame([params_dict])
        params_df.to_csv(params_path, index=False)
        print(f"Saved: {os.path.basename(params_path)}")
        
        # 可視化
        self.create_visualization(results, roi_str)
    
    def create_visualization(self, results, roi_str):
        """可視化図を作成"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        image = results['image']
        zstack_map = results['zstack_map']
        density_map = results['density_map']
        roi_params = results['roi_params']
        stats = results['stats']
        
        mask = zstack_map > 0
        
        # 1. 元画像 + ROI
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(image, cmap='gray', interpolation='nearest')
        ax1.set_title(f'Original Image\nFrame {results["frame_number"]}', 
                      fontsize=12, fontweight='bold')
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        plt.colorbar(im1, ax=ax1, label='Intensity')
        
        # ROI中心とアウトライン
        ax1.plot(roi_params['X'], roi_params['Y'], 'r+', markersize=15, markeredgewidth=2)
        ellipse = Ellipse(
            xy=(roi_params['X'], roi_params['Y']),
            width=roi_params['Major'],
            height=roi_params['Minor'],
            angle=-roi_params['Angle'],
            edgecolor='red',
            facecolor='none',
            linewidth=2
        )
        ax1.add_patch(ellipse)
        
        # 2. Z-stackマップ
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(zstack_map, cmap='viridis', interpolation='nearest')
        ax2.set_title(f'Z-stack Count Map\n(max={stats["zstack_max"]:.2f})', 
                      fontsize=12, fontweight='bold')
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')
        plt.colorbar(im2, ax=ax2, label='Z-stack count')
        
        # 3. 培地補正
        ax3 = fig.add_subplot(gs[0, 2])
        correction_map = zstack_map * self.correction_factor
        im3 = ax3.imshow(correction_map, cmap='hot', interpolation='nearest')
        ax3.set_title(f'Medium Correction\n(Z × {self.correction_factor})', 
                      fontsize=12, fontweight='bold')
        ax3.set_xlabel('X (pixels)')
        ax3.set_ylabel('Y (pixels)')
        plt.colorbar(im3, ax=ax3, label='Correction value')
        
        # 4. 密度マップ
        ax4 = fig.add_subplot(gs[1, 0])
        im4 = ax4.imshow(density_map, cmap='plasma', interpolation='nearest')
        ax4.set_title(f'Density Map\n(mean={stats["density_mean"]:.4f})', 
                      fontsize=12, fontweight='bold')
        ax4.set_xlabel('X (pixels)')
        ax4.set_ylabel('Y (pixels)')
        plt.colorbar(im4, ax=ax4, label='Density')
        
        # 5. Density vs Z-stack
        ax5 = fig.add_subplot(gs[1, 1])
        if np.any(mask):
            ax5.scatter(zstack_map[mask], density_map[mask], alpha=0.3, s=10, c='blue')
            
            theoretical = 1.0 + self.correction_factor
            ax5.axhline(y=theoretical, color='red', linestyle='--', linewidth=2,
                       label=f'Theoretical: {theoretical:.4f}')
            
            ax5.set_xlabel('Z-stack count', fontsize=11)
            ax5.set_ylabel('Density', fontsize=11)
            ax5.set_title('Density vs Z-stack Count', fontsize=12, fontweight='bold')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. 分布
        ax6 = fig.add_subplot(gs[1, 2])
        if np.any(mask):
            original_per_z = image[mask] / zstack_map[mask]
            ax6.hist(original_per_z, bins=50, alpha=0.5, label='Original/Z', color='blue')
            ax6.hist(density_map[mask], bins=50, alpha=0.5, label='Density', color='orange')
            
            ax6.set_xlabel('Value', fontsize=11)
            ax6.set_ylabel('Frequency', fontsize=11)
            ax6.set_title('Value Distribution', fontsize=12, fontweight='bold')
            ax6.legend()
            ax6.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle(f'{roi_str} - Density Analysis (Correction={self.correction_factor})', 
                     fontsize=16, fontweight='bold')
        
        viz_path = os.path.join(self.output_dir, "visualizations", f"{roi_str}_visualization.png")
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {os.path.basename(viz_path)}")
    
    def process_all_rois(self, max_rois=None):
        """
        全ROIを処理
        
        Parameters:
        -----------
        max_rois : int or None
            処理する最大ROI数（Noneなら全て）
        """
        num_rois = len(self.df) if max_rois is None else min(max_rois, len(self.df))
        
        print(f"\n{'#'*80}")
        print(f"# Starting time-series density workflow")
        print(f"# Total ROIs: {num_rois}")
        print(f"# Correction factor: {self.correction_factor}")
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
        
        # サマリー保存
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


# ===== メイン実行 =====
if __name__ == "__main__":
    # パラメータ設定
    RESULTS_CSV = "/mnt/user-data/uploads/Results.csv"
    
    # 画像ディレクトリ（ユーザーの環境に合わせて変更）
    # Windowsパスの例: r"C:\Users\QPI\Desktop\align_demo\from_outputphase\bg_corr\subtracted"
    # Linuxパスの例: "/home/user/images/subtracted"
    IMAGE_DIRECTORY = "/mnt/user-data/uploads"  # テスト用
    
    CORRECTION_FACTOR = 0.02
    MAX_ROIS = 10  # テスト実行（Noneで全ROI）
    
    # ワークフロー実行
    mapper = TimeSeriesDensityMapper(
        results_csv=RESULTS_CSV,
        image_directory=IMAGE_DIRECTORY,
        correction_factor=CORRECTION_FACTOR
    )
    
    mapper.process_all_rois(max_rois=MAX_ROIS)