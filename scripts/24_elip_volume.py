#!/usr/bin/env python3
"""
時系列画像対応の密度マップワークフロー

画像枚数とResults.csvの行数が異なる場合でも、
各ROIを対応する画像に自動マッチングして処理。
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
    """時系列画像とResults.csvから屈折率（RI）マップを生成"""
    
    def __init__(self, results_csv, image_directory, 
                 wavelength_nm=663, n_medium=1.333, pixel_size_um=0.348,
                 alpha_ri=0.0018, shape_type='ellipse', subpixel_sampling=5,
                 thickness_mode='continuous', voxel_z_um=0.3, discretize_method='round',
                 min_thickness_px=0.0, csv_suffix=None):
        """
        Parameters:
        -----------
        results_csv : str
            ImageJ Results.csvのパス
        image_directory : str
            時系列画像が入ったディレクトリ
        wavelength_nm : float
            波長（ナノメートル）。デフォルト: 663nm
        n_medium : float
            培地の屈折率。デフォルト: 1.333
        pixel_size_um : float
            ピクセルサイズ（マイクロメートル）。デフォルト: 0.348 µm
            507×507の再構成画像用: 0.08625 × (2048/507) ≈ 0.348 µm/pixel
        alpha_ri : float
            比屈折率増分（specific refractive index increment）[ml/mg]
            デフォルト: 0.0018 ml/mg（タンパク質の一般的な値）
        shape_type : str
            ROI形状の近似方法。デフォルト: 'ellipse'
            'ellipse': Major/Minor/Angleを使用（楕円近似）
            'feret': Feret/MinFeret/FeretAngleを使用（Feret径近似）
        subpixel_sampling : int
            ピクセル内サブサンプリング数（N×N）。デフォルト: 5
            1: ピクセル中心のみ（高速だが端で精度低）
            5: 5×5サブピクセル（推奨、バランス良い）
            10: 10×10サブピクセル（高精度だが遅い）
        thickness_mode : str
            厚みマップのモード。デフォルト: 'continuous'
            'continuous': 連続的な厚み値（ピクセル単位）
            'discrete': 離散的なZ-stackスライス数
        voxel_z_um : float
            Z方向のボクセルサイズ（マイクロメートル）。デフォルト: 0.3 µm
            discrete modeで厚みをスライス数に変換する際に使用
        discretize_method : str
            離散化の方法（discrete modeのみ有効）。デフォルト: 'round'
            'round': 四捨五入
            'ceil': 切り上げ
            'floor': 切り捨て
            'pomegranate': Pomegranate互換の閾値ベース判定
        min_thickness_px : float
            最小厚み閾値（ピクセル単位）。デフォルト: 0.0
            この値未満の厚みを持つピクセルは無視される（0にセット）
            例: 1.0 → 1ピクセル未満の厚みを無視
        csv_suffix : str, optional
            出力フォルダ名に追加するサフィックス。デフォルト: None
            Noneの場合、CSVファイル名から自動抽出（例: Results_enlarge.csv → enlarge）
            手動で指定する場合: 'enlarge', 'interpolate', 'custom_name'など
        
        Note:
        -----
        位相差から屈折率への変換:
        φ = (2π/λ) × (n_sample - n_medium) × thickness
        n_sample = n_medium + (φ × λ) / (2π × thickness)
        
        屈折率から質量濃度への変換:
        C [mg/ml] = (RI - RI_medium) / α
        
        サブピクセルサンプリング:
        マスク端の精度を向上させるため、各ピクセルをN×Nに分割し、
        サブピクセル中心での厚みを平均します。
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
        
        # CSVサフィックスを決定（手動指定 or 自動抽出）
        if csv_suffix is not None:
            self.csv_suffix = csv_suffix
        else:
            # CSVファイル名から自動抽出 (例: Results_enlarge.csv → enlarge)
            csv_filename = os.path.basename(results_csv)
            csv_name_without_ext = os.path.splitext(csv_filename)[0]  # Results_enlarge
            # "Results_"の後の部分を取得（あれば）
            if '_' in csv_name_without_ext:
                parts = csv_name_without_ext.split('_', 1)  # ['Results', 'enlarge']
                if len(parts) > 1 and parts[1]:
                    self.csv_suffix = parts[1]
                else:
                    self.csv_suffix = None
            else:
                self.csv_suffix = None
        
        # 単位変換
        self.wavelength_um = wavelength_nm / 1000.0  # nm → µm
        
        # Results.csvを読み込み
        print(f"Loading Results.csv: {results_csv}")
        self.df = pd.read_csv(results_csv)
        print(f"  Found {len(self.df)} ROIs")
        
        # フレーム番号を抽出
        self._extract_frame_numbers()
        
        # 画像ファイルを検索
        self._scan_image_files()
        
        # 出力ディレクトリ（パラメータに応じた名前）
        if self.csv_suffix:
            base_dir_suffix = f"{self.shape_type}_subpixel{self.subpixel_sampling}_{self.csv_suffix}"
        else:
            base_dir_suffix = f"{self.shape_type}_subpixel{self.subpixel_sampling}"
        
        # thickness_modeがdiscreteの場合は追加情報を含める
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
    
    def create_rod_zstack_map(self, roi_params, image_shape, shape_type='ellipse', subpixel_sampling=5):
        """
        ROI用のz-stackカウントマップを生成（サブピクセルサンプリング対応）
        
        Parameters:
        -----------
        roi_params : dict
            ROIのパラメータ
        image_shape : tuple
            画像のサイズ (height, width)
        shape_type : str
            'ellipse': Major/Minor/Angleを使用（楕円近似）
            'feret': Feret/MinFeret/FeretAngleを使用（Feret径近似）
        subpixel_sampling : int
            ピクセル内のサブサンプリング数（N×Nグリッド）
            1: ピクセル中心のみ（高速だが精度低）
            5: 5×5サブピクセル（推奨、バランス良い）
            10: 10×10サブピクセル（高精度だが遅い）
        
        Returns:
        --------
        zstack_map : 2D numpy array
            z-stackカウントマップ（ピクセル内平均厚み）
        
        Note:
        -----
        サブピクセルサンプリングにより、マスク端での精度が向上します。
        各ピクセルを N×N のサブピクセルに分割し、各サブピクセル中心で
        厚みを計算して平均を取ります。これにより、部分的にマスク領域に
        かかるピクセルでも正確な平均厚みが得られます。
        """
        # パラメータ取得（shape_typeに応じて切り替え）
        center_x = roi_params['X']
        center_y = roi_params['Y']
        
        if shape_type == 'feret':
            # Feret径を使用
            length = roi_params.get('Feret', roi_params.get('Major'))
            width = roi_params.get('MinFeret', roi_params.get('Minor'))
            angle = roi_params.get('FeretAngle', roi_params.get('Angle'))
        else:  # 'ellipse' (デフォルト)
            # 楕円パラメータを使用
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
        
        # サブピクセルオフセットを計算
        if subpixel_sampling > 1:
            # N×Nサブピクセルの中心座標オフセット
            offsets = np.linspace(0.5/subpixel_sampling, 1 - 0.5/subpixel_sampling, subpixel_sampling) - 0.5
        else:
            # サブピクセルなし（ピクセル中心のみ）
            offsets = np.array([0.0])
        
        # 各ピクセルについてz-stack数を計算（サブピクセルサンプリング）
        for py in range(img_height):
            for px in range(img_width):
                thickness_sum = 0.0
                valid_subpixels = 0
                
                # サブピクセルごとに厚みを計算
                for dy_offset in offsets:
                    for dx_offset in offsets:
                        # サブピクセル中心座標
                        px_sub = px + 0.5 + dx_offset
                        py_sub = py + 0.5 + dy_offset
                        
                        dx = px_sub - center_x
                        dy = py_sub - center_y
                        
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
                            thickness_sum += thickness
                            valid_subpixels += 1
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
                                thickness_sum += thickness_sphere
                                valid_subpixels += 1
                
                # ピクセル内の平均厚みを計算
                if valid_subpixels > 0:
                    zstack_map[py, px] = thickness_sum / valid_subpixels
        
        return zstack_map
    
    def _discretize_thickness(self, z_continuous_px):
        """
        連続的なZ方向の厚み（ピクセル単位）を離散的なスライス数に変換する。
        
        Parameters
        ----------
        z_continuous_px : float or np.ndarray
            連続的なZ方向の厚み（ピクセル単位）
        
        Returns
        -------
        int or np.ndarray
            離散化されたスライス数
        """
        if self.voxel_z_um <= 0:
            # voxel_z_umが無効な場合は、元の値を返す
            if isinstance(z_continuous_px, np.ndarray):
                return z_continuous_px.astype(int)
            return int(z_continuous_px)
        
        # ピクセル単位の厚みをµm単位に変換
        z_um = z_continuous_px * self.pixel_size_um
        
        if self.discretize_method == 'round':
            z_slices = np.round(z_um / self.voxel_z_um)
        elif self.discretize_method == 'ceil':
            z_slices = np.ceil(z_um / self.voxel_z_um)
        elif self.discretize_method == 'floor':
            z_slices = np.floor(z_um / self.voxel_z_um)
        elif self.discretize_method == 'pomegranate':
            # Pomegranate方式の離散化
            # 最小半径閾値（ここではピクセルサイズの2倍とする）
            min_radius_threshold_um = 2.0 * self.pixel_size_um
            
            # 配列の場合と単一値の場合で処理を分岐
            if isinstance(z_um, np.ndarray):
                num_z_voxels_float = z_um / self.voxel_z_um
                z_slices = np.zeros_like(num_z_voxels_float)
                
                # 閾値を超える要素のみ処理
                mask = z_um > min_radius_threshold_um
                z_slices[mask] = np.round(num_z_voxels_float[mask])
                
                # 少なくとも1スライスは確保（閾値を超える場合）
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
            # デフォルトはround
            z_slices = np.round(z_um / self.voxel_z_um)
        
        # 負の値を0にクリップ
        z_slices = np.maximum(0, z_slices)
        
        # 整数に変換
        if isinstance(z_slices, np.ndarray):
            return z_slices.astype(int)
        return int(z_slices)
    
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
        
        # Feret径パラメータを追加（存在する場合）
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
        
        # ROI識別子を生成（キャッシュファイル名用）
        roi_id = f"frame{frame_number:04d}"
        
        # z-stackカウントマップを生成または読み込み
        # キャッシュファイル名を生成
        cache_dir = os.path.join(self.output_dir, 'thickness_cache')
        cache_path = os.path.join(cache_dir, f"roi_{roi_index:04d}_{roi_id}.npz")
        
        # discreteモードで、キャッシュが存在する場合は読み込む
        if self.thickness_mode == 'discrete' and os.path.exists(cache_path):
            print(f"  Loading cached thickness map from: {os.path.basename(cache_path)}")
            cached_data = np.load(cache_path)
            zstack_map_continuous = cached_data['thickness_map_continuous']
            print(f"    Loaded shape: {zstack_map_continuous.shape}")
        else:
            # 新規に計算
            print(f"  Generating z-stack map (shape_type: {self.shape_type}, subpixel: {self.subpixel_sampling}×{self.subpixel_sampling})...")
            print(f"    Thickness mode: {self.thickness_mode}")
            if self.thickness_mode == 'discrete':
                print(f"    Discretize method: {self.discretize_method}")
                print(f"    Voxel Z size: {self.voxel_z_um} µm")
            
            zstack_map_continuous = self.create_rod_zstack_map(roi_params, image.shape, 
                                                                 shape_type=self.shape_type,
                                                                 subpixel_sampling=self.subpixel_sampling)
            
            # continuousモードの場合はキャッシュに保存
            if self.thickness_mode == 'continuous':
                os.makedirs(cache_dir, exist_ok=True)
                np.savez_compressed(cache_path, 
                                    thickness_map_continuous=zstack_map_continuous,
                                    roi_id=roi_id,
                                    roi_index=roi_index)
                print(f"    Saved thickness cache: {os.path.basename(cache_path)}")
        
        # thickness_modeに応じてzstack_mapを決定
        if self.thickness_mode == 'discrete':
            # 離散化されたスライス数
            zstack_map = self._discretize_thickness(zstack_map_continuous)
        else:
            # 連続値（ピクセル単位の厚み）
            zstack_map = zstack_map_continuous
        
        # 最小厚み閾値フィルタリング（ピクセル単位で判定）
        if self.min_thickness_px > 0:
            # continuousモードでは直接比較、discreteモードでは換算して比較
            if self.thickness_mode == 'discrete':
                # スライス数をピクセル単位に換算して閾値判定
                thickness_px_for_threshold = zstack_map * (self.voxel_z_um / self.pixel_size_um)
            else:
                thickness_px_for_threshold = zstack_map
            
            # 閾値未満を0にする
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
        
        # 位相差から屈折率（RI）を計算
        print(f"  Converting phase to refractive index...")
        print(f"    Wavelength: {self.wavelength_nm} nm")
        print(f"    Medium RI: {self.n_medium}")
        print(f"    Pixel size: {self.pixel_size_um} µm")
        
        # 厚みをµm単位に変換（thickness_modeに応じて）
        if self.thickness_mode == 'discrete':
            # 離散モード：スライス数 × Z方向のボクセルサイズ
            thickness_um = zstack_map * self.voxel_z_um
        else:
            # 連続モード：ピクセル単位の厚み × XY方向のピクセルサイズ
            thickness_um = zstack_map * self.pixel_size_um
        
        # RIマップを初期化（培地の屈折率で）
        ri_map = np.full_like(image, self.n_medium, dtype=np.float64)
        
        # 位相差→屈折率変換
        # n_sample = n_medium + (φ × λ) / (2π × thickness)
        # ここで、imageは位相差φに比例する値と仮定
        # 実際のシステムでは、imageの単位を確認して適切なスケーリングが必要
        ri_map[mask] = self.n_medium + (image[mask] * self.wavelength_um) / (2 * np.pi * thickness_um[mask])
        
        print(f"    RI range: [{ri_map[mask].min():.6f}, {ri_map[mask].max():.6f}]")
        print(f"    Mean RI: {ri_map[mask].mean():.6f}")
        print(f"    ΔRI from medium: {(ri_map[mask].mean() - self.n_medium):.6f}")
        
        # 質量濃度マップを計算（mg/ml）
        # C = (RI - RI_medium) / α
        concentration_map = np.zeros_like(image, dtype=np.float64)
        concentration_map[mask] = (ri_map[mask] - self.n_medium) / self.alpha_ri
        
        print(f"  Calculating mass concentration...")
        print(f"    Concentration range: [{concentration_map[mask].min():.2f}, {concentration_map[mask].max():.2f}] mg/ml")
        print(f"    Mean concentration: {concentration_map[mask].mean():.2f} mg/ml")
        
        # 体積を計算
        # 体積 = Σ(各ピクセルの厚み[µm] × ピクセル面積[µm²])
        pixel_area_um2 = self.pixel_size_um ** 2
        volume_um3 = np.sum(thickness_um[mask]) * pixel_area_um2
        
        print(f"  Calculating volume...")
        print(f"    Volume: {volume_um3:.2f} µm³")
        print(f"    Mean thickness: {thickness_um[mask].mean():.4f} µm")
        print(f"    Max thickness: {thickness_um[mask].max():.4f} µm")
        
        # Total massを計算
        # Total mass [pg] = Σ(concentration [mg/ml] × pixel_volume [µm³])
        # 単位変換: 1 mg/ml = 1 pg/µm³
        pixel_volumes = thickness_um[mask] * pixel_area_um2  # 各ピクセルの体積 [µm³]
        total_mass_pg = np.sum(concentration_map[mask] * pixel_volumes)  # [pg]
        
        print(f"  Calculating total mass...")
        print(f"    Total mass: {total_mass_pg:.2f} pg")
        print(f"    Mean concentration: {concentration_map[mask].mean():.2f} mg/ml")
        
        # 結果をパッケージング
        results = {
            'roi_index': roi_index,
            'frame_number': frame_number,
            'image_path': img_path,
            'image': image,
            'zstack_map': zstack_map,
            'ri_map': ri_map,  # 屈折率マップ
            'concentration_map': concentration_map,  # 質量濃度マップ（mg/ml）
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
                'ri_delta': float(ri_map[mask].mean() - self.n_medium),  # 培地との差
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
        ri_tiff = os.path.join(self.output_dir, "density_tiff", f"{roi_str}_ri.tif")
        concentration_tiff = os.path.join(self.output_dir, "density_tiff", f"{roi_str}_concentration.tif")
        zstack_tiff = os.path.join(self.output_dir, "density_tiff", f"{roi_str}_zstack.tif")
        phase_tiff = os.path.join(self.output_dir, "density_tiff", f"{roi_str}_phase.tif")
        
        tifffile.imwrite(ri_tiff, results['ri_map'].astype(np.float32))
        tifffile.imwrite(concentration_tiff, results['concentration_map'].astype(np.float32))
        tifffile.imwrite(zstack_tiff, results['zstack_map'].astype(np.float32))
        tifffile.imwrite(phase_tiff, results['image'].astype(np.float32))  # 元の位相差画像
        
        print(f"\nSaved: {os.path.basename(ri_tiff)}")
        print(f"Saved: {os.path.basename(concentration_tiff)}")
        print(f"Saved: {os.path.basename(zstack_tiff)}")
        print(f"Saved: {os.path.basename(phase_tiff)}")
        
        # CSV保存
        csv_path = os.path.join(self.output_dir, "csv_data", f"{roi_str}_pixel_data.csv")
        
        mask = results['zstack_map'] > 0
        y_coords, x_coords = np.where(mask)
        
        # 厚みをµm単位に変換（thickness_modeに応じて）
        if self.thickness_mode == 'discrete':
            # 離散モード：スライス数 × Z方向のボクセルサイズ
            thickness_um_map = results['zstack_map'][mask] * self.voxel_z_um
            z_column_name = 'Z_slice_count'
        else:
            # 連続モード：ピクセル単位の厚み × XY方向のピクセルサイズ
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
        
        # パラメータCSV
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
        
        # 可視化
        self.create_visualization(results, roi_str)
    
    def create_visualization(self, results, roi_str):
        """可視化図を作成"""
        fig = plt.figure(figsize=(26, 12))
        gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
        
        image = results['image']
        zstack_map = results['zstack_map']
        ri_map = results['ri_map']  # 屈折率マップ
        concentration_map = results['concentration_map']  # 質量濃度マップ
        roi_params = results['roi_params']
        stats = results['stats']
        
        mask = zstack_map > 0
        
        # 1. 元画像 + ROI + マスク輪郭線
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(image, cmap='gray',vmin=-0.5,vmax=2.5)
        ax1.set_title(f'Original Image + Mask Contour\nFrame {results["frame_number"]}', 
                      fontsize=12, fontweight='bold')
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        plt.colorbar(im1, ax=ax1, label='Intensity')
        
        # マスク輪郭線を描画（zstack > 0の領域）
        from skimage import measure
        binary_mask = (zstack_map > 0).astype(np.uint8)
        contours = measure.find_contours(binary_mask, 0.5)
        for contour in contours:
            ax1.plot(contour[:, 1], contour[:, 0], 'c-', linewidth=2, alpha=0.8)
        

        # 2. Z-stackマップ（厚みマップ）
        ax2 = fig.add_subplot(gs[0, 1])
        thickness_um = zstack_map * self.pixel_size_um
        im2 = ax2.imshow(thickness_um, cmap='viridis',vmin=0,vmax=12)
        ax2.set_title(f'Thickness Map\n(max={stats["zstack_max"]*self.pixel_size_um:.2f} µm)', 
                      fontsize=12, fontweight='bold')
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')
        plt.colorbar(im2, ax=ax2, label='Thickness (µm)')
        
        # 3. 屈折率マップ（RI map）
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(ri_map, cmap='jet',
                        vmin=1.3, vmax=1.39)
        ax3.set_title(f'Refractive Index Map\n(mean={stats["ri_mean"]:.6f})', 
                      fontsize=12, fontweight='bold')
        ax3.set_xlabel('X (pixels)')
        ax3.set_ylabel('Y (pixels)')
        cbar3 = plt.colorbar(im3, ax=ax3, label='RI')
        # 培地RIを示す線を追加
        cbar3.ax.axhline(y=self.n_medium, color='cyan', linestyle='--', linewidth=2, alpha=0.8)
        
        # 4. 質量濃度マップ（mg/ml）
        ax4 = fig.add_subplot(gs[0, 3])
        im4 = ax4.imshow(concentration_map, cmap='hot',
                        vmin=0, vmax=450)
        ax4.set_title(f'Protein Concentration Map\n(mean={stats["concentration_mean"]:.1f} mg/ml)', 
                      fontsize=12, fontweight='bold')
        ax4.set_xlabel('X (pixels)')
        ax4.set_ylabel('Y (pixels)')
        plt.colorbar(im4, ax=ax4, label='Concentration (mg/ml)')
        
        # 5. ΔRI マップ（培地との差）
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
            
            # 培地RIの参照線
            ax6.axhline(y=self.n_medium, color='cyan', linestyle='--', linewidth=2,
                       label=f'Medium RI: {self.n_medium:.3f}')
            # 平均RIの参照線
            ax6.axhline(y=stats['ri_mean'], color='red', linestyle='--', linewidth=2,
                       label=f'Mean RI: {stats["ri_mean"]:.6f}')
            
            ax6.set_xlabel('Thickness (µm)', fontsize=11)
            ax6.set_ylabel('Refractive Index', fontsize=11)
            ax6.set_title('RI vs Thickness', fontsize=12, fontweight='bold')
            ax6.set_xlim(0, 6)      # Thickness range: 0-6 µm
            ax6.set_ylim(1.3, 1.6)  # RI range: 1.3-1.6
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # 7. RI分布
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
            ax7.set_xlim(1.30, 1.60)  # X軸範囲を固定
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        
        # 8. 質量濃度分布
        ax8 = fig.add_subplot(gs[1, 3])
        if np.any(mask):
            concentration_masked = concentration_map[mask].flatten()
            ax8.hist(concentration_masked, bins=50, alpha=0.7, color='orange', edgecolor='black', range=(0, 450))
            ax8.axvline(x=stats['concentration_mean'], color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {stats["concentration_mean"]:.1f} mg/ml')
            ax8.set_xlabel('Concentration (mg/ml)', fontsize=11)
            ax8.set_ylabel('Frequency', fontsize=11)
            ax8.set_title('Protein Concentration Distribution', fontsize=12, fontweight='bold')
            ax8.set_xlim(0, 600)  # X軸範囲を固定
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
        全ROIを処理
        
        Parameters:
        -----------
        max_rois : int or None
            処理する最大ROI数（Noneなら全て）
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
    # バッチ実行時はglobals()から、単独実行時はデフォルト値を使用
    RESULTS_CSV = r"C:\Users\QPI\Desktop\align_demo\from_outputphase\bg_corr\subtracted\inference_out\Results_enlarge.csv" if 'RESULTS_CSV' not in globals() else globals()['RESULTS_CSV']
    
    # 画像ディレクトリ（ユーザーの環境に合わせて変更）
    # Windowsパスの例: r"C:\Users\QPI\Desktop\align_demo\from_outputphase\bg_corr\subtracted"
    # Linuxパスの例: "/home/user/images/subtracted"
    IMAGE_DIRECTORY = r"C:\Users\QPI\Desktop\align_demo\from_outputphase\bg_corr\subtracted" if 'IMAGE_DIRECTORY' not in globals() else globals()['IMAGE_DIRECTORY']
    
    # QPI実験パラメータ（01_QPI_analysis.pyと同じ値）
    WAVELENGTH_NM = 663 if 'WAVELENGTH_NM' not in globals() else globals()['WAVELENGTH_NM']
                                 # レーザー波長（ナノメートル）
                                 # 実験値: 663nm (赤色レーザー)
    N_MEDIUM = 1.333 if 'N_MEDIUM' not in globals() else globals()['N_MEDIUM']
                                 # 培地の屈折率
                                 # 水: 1.333, PBS: 1.334, DMEM: ~1.335
    PIXEL_SIZE_UM = 0.348 if 'PIXEL_SIZE_UM' not in globals() else globals()['PIXEL_SIZE_UM']
                                 # ピクセルサイズ（マイクロメートル）
                                 # 507×507の再構成画像用
                                 # 計算: 0.08625 µm × (2048/507) ≈ 0.348 µm/pixel
                                 # ※元のホログラム2048×2048では0.08625 µm/pixel
    ALPHA_RI = 0.00018 if 'ALPHA_RI' not in globals() else globals()['ALPHA_RI']
                                 # 比屈折率増分 [ml/mg]
                                 # タンパク質の一般的な値: 0.0018 ml/mg
                                 # 参考: RI = RI_medium + α × C [mg/ml]
    
    SHAPE_TYPE = 'feret' if 'SHAPE_TYPE' not in globals() else globals()['SHAPE_TYPE']
                                 # ROI形状近似方法
                                 # 'ellipse': Major/Minor/Angle（楕円近似）
                                 # 'feret': Feret/MinFeret/FeretAngle（Feret径近似）
    
    SUBPIXEL_SAMPLING = 5 if 'SUBPIXEL_SAMPLING' not in globals() else globals()['SUBPIXEL_SAMPLING']
                                 # サブピクセルサンプリング数（N×N）
                                 # 1: ピクセル中心のみ（高速、端で精度低）
                                 # 5: 5×5サブピクセル（推奨、バランス良い）
                                 # 10: 10×10サブピクセル（高精度、遅い）
    
    THICKNESS_MODE = 'continuous' if 'THICKNESS_MODE' not in globals() else globals()['THICKNESS_MODE']
                                 # 厚みマップのモード
                                 # 'continuous': 連続的な厚み値（ピクセル単位）
                                 # 'discrete': 離散的なZ-stackスライス数
    
    VOXEL_Z_UM = 0.3 if 'VOXEL_Z_UM' not in globals() else globals()['VOXEL_Z_UM']
                                 # Z方向のボクセルサイズ（マイクロメートル）
                                 # discrete modeで厚みをスライス数に変換する際に使用
    
    DISCRETIZE_METHOD = 'round' if 'DISCRETIZE_METHOD' not in globals() else globals()['DISCRETIZE_METHOD']
                                 # 離散化の方法（discrete modeのみ有効）
                                 # 'round': 四捨五入
                                 # 'ceil': 切り上げ
                                 # 'floor': 切り捨て
                                 # 'pomegranate': Pomegranate互換の閾値ベース判定
    
    MIN_THICKNESS_PX = 0.0 if 'MIN_THICKNESS_PX' not in globals() else globals()['MIN_THICKNESS_PX']
                                 # 最小厚み閾値（ピクセル単位）
                                 # 0.0: 閾値なし（すべて含む）
                                 # 1.0: 1ピクセル未満を無視
                                 # 2.0: 2ピクセル未満を無視
    
    MAX_ROIS = None if 'MAX_ROIS' not in globals() else globals()['MAX_ROIS']  # テスト実行（Noneで全ROI）
    
    # CSVサフィックス（出力フォルダ名の識別用）
    # None: CSVファイル名から自動抽出（Results_enlarge.csv → 'enlarge'）
    # 文字列: 手動で指定（例: 'custom_name'）
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
    
    # ワークフロー実行
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
    
    # ===== 時系列プロット生成 =====
    print(f"\n{'#'*80}")
    print(f"# Generating time-series plots...")
    print(f"{'#'*80}\n")
    
    # 27_timeseries_plot.pyの機能を統合
    from matplotlib.gridspec import GridSpec
    
    # all_rois_summary.csvを読み込み
    summary_path = os.path.join(mapper.output_dir, "all_rois_summary.csv")
    
    if os.path.exists(summary_path):
        print(f"Loading summary data: {summary_path}")
        df_summary = pd.read_csv(summary_path)
        
        # 時間を計算（frame_number / 12 = 時間[h]）
        # または既存のsliceがあればそれを使用
        if 'Slice' in mapper.df.columns:
            df_summary['time_h'] = df_summary['frame_number'] / 12.0
        else:
            df_summary['time_h'] = df_summary['frame_number'] / 12.0
        
        print(f"  Time range: {df_summary['time_h'].min():.2f} - {df_summary['time_h'].max():.2f} h")
        print(f"  Number of data points: {len(df_summary)}")
        
        # 時系列プロット用の出力ディレクトリ
        plot_output_dir = f"timeseries_plots_{mapper.dir_suffix}"
        os.makedirs(plot_output_dir, exist_ok=True)
        
        # 時間ビンを作成
        time_bin_h = 1.0  # 1時間ごとに集計
        time_bins = np.arange(
            np.floor(df_summary['time_h'].min()),
            np.ceil(df_summary['time_h'].max()) + time_bin_h,
            time_bin_h
        )
        
        # 時間ごとの統計を計算
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
        
        # プロット作成
        fig = plt.figure(figsize=(14, 14))
        gs = GridSpec(3, 1, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. 体積の時間変化
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
        
        # 2. 平均RIの時間変化
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
        
        # 3. Total Massの時間変化
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
        
        # 保存
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
    
    # 完了フラグファイルを作成（レジューム機能用）
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
