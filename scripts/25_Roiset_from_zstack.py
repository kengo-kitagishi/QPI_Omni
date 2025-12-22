#!/usr/bin/env python3
"""
zstack.tifから境界線を抽出し、ImageJ RoiSet.zipを作成

機能:
1. zstack.tifファイルを読み込み
2. 閾値処理してバイナリマスクを作成
3. 境界線を抽出
4. 輪郭（contour）を検出
5. ImageJ ROI形式で保存（RoiSet.zip）
"""

import os
import glob
import numpy as np
import tifffile
from scipy import ndimage
from skimage import measure
import roifile
from roifile import ImagejRoi

def extract_boundary_from_zstack(zstack_path, threshold=0.5, output_dir=None):
    """
    zstack.tifから境界線を抽出し、ImageJ ROIを作成
    
    Parameters:
    -----------
    zstack_path : str
        zstack.tifファイルのパス
    threshold : float
        閾値（0以上の値をマスクとして扱う）
    output_dir : str or None
        出力ディレクトリ（Noneなら入力ファイルと同じ場所）
    
    Returns:
    --------
    roi_path : str
        保存されたROIファイルのパス
    """
    # zstackを読み込み
    print(f"Loading: {os.path.basename(zstack_path)}")
    zstack = tifffile.imread(zstack_path)
    print(f"  Shape: {zstack.shape}")
    print(f"  Value range: [{zstack.min():.4f}, {zstack.max():.4f}]")
    
    # バイナリマスクを作成（閾値以上を1、それ以外を0）
    binary_mask = (zstack > threshold).astype(np.uint8)
    print(f"  Mask pixels: {np.count_nonzero(binary_mask)}")
    
    if np.count_nonzero(binary_mask) == 0:
        print("  WARNING: No pixels above threshold!")
        return None
    
    # 境界線を抽出（エッジ検出）
    border = ((binary_mask != np.roll(binary_mask, 1, axis=0)) |
              (binary_mask != np.roll(binary_mask, -1, axis=0)) |
              (binary_mask != np.roll(binary_mask, 1, axis=1)) |
              (binary_mask != np.roll(binary_mask, -1, axis=1))) & (binary_mask > 0)
    
    print(f"  Border pixels: {np.count_nonzero(border)}")
    
    # 輪郭を検出（find_contours）
    contours = measure.find_contours(binary_mask, 0.5)
    print(f"  Found {len(contours)} contour(s)")
    
    if len(contours) == 0:
        print("  WARNING: No contours found!")
        return None
    
    # 最大の輪郭を選択（通常はロッド本体）
    largest_contour = max(contours, key=len)
    print(f"  Largest contour: {len(largest_contour)} points")
    
    # ImageJ ROIを作成
    # roifileの座標系: (row, col) → (y, x)
    # ImageJ ROI座標: (x, y)なので、列と行を入れ替える
    roi_coords = largest_contour[:, [1, 0]]  # (row, col) → (x, y)
    
    # ROI名を生成
    base_name = os.path.splitext(os.path.basename(zstack_path))[0]
    roi_name = base_name.replace('_zstack', '')
    
    # ImagejRoiオブジェクトを作成（Polygon型）
    roi = ImagejRoi.frompoints(roi_coords)
    roi.name = roi_name
    
    # 出力パスを決定
    if output_dir is None:
        output_dir = os.path.dirname(zstack_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ROIファイルを保存
    roi_filename = f"{roi_name}.roi"
    roi_path = os.path.join(output_dir, roi_filename)
    roi.tofile(roi_path)
    print(f"  Saved: {roi_filename}")
    
    # バイナリ画像も保存（オプション）
    binary_path = zstack_path.replace('_zstack.tif', '_binary.tif')
    binary_image = np.where(border, 0, 255).astype(np.uint8)
    tifffile.imwrite(binary_path, binary_image)
    print(f"  Saved: {os.path.basename(binary_path)}")
    
    return roi_path


def process_all_zstacks(input_dir, output_dir=None, threshold=0.5, create_roiset=True):
    """
    ディレクトリ内の全zstack.tifファイルを処理
    
    Parameters:
    -----------
    input_dir : str
        zstack.tifファイルが入ったディレクトリ
    output_dir : str or None
        出力ディレクトリ（Noneなら入力と同じ）
    threshold : float
        閾値
    create_roiset : bool
        RoiSet.zipを作成するか
    
    Returns:
    --------
    roi_paths : list
        作成されたROIファイルのパスリスト
    """
    # zstack.tifファイルを検索
    zstack_files = sorted(glob.glob(os.path.join(input_dir, "*_zstack.tif")))
    
    if not zstack_files:
        raise FileNotFoundError(f"No *_zstack.tif files found in {input_dir}")
    
    print(f"\n{'='*70}")
    print(f"Found {len(zstack_files)} zstack files")
    print(f"{'='*70}\n")
    
    if output_dir is None:
        output_dir = os.path.join(input_dir, "rois")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 各zstackファイルを処理
    roi_paths = []
    all_rois = []
    
    for i, zstack_path in enumerate(zstack_files, 1):
        print(f"[{i}/{len(zstack_files)}]")
        
        try:
            roi_path = extract_boundary_from_zstack(
                zstack_path,
                threshold=threshold,
                output_dir=output_dir
            )
            
            if roi_path:
                roi_paths.append(roi_path)
                # ROIを読み込んでリストに追加（RoiSet用）
                roi = ImagejRoi.fromfile(roi_path)
                all_rois.append(roi)
        
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    # RoiSet.zipを作成
    if create_roiset and all_rois:
        roiset_path = os.path.join(output_dir, "RoiSet.zip")
        roifile.roiwrite(roiset_path, all_rois)
        print(f"\n{'='*70}")
        print(f"Created RoiSet.zip with {len(all_rois)} ROIs")
        print(f"Saved: {roiset_path}")
        print(f"{'='*70}\n")
    
    return roi_paths


def create_roiset_from_existing_rois(roi_dir, output_path=None):
    """
    既存の.roiファイルからRoiSet.zipを作成
    
    Parameters:
    -----------
    roi_dir : str
        .roiファイルが入ったディレクトリ
    output_path : str or None
        RoiSet.zipの出力パス（Noneならroi_dir/RoiSet.zip）
    """
    # .roiファイルを検索
    roi_files = sorted(glob.glob(os.path.join(roi_dir, "*.roi")))
    
    if not roi_files:
        raise FileNotFoundError(f"No .roi files found in {roi_dir}")
    
    print(f"Found {len(roi_files)} ROI files")
    
    # 全ROIを読み込み
    all_rois = []
    for roi_file in roi_files:
        roi = ImagejRoi.fromfile(roi_file)
        all_rois.append(roi)
        print(f"  Loaded: {os.path.basename(roi_file)}")
    
    # RoiSet.zipを保存
    if output_path is None:
        output_path = os.path.join(roi_dir, "RoiSet.zip")
    
    roifile.roiwrite(output_path, all_rois)
    print(f"\nCreated RoiSet.zip: {output_path}")
    print(f"Total ROIs: {len(all_rois)}")


# ===== メイン実行 =====
if __name__ == "__main__":
    # パラメータ設定
    INPUT_DIR = "/mnt/user-data/outputs/timeseries_density_output/density_tiff"
    OUTPUT_DIR = "/mnt/user-data/outputs/timeseries_density_output/rois"
    THRESHOLD = 0.5  # zstack値の閾値（0.5以上をマスクとして扱う）
    
    # 全zstackファイルを処理してROIを作成
    roi_paths = process_all_zstacks(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        threshold=THRESHOLD,
        create_roiset=True
    )
    
    print(f"\n{'#'*70}")
    print(f"# Processing completed!")
    print(f"# Created {len(roi_paths)} ROI files")
    print(f"# Output directory: {OUTPUT_DIR}")
    print(f"# RoiSet.zip: {os.path.join(OUTPUT_DIR, 'RoiSet.zip')}")
    print(f"{'#'*70}\n")