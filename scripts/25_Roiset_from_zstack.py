#!/usr/bin/env python3
"""
zstack.tifから境界線を抽出し、密度マップをRefractive Index (RI)に変換して可視化

機能:
1. zstack.tifファイルを読み込み（厚みマップ）
2. 閾値処理してバイナリマスクを作成
3. 境界線を抽出
4. 輪郭（contour）を検出
5. 密度マップを位相差→屈折率（RI）に変換
6. RIマップに輪郭線を重ねて可視化
7. マスク適用範囲を確認できる画像を保存

物理原理:
位相差 φ = (2π/λ) × (n_sample - n_medium) × thickness
屈折率 n_sample = n_medium + (φ × λ) / (2π × thickness)
"""
# %% 
import os
import glob
import numpy as np
import tifffile
from scipy import ndimage
from skimage import measure
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection

def phase_to_refractive_index(phase_map, thickness_map, wavelength_nm=663, 
                               n_medium=1.333, pixel_size_um=0.348):
    """
    位相差マップから屈折率（RI）マップを計算
    
    Parameters:
    -----------
    phase_map : numpy array
        位相差マップ（ラジアン単位）
    thickness_map : numpy array
        厚みマップ（ピクセル単位）
    wavelength_nm : float
        波長（ナノメートル）。デフォルト: 663nm (赤色レーザー)
    n_medium : float
        培地の屈折率。デフォルト: 1.333 (水性培地)
    pixel_size_um : float
        ピクセルサイズ（マイクロメートル）。デフォルト: 0.348 µm
        507×507の再構成画像用
    
    Returns:
    --------
    ri_map : numpy array
        屈折率マップ
    
    Formula:
    --------
    φ = (2π/λ) × (n_sample - n_medium) × thickness
    n_sample = n_medium + (φ × λ) / (2π × thickness)
    """
    # 単位を揃える: すべてµmに変換
    wavelength_um = wavelength_nm / 1000.0  # nm → µm
    thickness_um = thickness_map * pixel_size_um  # ピクセル → µm
    
    # マスク: 厚みが0でない領域のみ計算
    mask = thickness_um > 0
    
    # 屈折率マップを初期化（培地の屈折率で）
    ri_map = np.full_like(phase_map, n_medium, dtype=np.float64)
    
    # 位相差から屈折率を計算
    # n_sample = n_medium + (φ × λ) / (2π × thickness)
    if np.any(mask):
        ri_map[mask] = n_medium + (phase_map[mask] * wavelength_um) / (2 * np.pi * thickness_um[mask])
    
    return ri_map


def ri_to_concentration(ri_map, n_medium=1.333, alpha_ri=0.0018):
    """
    屈折率マップから質量濃度マップを計算
    
    Parameters:
    -----------
    ri_map : numpy array
        屈折率マップ
    n_medium : float
        培地の屈折率。デフォルト: 1.333
    alpha_ri : float
        比屈折率増分 [ml/mg]。デフォルト: 0.0018
    
    Returns:
    --------
    concentration_map : numpy array
        質量濃度マップ [mg/ml]
    
    Formula:
    --------
    C [mg/ml] = (RI - RI_medium) / α
    """
    concentration_map = (ri_map - n_medium) / alpha_ri
    return concentration_map


def visualize_mask_on_density(zstack_path, density_path=None, threshold=0, output_dir=None,
                               wavelength_nm=663, n_medium=1.333, pixel_size_um=0.348,
                               alpha_ri=0.0018):
    """
    zstack.tifから境界線を抽出し、密度マップを屈折率（RI）に変換して可視化
    
    Parameters:
    -----------
    zstack_path : str
        zstack.tifファイルのパス（厚みマップ）
    density_path : str or None
        密度マップファイルのパス（位相差マップ、Noneなら自動検索）
    threshold : float
        閾値（この値より大きい値をマスクとして扱う。デフォルト: 0）
    output_dir : str or None
        出力ディレクトリ（Noneなら入力ファイルと同じ場所）
    wavelength_nm : float
        波長（ナノメートル）。デフォルト: 663nm
    n_medium : float
        培地の屈折率。デフォルト: 1.333
    pixel_size_um : float
        ピクセルサイズ（マイクロメートル）。デフォルト: 0.348 µm
    alpha_ri : float
        比屈折率増分 [ml/mg]。デフォルト: 0.0018
    
    Returns:
    --------
    output_path : str
        保存された可視化画像のパス
    
    Note:
    -----
    - 輪郭線はマスク領域の境界を示します
    - 輪郭線の内側がマスク適用領域です
    - 密度マップを位相差と解釈し、屈折率（RI）と質量濃度（mg/ml）に変換します
    """
    # zstackを読み込み（厚みマップ）
    print(f"Loading: {os.path.basename(zstack_path)}")
    zstack = tifffile.imread(zstack_path)
    print(f"  Zstack (thickness) shape: {zstack.shape}")
    print(f"  Thickness range: [{zstack.min():.4f}, {zstack.max():.4f}] pixels")
    print(f"  Thickness range: [{zstack.min()*pixel_size_um:.4f}, {zstack.max()*pixel_size_um:.4f}] µm")
    
    # バイナリマスクを作成（閾値より大きい値を1、それ以外を0）
    binary_mask = (zstack > threshold).astype(np.uint8)
    print(f"  Threshold: > {threshold}")
    print(f"  Mask pixels: {np.count_nonzero(binary_mask)} / {binary_mask.size} ({100*np.count_nonzero(binary_mask)/binary_mask.size:.1f}%)")
    
    if np.count_nonzero(binary_mask) == 0:
        print("  WARNING: No pixels above threshold!")
        return None
    
    # 輪郭を検出（find_contours）
    contours = measure.find_contours(binary_mask, 0.5)
    print(f"  Found {len(contours)} contour(s)")
    
    if len(contours) == 0:
        print("  WARNING: No contours found!")
        return None
    
    # 密度マップを読み込み（位相差マップ）
    if density_path is None:
        # zstack.tifと同じ名前の密度マップを探す
        base_name = os.path.basename(zstack_path).replace('_zstack.tif', '')
        dir_name = os.path.dirname(zstack_path)
        
        # 可能性のあるファイル名パターン
        possible_patterns = [
            f"{base_name}_density.tif",
            f"{base_name}.tif",
            f"{base_name}_mean.tif",
        ]
        
        for pattern in possible_patterns:
            candidate_path = os.path.join(dir_name, pattern)
            if os.path.exists(candidate_path):
                density_path = candidate_path
                break
        
        if density_path is None:
            print("  ERROR: Could not find corresponding density map!")
            print(f"  Searched for: {possible_patterns}")
            return None
    
    # 密度マップを読み込み（位相差マップとして解釈）
    print(f"  Loading phase map: {os.path.basename(density_path)}")
    phase_map = tifffile.imread(density_path)
    print(f"  Phase map shape: {phase_map.shape}")
    print(f"  Phase range: [{phase_map.min():.4f}, {phase_map.max():.4f}] (arbitrary units)")
    
    # 位相差を屈折率（RI）に変換
    print(f"  Converting phase to refractive index...")
    print(f"    Wavelength: {wavelength_nm} nm")
    print(f"    Medium RI: {n_medium}")
    print(f"    Pixel size: {pixel_size_um} µm")
    
    ri_map = phase_to_refractive_index(
        phase_map, zstack, 
        wavelength_nm=wavelength_nm,
        n_medium=n_medium,
        pixel_size_um=pixel_size_um
    )
    
    mask = binary_mask > 0
    if np.any(mask):
        print(f"  RI range (masked region): [{ri_map[mask].min():.6f}, {ri_map[mask].max():.6f}]")
        print(f"  Mean RI (masked region): {ri_map[mask].mean():.6f}")
    
    # 質量濃度マップを計算
    print(f"  Converting RI to protein concentration...")
    print(f"    Alpha (RI increment): {alpha_ri} ml/mg")
    concentration_map = ri_to_concentration(ri_map, n_medium=n_medium, alpha_ri=alpha_ri)
    
    if np.any(mask):
        print(f"  Concentration range (masked region): [{concentration_map[mask].min():.2f}, {concentration_map[mask].max():.2f}] mg/ml")
        print(f"  Mean concentration (masked region): {concentration_map[mask].mean():.2f} mg/ml")
    
    # 出力パスを決定
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(zstack_path), "visualized")
    
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(zstack_path))[0]
    output_filename = f"{base_name}_RI_Conc_map.png"
    output_path = os.path.join(output_dir, output_filename)
    
    # 可視化（RIマップと質量濃度マップを横に並べる）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10), dpi=150)
    
    # === 左側: RIマップ ===
    # カラースケールの範囲を設定（生理的に妥当な範囲）
    # 細胞質: ~1.35-1.37
    # 核: ~1.38-1.40
    # タンパク質凝集体: ~1.40-1.45
    vmin_ri = n_medium  # 培地の屈折率
    vmax_ri = max(1.45, ri_map[mask].max() if np.any(mask) else 1.45)  # 上限を設定
    
    im1 = ax1.imshow(ri_map, cmap='jet', interpolation='nearest', 
                     vmin=vmin_ri, vmax=vmax_ri)
    
    # すべての輪郭を白線で描画（視認性向上のため）
    for contour in contours:
        ax1.plot(contour[:, 1], contour[:, 0], 'w-', linewidth=2.5, alpha=0.9)
        ax1.plot(contour[:, 1], contour[:, 0], 'r-', linewidth=1.5, alpha=0.8)
    
    # カラーバーを追加
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Refractive Index (RI)', rotation=270, labelpad=25, fontsize=12)
    
    # 参照値を追加
    cbar1.ax.axhline(y=n_medium, color='cyan', linestyle='--', linewidth=1.5, alpha=0.8)
    cbar1.ax.text(1.5, n_medium, f'Medium ({n_medium:.3f})', va='center', fontsize=9, color='cyan')
    
    if np.any(mask):
        mean_ri = ri_map[mask].mean()
        cbar1.ax.axhline(y=mean_ri, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
        cbar1.ax.text(1.5, mean_ri, f'Mean ({mean_ri:.3f})', va='center', fontsize=9, color='white')
    
    # タイトルに情報を追加
    mask_percentage = 100 * np.count_nonzero(binary_mask) / binary_mask.size
    title1 = f'Refractive Index Map\n{base_name.replace("_zstack", "")}\n'
    title1 += f'λ={wavelength_nm}nm, n_medium={n_medium:.3f}\n'
    title1 += f'(Red line = mask boundary, {mask_percentage:.1f}% masked)'
    ax1.set_title(title1, fontsize=11)
    ax1.axis('off')
    
    # === 右側: 質量濃度マップ ===
    # カラースケールの範囲を設定
    vmin_conc = 0
    vmax_conc = max(50, concentration_map[mask].max() if np.any(mask) else 50)
    
    im2 = ax2.imshow(concentration_map, cmap='hot', interpolation='nearest',
                     vmin=vmin_conc, vmax=vmax_conc)
    
    # すべての輪郭を白線で描画（視認性向上のため）
    for contour in contours:
        ax2.plot(contour[:, 1], contour[:, 0], 'w-', linewidth=2.5, alpha=0.9)
        ax2.plot(contour[:, 1], contour[:, 0], 'cyan-', linewidth=1.5, alpha=0.8)
    
    # カラーバーを追加
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Concentration (mg/ml)', rotation=270, labelpad=25, fontsize=12)
    
    # 平均値を追加
    if np.any(mask):
        mean_conc = concentration_map[mask].mean()
        cbar2.ax.axhline(y=mean_conc, color='cyan', linestyle='--', linewidth=1.5, alpha=0.8)
        cbar2.ax.text(1.5, mean_conc, f'Mean ({mean_conc:.1f})', va='center', fontsize=9, color='cyan')
    
    # タイトルに情報を追加
    title2 = f'Protein Concentration Map\n{base_name.replace("_zstack", "")}\n'
    title2 += f'α={alpha_ri} ml/mg, pixel={pixel_size_um}µm\n'
    title2 += f'(Cyan line = mask boundary, {mask_percentage:.1f}% masked)'
    ax2.set_title(title2, fontsize=11)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"  Saved: {output_filename}")
    
    return output_path


def process_all_zstacks(input_dir, output_dir=None, threshold=0,
                        wavelength_nm=663, n_medium=1.333, pixel_size_um=0.348,
                        alpha_ri=0.0018):
    """
    ディレクトリ内の全zstack.tifファイルを処理して可視化
    
    Parameters:
    -----------
    input_dir : str
        zstack.tifファイルが入ったディレクトリ
    output_dir : str or None
        出力ディレクトリ（Noneなら自動設定）
    threshold : float
        閾値（この値より大きい値をマスクとして扱う。デフォルト: 0）
    wavelength_nm : float
        波長（ナノメートル）。デフォルト: 663nm
    n_medium : float
        培地の屈折率。デフォルト: 1.333
    pixel_size_um : float
        ピクセルサイズ（マイクロメートル）。デフォルト: 0.348 µm
    alpha_ri : float
        比屈折率増分 [ml/mg]。デフォルト: 0.0018
    
    Returns:
    --------
    output_paths : list
        作成された可視化画像のパスリスト
    """
    # zstack.tifファイルを検索
    zstack_files = sorted(glob.glob(os.path.join(input_dir, "*_zstack.tif")))
    
    if not zstack_files:
        raise FileNotFoundError(f"No *_zstack.tif files found in {input_dir}")
    
    print(f"\n{'='*70}")
    print(f"Found {len(zstack_files)} zstack files")
    print(f"{'='*70}\n")
    
    if output_dir is None:
        output_dir = os.path.join(input_dir, "visualized")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 各zstackファイルを処理
    output_paths = []
    
    for i, zstack_path in enumerate(zstack_files, 1):
        print(f"[{i}/{len(zstack_files)}]")
        
        try:
            output_path = visualize_mask_on_density(
                zstack_path,
                threshold=threshold,
                output_dir=output_dir,
                wavelength_nm=wavelength_nm,
                n_medium=n_medium,
                pixel_size_um=pixel_size_um,
                alpha_ri=alpha_ri
            )
            
            if output_path:
                output_paths.append(output_path)
        
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    return output_paths


# ===== メイン実行 =====
if __name__ == "__main__":
    # パラメータ設定
    INPUT_DIR = "/mnt/user-data/outputs/timeseries_density_output/density_tiff"
    OUTPUT_DIR = "/mnt/user-data/outputs/timeseries_density_output/visualized"
    THRESHOLD = 0  # zstack値の閾値（この値より大きい値をマスクとして扱う）
    
    # QPI実験パラメータ（01_QPI_analysis.pyと同じ値）
    WAVELENGTH_NM = 663      # レーザー波長（ナノメートル）
                             # 実験値: 663nm (赤色レーザー)
    N_MEDIUM = 1.333         # 培地の屈折率
                             # 水: 1.333, PBS: 1.334, DMEM: ~1.335
    PIXEL_SIZE_UM = 0.348    # ピクセルサイズ（マイクロメートル）
                             # 507×507の再構成画像用
                             # 計算: 0.08625 µm × (2048/507) ≈ 0.348 µm/pixel
                             # ※元のホログラム2048×2048では0.08625 µm/pixel
    ALPHA_RI = 0.0018        # 比屈折率増分 [ml/mg]
                             # タンパク質の一般的な値: 0.0018 ml/mg
                             # 参考: C [mg/ml] = (RI - RI_medium) / α
    
    print(f"\n{'='*70}")
    print(f"QPI Parameters:")
    print(f"  Wavelength: {WAVELENGTH_NM} nm")
    print(f"  Medium RI: {N_MEDIUM}")
    print(f"  Pixel size: {PIXEL_SIZE_UM} µm (for 507×507 reconstructed images)")
    print(f"  Alpha (RI increment): {ALPHA_RI} ml/mg")
    print(f"  Note: Concentration (mg/ml) = (RI - {N_MEDIUM}) / {ALPHA_RI}")
    print(f"{'='*70}\n")
    
    # 全zstackファイルを処理して可視化画像を作成
    # 注意: 輪郭線の内側がマスク適用領域を示します
    output_paths = process_all_zstacks(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        threshold=THRESHOLD,
        wavelength_nm=WAVELENGTH_NM,
        n_medium=N_MEDIUM,
        pixel_size_um=PIXEL_SIZE_UM,
        alpha_ri=ALPHA_RI
    )
    
    print(f"\n{'#'*70}")
    print(f"# Processing completed!")
    print(f"# Created {len(output_paths)} RI & Concentration visualization images")
    print(f"# Output directory: {OUTPUT_DIR}")
    print(f"# Note: Images show Refractive Index (RI) and Protein Concentration maps")
    print(f"#       Red/Cyan contour = mask boundary (inside = masked region)")
    print(f"#       RI range: {N_MEDIUM:.3f} (medium) to ~1.40 (protein)")
    print(f"#       Concentration: C (mg/ml) = (RI - {N_MEDIUM:.3f}) / {ALPHA_RI}")
    print(f"{'#'*70}\n")
# %%
