"""
QPI Analysis - Common Functions and Constants

共通で使用する関数と定数を定義
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.optimize import curve_fit
from qpi import QPIParameters, make_disk

# ==================== Default Parameters ====================

# 光学系の定数
WAVELENGTH = 663e-9  # 663 nm
NA = 0.95
PIXELSIZE = 3.45e-6 / 40  # 8.625e-8 m

# よく使うオフ軸中心の座標
OFFAXIS_CENTER_DEFAULT = (1623, 1621)

# ==================== Visibility Function ====================

def visibility(array: np.ndarray, params: QPIParameters) -> np.ndarray:
    """
    干渉縞の可視性（visibility）を計算
    
    Parameters
    ----------
    array : np.ndarray
        入力画像
    params : QPIParameters
        QPI パラメータ
    
    Returns
    -------
    np.ndarray
        可視性マップ
    """
    img = np.array(array)
    radius = params.aperturesize // 2
    
    # 0次光の画像を取得
    img_freq = np.fft.fftshift(np.fft.fft2(img))
    disk_0th = make_disk(params.img_center, radius, img.shape)
    img_freq_0th = img_freq * disk_0th
    img_0th = np.fft.ifft2(np.fft.ifftshift(img_freq_0th))
    img_0th_abs = np.abs(img_0th)
    
    # 1次光の画像を取得
    disk_1th = make_disk(params.offaxis_center, radius, img.shape)
    img_freq_interfere = img_freq * disk_1th
    img_interfere = np.fft.ifft2(np.fft.ifftshift(img_freq_interfere))
    img_interfere_abs = np.abs(img_interfere)
    
    # 可視性を計算
    img_interfere_vis = img_interfere_abs / img_0th_abs * 2
    
    return img_interfere_vis


# ==================== Gaussian Background Subtraction ====================

def gaussian(x, amp, mean, std):
    """ガウス関数"""
    return amp * np.exp(-((x - mean)**2) / (2 * std**2))


def gaussian_background_subtraction(phase_image, 
                                   hist_min=-1.1, hist_max=1.5, 
                                   n_bins=512, smooth_window=20,
                                   min_phase=-1.1):
    """
    位相画像のGaussian背景引き
    
    ヒストグラムのピークを検出し、Gaussianフィットして0にシフトする
    
    Parameters
    ----------
    phase_image : np.ndarray
        位相画像（rad）
    hist_min : float
        ヒストグラムの最小値
    hist_max : float
        ヒストグラムの最大値
    n_bins : int
        ヒストグラムのビン数
    smooth_window : int
        スムージングのウィンドウサイズ
    min_phase : float
        ピーク検出の下限
    
    Returns
    -------
    corrected_image : np.ndarray
        補正後の位相画像
    correction_value : float
        補正値（rad）
    """
    # ヒストグラム作成
    bin_edges = np.linspace(hist_min, hist_max, n_bins + 1)
    hist_counts, _ = np.histogram(phase_image.flatten(), bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # スムージング
    smoothed_histo = uniform_filter1d(hist_counts, size=smooth_window, mode='nearest')
    smoothed_histo = uniform_filter1d(smoothed_histo, size=smooth_window, mode='nearest')
    
    # ピーク検出
    valid_indices = np.where(bin_centers >= min_phase)[0]
    max_search_idx = int(len(bin_centers) * 0.95)
    search_indices = valid_indices[valid_indices < max_search_idx]
    
    search_range = smoothed_histo[search_indices]
    peak_idx_relative = np.argmax(search_range)
    peak_idx = search_indices[peak_idx_relative]
    peak_value = bin_centers[peak_idx]
    
    # Gaussianフィット
    fit_width = 300
    start_idx = max(0, peak_idx - fit_width)
    end_idx = min(len(bin_centers), peak_idx + fit_width)
    
    x_data = bin_centers[start_idx:end_idx]
    y_data = smoothed_histo[start_idx:end_idx]
    
    # 初期推定値
    p0 = [np.max(y_data), peak_value, (bin_centers[1] - bin_centers[0]) * 20]
    
    popt, _ = curve_fit(gaussian, x_data, y_data, p0=p0, maxfev=5000)
    amp, mean, std = popt
    
    # 背景補正（ピークを0にシフト）
    correction_value = 0.0 - mean
    corrected_image = phase_image + correction_value
    
    return corrected_image, correction_value


# ==================== Utility Functions ====================

def visualize_crop_region(image, crop_coords, title="Crop Region Visualization"):
    """
    Crop領域を元画像上に可視化
    
    Parameters
    ----------
    image : np.ndarray
        元画像
    crop_coords : tuple
        (y_start, y_end, x_start, x_end)
    title : str
        タイトル
    """
    y_s, y_e, x_s, x_e = crop_coords
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(image, cmap='gray')
    
    # Crop領域を矩形で描画
    from matplotlib.patches import Rectangle
    rect = Rectangle((x_s, y_s), x_e - x_s, y_e - y_s, 
                     linewidth=3, edgecolor='red', facecolor='none',
                     label=f'Crop Region\n({y_s}:{y_e}, {x_s}:{x_e})')
    ax.add_patch(rect)
    
    # 角に点を追加
    ax.plot([x_s, x_e, x_e, x_s], [y_s, y_s, y_e, y_e], 
           'ro', markersize=8)
    
    # 領域サイズを表示
    width = x_e - x_s
    height = y_e - y_s
    ax.text(x_s + width/2, y_s - 20, f'Width: {width}px', 
           ha='center', color='red', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(x_s - 20, y_s + height/2, f'Height: {height}px', 
           ha='right', va='center', color='red', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
           rotation=90)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12)
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    
    plt.tight_layout()
    return fig


def save_with_colormap(data, save_path, cmap="viridis", vmin=None, vmax=None, 
                       colorbar_label="", title="", dpi=300):
    """
    カラーマップ付きPNG画像として保存
    
    Parameters
    ----------
    data : np.ndarray
        保存するデータ
    save_path : str
        保存先パス
    cmap : str, optional
        カラーマップ名, by default "viridis"
    vmin : float, optional
        最小値, by default None
    vmax : float, optional
        最大値, by default None
    colorbar_label : str, optional
        カラーバーのラベル, by default ""
    title : str, optional
        タイトル, by default ""
    dpi : int, optional
        解像度, by default 300
    """
    plt.figure(figsize=(6, 6))
    im = plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, label=colorbar_label)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def to_uint8(img):
    """
    float画像をuint8に変換（OpenCV用）
    
    Parameters
    ----------
    img : np.ndarray
        入力画像（float）
    
    Returns
    -------
    np.ndarray
        uint8画像
    """
    img_min = np.min(img)
    img_max = np.max(img)
    if img_max - img_min > 0:
        normalized = (img - img_min) / (img_max - img_min)
    else:
        normalized = img
    return (normalized * 255).astype(np.uint8)


def create_qpi_params(img_shape, offaxis_center=OFFAXIS_CENTER_DEFAULT,
                      wavelength=WAVELENGTH, NA=NA, pixelsize=PIXELSIZE):
    """
    QPIParametersオブジェクトを作成するヘルパー関数
    
    Parameters
    ----------
    img_shape : tuple
        画像のshape
    offaxis_center : tuple, optional
        オフ軸中心座標, by default OFFAXIS_CENTER_DEFAULT
    wavelength : float, optional
        波長, by default WAVELENGTH
    NA : float, optional
        開口数, by default NA
    pixelsize : float, optional
        ピクセルサイズ, by default PIXELSIZE
    
    Returns
    -------
    QPIParameters
        QPIパラメータオブジェクト
    """
    return QPIParameters(
        wavelength=wavelength,
        NA=NA,
        img_shape=img_shape,
        pixelsize=pixelsize,
        offaxis_center=offaxis_center
    )

