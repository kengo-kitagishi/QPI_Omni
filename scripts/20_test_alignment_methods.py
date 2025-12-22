# %%
"""
多様なアライメント手法の比較実験
- Template Matching (OpenCV)
- Feature-based (ORB, SIFT)
- ECC (Enhanced Correlation Coefficient)
- imreg_dft
- scikit-image の各種手法
"""

import numpy as np
from skimage import io, registration, transform, feature
from scipy import ndimage, optimize
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# OpenCVのインポート（エラーハンドリング付き）
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("警告: OpenCVがインストールされていません。一部の機能が使えません。")
    print("インストール: pip install opencv-python")

# imreg_dftのインポート（エラーハンドリング付き）
try:
    import imreg_dft as ird
    HAS_IMREG = True
except ImportError:
    HAS_IMREG = False
    print("警告: imreg_dftがインストールされていません。")
    print("インストール: pip install imreg_dft")

def load_tif_image(path):
    """TIF画像を読み込んでfloatに変換"""
    img = io.imread(path)
    return img.astype(np.float64)

def normalize_for_display(img):
    """表示用に0-1に正規化（アライメントには使用しない）"""
    img_min = np.min(img)
    img_max = np.max(img)
    if img_max - img_min > 0:
        return (img - img_min) / (img_max - img_min)
    return img

def to_uint8(img):
    """uint8に変換（OpenCV用）"""
    normalized = normalize_for_display(img)
    return (normalized * 255).astype(np.uint8)

# ================================================================
# 方法1: 位相相関法 (Phase Cross Correlation) - scikit-image
# ================================================================
def align_phase_correlation(reference_img, target_img, upsample=10):
    """
    位相相関法 - 高速で正確
    """
    shift, error, diffphase = registration.phase_cross_correlation(
        reference_img, target_img, upsample_factor=upsample
    )
    aligned_img = ndimage.shift(target_img, shift, order=1)
    return aligned_img, shift, error

# ================================================================
# 方法2: テンプレートマッチング (Template Matching) - OpenCV
# ================================================================
def align_template_matching(reference_img, target_img, method=cv2.TM_CCOEFF_NORMED):
    """
    テンプレートマッチング - シンプルだが大きなずれには弱い
    
    Parameters:
    -----------
    method : int
        cv2.TM_CCOEFF_NORMED (推奨)
        cv2.TM_CCORR_NORMED
        cv2.TM_SQDIFF_NORMED
    """
    if not HAS_CV2:
        raise ImportError("OpenCVが必要です")
    
    ref_uint8 = to_uint8(reference_img)
    tgt_uint8 = to_uint8(target_img)
    
    # 全体をテンプレートとして使用
    result = cv2.matchTemplate(tgt_uint8, ref_uint8, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    # シフト量を計算
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    
    shift = np.array([0, 0])  # テンプレート全体なのでシフトは0
    error = 1.0 - max_val if method == cv2.TM_CCOEFF_NORMED else min_val
    
    # 実際には部分マッチングの方が有用なので、中心領域を使う
    h, w = reference_img.shape
    template_size = (int(h * 0.8), int(w * 0.8))
    template_offset = ((h - template_size[0]) // 2, (w - template_size[1]) // 2)
    
    template = ref_uint8[
        template_offset[0]:template_offset[0] + template_size[0],
        template_offset[1]:template_offset[1] + template_size[1]
    ]
    
    result = cv2.matchTemplate(tgt_uint8, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        match_loc = min_loc
    else:
        match_loc = max_loc
    
    shift = np.array([
        match_loc[1] - template_offset[0],
        match_loc[0] - template_offset[1]
    ])
    
    aligned_img = ndimage.shift(target_img, shift, order=1)
    error = 1.0 - max_val if method == cv2.TM_CCOEFF_NORMED else min_val
    
    return aligned_img, shift, error

# ================================================================
# 方法3: ECC (Enhanced Correlation Coefficient) - OpenCV
# ================================================================
def align_ecc(reference_img, target_img, motion_type='translation'):
    """
    ECC Image Alignment - 反復的最適化、高精度
    
    Parameters:
    -----------
    motion_type : str
        'translation' - 並進のみ
        'euclidean' - 並進+回転
        'affine' - アフィン変換
    """
    if not HAS_CV2:
        raise ImportError("OpenCVが必要です")
    
    ref_uint8 = to_uint8(reference_img)
    tgt_uint8 = to_uint8(target_img)
    
    # モーション型の設定
    if motion_type == 'translation':
        warp_mode = cv2.MOTION_TRANSLATION
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    elif motion_type == 'euclidean':
        warp_mode = cv2.MOTION_EUCLIDEAN
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    elif motion_type == 'affine':
        warp_mode = cv2.MOTION_AFFINE
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    else:
        raise ValueError(f"Unknown motion_type: {motion_type}")
    
    # ECC実行
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-6)
    try:
        cc, warp_matrix = cv2.findTransformECC(
            ref_uint8, tgt_uint8, warp_matrix, warp_mode, criteria
        )
    except cv2.error as e:
        print(f"    ECC警告: {e}")
        return target_img, np.array([0, 0]), 1.0
    
    # 変換適用
    aligned_uint8 = cv2.warpAffine(
        tgt_uint8, warp_matrix, 
        (tgt_uint8.shape[1], tgt_uint8.shape[0]),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
    )
    
    # float64に戻す
    aligned_img = aligned_uint8.astype(np.float64) / 255.0
    aligned_img = aligned_img * (np.max(target_img) - np.min(target_img)) + np.min(target_img)
    
    # シフト量抽出（並進成分）
    shift = np.array([warp_matrix[1, 2], warp_matrix[0, 2]])
    error = 1.0 - cc
    
    return aligned_img, shift, error

# ================================================================
# 方法4: 特徴点ベース (Feature-based) - ORB - OpenCV
# ================================================================
def align_feature_orb(reference_img, target_img, n_features=500):
    """
    ORB特徴点ベースのアライメント - 大きなずれに強い
    """
    if not HAS_CV2:
        raise ImportError("OpenCVが必要です")
    
    ref_uint8 = to_uint8(reference_img)
    tgt_uint8 = to_uint8(target_img)
    
    # ORB検出器
    orb = cv2.ORB_create(n_features)
    kp1, des1 = orb.detectAndCompute(ref_uint8, None)
    kp2, des2 = orb.detectAndCompute(tgt_uint8, None)
    
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        print("    特徴点不足")
        return target_img, np.array([0, 0]), 1.0
    
    # マッチング
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    if len(matches) < 4:
        print("    マッチング不足")
        return target_img, np.array([0, 0]), 1.0
    
    # 変換行列推定
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
    
    M, mask = cv2.estimateAffinePartial2D(dst_pts, src_pts)
    
    if M is None:
        print("    変換推定失敗")
        return target_img, np.array([0, 0]), 1.0
    
    # 変換適用
    aligned_uint8 = cv2.warpAffine(tgt_uint8, M, (tgt_uint8.shape[1], tgt_uint8.shape[0]))
    
    # float64に戻す
    aligned_img = aligned_uint8.astype(np.float64) / 255.0
    aligned_img = aligned_img * (np.max(target_img) - np.min(target_img)) + np.min(target_img)
    
    # シフト量
    shift = np.array([M[1, 2], M[0, 2]])
    error = np.mean([m.distance for m in matches[:10]]) / 100.0
    
    return aligned_img, shift, error

# ================================================================
# 方法5: imreg_dft - Fourier-based
# ================================================================
def align_imreg_dft(reference_img, target_img):
    """
    imreg_dft - FFTベース、回転・スケール・並進に対応
    """
    if not HAS_IMREG:
        raise ImportError("imreg_dftが必要です")
    
    result = ird.similarity(reference_img, target_img, numiter=3)
    
    aligned_img = result['timg']
    shift = np.array([result['ty'], result['tx']])
    error = 1.0 - result['success']
    
    return aligned_img, shift, error

# ================================================================
# 方法6: Optical Flow - Lucas-Kanade
# ================================================================
def align_optical_flow(reference_img, target_img):
    """
    オプティカルフロー - 密なピクセル対応
    """
    if not HAS_CV2:
        raise ImportError("OpenCVが必要です")
    
    ref_uint8 = to_uint8(reference_img)
    tgt_uint8 = to_uint8(target_img)
    
    # 平均フローを計算
    flow = cv2.calcOpticalFlowFarneback(
        ref_uint8, tgt_uint8, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    
    # 平均シフトを計算
    shift = np.array([np.median(flow[:, :, 1]), np.median(flow[:, :, 0])])
    
    # シフト適用
    aligned_img = ndimage.shift(target_img, shift, order=1)
    
    error = np.std(flow)
    
    return aligned_img, shift, error

# ================================================================
# 統合処理関数
# ================================================================
def process_with_method(empty_channel_path, folder_path, output_folder, 
                       method='phase_correlation', mode='fixed',
                       vmin=-0.1, vmax=1.7, **kwargs):
    """
    指定した方法でアライメント処理
    
    Parameters:
    -----------
    method : str
        'phase_correlation' - 位相相関法
        'template_matching' - テンプレートマッチング
        'ecc' - ECC
        'feature_orb' - ORB特徴点
        'imreg_dft' - imreg_dft
        'optical_flow' - オプティカルフロー
    mode : str
        'sequential' - 連続アライメント
        'fixed' - 固定基準アライメント
    """
    print("=" * 80)
    print(f"アライメント処理")
    print(f"  方法: {method}")
    print(f"  モード: {mode}")
    print("=" * 80)
    
    # フォルダ作成
    os.makedirs(output_folder, exist_ok=True)
    aligned_folder = os.path.join(output_folder, "aligned")
    subtracted_folder = os.path.join(output_folder, "subtracted")
    colored_folder = os.path.join(output_folder, "colored")
    os.makedirs(aligned_folder, exist_ok=True)
    os.makedirs(subtracted_folder, exist_ok=True)
    os.makedirs(colored_folder, exist_ok=True)
    
    # 空チャネル読み込み
    print(f"\n[1] 空チャネル画像読み込み...")
    empty_img = load_tif_image(empty_channel_path)
    print(f"    サイズ: {empty_img.shape}")
    empty_filename = Path(empty_channel_path).name
    
    # ファイルリスト取得
    tif_files = []
    for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
        tif_files.extend(Path(folder_path).glob(ext))
    tif_files = [f for f in tif_files if f.name != empty_filename]
    tif_files = sorted(tif_files)
    
    if len(tif_files) == 0:
        print("\n警告: TIFファイルが見つかりません")
        return
    
    print(f"\n[2] 処理ファイル数: {len(tif_files)}")
    print(f"\n[3] 処理開始...")
    
    results = []
    reference_img = empty_img
    cumulative_shift = np.array([0.0, 0.0])
    
    # カラーマップ設定
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    
    # アライメント関数の選択
    align_func_map = {
        'phase_correlation': lambda r, t: align_phase_correlation(r, t, kwargs.get('upsample', 10)),
        'template_matching': align_template_matching,
        'ecc': lambda r, t: align_ecc(r, t, kwargs.get('motion_type', 'translation')),
        'feature_orb': lambda r, t: align_feature_orb(r, t, kwargs.get('n_features', 500)),
        'imreg_dft': align_imreg_dft,
        'optical_flow': align_optical_flow,
    }
    
    if method not in align_func_map:
        raise ValueError(f"Unknown method: {method}")
    
    align_func = align_func_map[method]
    
    for i, tif_path in enumerate(tif_files, 1):
        print(f"\n  [{i}/{len(tif_files)}] {tif_path.name}")
        
        try:
            target_img = load_tif_image(str(tif_path))
            
            if target_img.shape != empty_img.shape:
                print(f"    警告: サイズ不一致 - スキップ")
                continue
            
            # アライメント実行
            if mode == 'sequential':
                aligned_img, shift, error = align_func(reference_img, target_img)
                cumulative_shift += shift
                reference_img = aligned_img
                print(f"    相対シフト: Y={shift[0]:.2f}, X={shift[1]:.2f}")
                print(f"    累積シフト: Y={cumulative_shift[0]:.2f}, X={cumulative_shift[1]:.2f}")
            else:
                aligned_img, shift, error = align_func(empty_img, target_img)
                print(f"    シフト: Y={shift[0]:.2f}, X={shift[1]:.2f}")
            
            print(f"    誤差: {error:.4f}")
            
            # 差分計算
            subtracted = aligned_img - empty_img
            
            # 保存
            base_name = tif_path.stem
            aligned_path = os.path.join(aligned_folder, f"{base_name}_aligned.tif")
            subtracted_path = os.path.join(subtracted_folder, f"{base_name}_subtracted.tif")
            colored_path = os.path.join(colored_folder, f"{base_name}_colored.png")
            
            io.imsave(aligned_path, aligned_img.astype(np.float32))
            io.imsave(subtracted_path, subtracted.astype(np.float32))
            
            # カラー画像保存
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(subtracted, cmap='RdBu_r', norm=norm)
            ax.axis('off')
            ax.set_title(f'{base_name}\n平均: {np.mean(subtracted):.3f}, 標準偏差: {np.std(subtracted):.3f}')
            plt.colorbar(im, ax=ax, fraction=0.046, label='差分 (a.u.)')
            plt.tight_layout()
            plt.savefig(colored_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 結果記録
            result = {
                'filename': tif_path.name,
                'shift_y': shift[0],
                'shift_x': shift[1],
                'error': error,
                'subtracted_mean': np.mean(subtracted),
                'subtracted_std': np.std(subtracted)
            }
            if mode == 'sequential':
                result['cumulative_shift_y'] = cumulative_shift[0]
                result['cumulative_shift_x'] = cumulative_shift[1]
            results.append(result)
            
        except Exception as e:
            print(f"    エラー: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # レポート保存
    print("\n" + "=" * 80)
    print("処理完了")
    print("=" * 80)
    print(f"成功: {len(results)}ファイル")
    print(f"出力: {output_folder}")
    
    if len(results) > 0:
        report_path = os.path.join(output_folder, f"report_{method}.csv")
        with open(report_path, 'w', encoding='utf-8') as f:
            if mode == 'sequential':
                f.write("Filename,Shift_Y,Shift_X,Cumulative_Y,Cumulative_X,Error,Mean,Std\n")
                for r in results:
                    f.write(f"{r['filename']},{r['shift_y']:.4f},{r['shift_x']:.4f},"
                           f"{r['cumulative_shift_y']:.4f},{r['cumulative_shift_x']:.4f},"
                           f"{r['error']:.6f},{r['subtracted_mean']:.4f},{r['subtracted_std']:.4f}\n")
            else:
                f.write("Filename,Shift_Y,Shift_X,Error,Mean,Std\n")
                for r in results:
                    f.write(f"{r['filename']},{r['shift_y']:.4f},{r['shift_x']:.4f},"
                           f"{r['error']:.6f},{r['subtracted_mean']:.4f},{r['subtracted_std']:.4f}\n")
        print(f"レポート保存: {report_path}")
        
        # 統計表示
        shifts_y = [r['shift_y'] for r in results]
        shifts_x = [r['shift_x'] for r in results]
        errors = [r['error'] for r in results]
        print(f"\n統計:")
        print(f"  Y: 平均={np.mean(shifts_y):.2f}, 標準偏差={np.std(shifts_y):.2f}, 範囲=[{np.min(shifts_y):.2f}, {np.max(shifts_y):.2f}]")
        print(f"  X: 平均={np.mean(shifts_x):.2f}, 標準偏差={np.std(shifts_x):.2f}, 範囲=[{np.min(shifts_x):.2f}, {np.max(shifts_x):.2f}]")
        print(f"  誤差: 平均={np.mean(errors):.4f}, 標準偏差={np.std(errors):.4f}")
    
    return results

# ================================================================
# メイン実行
# ================================================================
if __name__ == "__main__":
    # パス設定
    empty_channel_path = r"C:\Users\QPI\Desktop\align_demo\subtracted_by_maskmean_float320001.tif"
    folder_path = r"C:\Users\QPI\Desktop\align_demo"
    base_output = r"C:\Users\QPI\Desktop\align_demo"
    
    # 実験する方法のリスト
    experiments = [
        # 基本的な方法
        {'method': 'phase_correlation', 'mode': 'fixed', 'name': '位相相関_固定'},
        {'method': 'phase_correlation', 'mode': 'sequential', 'name': '位相相関_連続'},
    ]
    
    # OpenCVが使える場合
    if HAS_CV2:
        experiments.extend([
            {'method': 'template_matching', 'mode': 'fixed', 'name': 'テンプレート_固定'},
            {'method': 'ecc', 'mode': 'fixed', 'name': 'ECC_固定', 'motion_type': 'translation'},
            {'method': 'feature_orb', 'mode': 'fixed', 'name': 'ORB特徴点_固定', 'n_features': 500},
            {'method': 'optical_flow', 'mode': 'fixed', 'name': 'オプティカルフロー_固定'},
        ])
    
    # imreg_dftが使える場合
    if HAS_IMREG:
        experiments.append(
            {'method': 'imreg_dft', 'mode': 'fixed', 'name': 'imreg_dft_固定'}
        )
    
    # すべての実験を実行
    all_results = {}
    for exp in experiments:
        print("\n\n" + "#" * 80)
        print(f"実験: {exp['name']}")
        print("#" * 80)
        
        output_folder = os.path.join(base_output, f"output_{exp['name']}")
        
        try:
            results = process_with_method(
                empty_channel_path,
                folder_path,
                output_folder,
                method=exp['method'],
                mode=exp['mode'],
                vmin=-0.1,
                vmax=1.7,
                **{k: v for k, v in exp.items() if k not in ['method', 'mode', 'name']}
            )
            all_results[exp['name']] = results
        except Exception as e:
            print(f"\n実験失敗: {e}")
            import traceback
            traceback.print_exc()
    
    # 全体サマリー
    print("\n\n" + "=" * 80)
    print("全実験完了サマリー")
    print("=" * 80)
    for name, results in all_results.items():
        if results:
            errors = [r['error'] for r in results]
            print(f"\n{name}:")
            print(f"  処理枚数: {len(results)}")
            print(f"  平均誤差: {np.mean(errors):.4f}")
            print(f"  誤差標準偏差: {np.std(errors):.4f}")
    
    print("\n処理が完了しました！各手法のフォルダを比較してください。")

# %%