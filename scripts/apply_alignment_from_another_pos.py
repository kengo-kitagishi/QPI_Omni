# %%
"""
他のPosのアライメント情報を適用するスクリプト
特定のPosで計算されたアライメント行列を別のPosに適用
"""
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import os
import json
import cv2
from tqdm import tqdm

# 日本語フォント設定
plt.rcParams['font.family'] = 'MS Gothic'
plt.rcParams['axes.unicode_minus'] = False


def load_tif_image(path):
    """TIF画像を読み込んでfloatに変換"""
    img = io.imread(path)
    return img.astype(np.float64)


def get_image_number(filename):
    """ファイル名から画像番号を抽出"""
    try:
        base = filename.replace("_phase.tif", "").replace("_phase.tiff", "")
        parts = base.split("_")
        return int(parts[-1])
    except:
        return None


def apply_alignment_from_reference(source_pos, target_pos, wo0_base, wo2_base,
                                   save_png=True, vmin=-0.1, vmax=1.7, cmap='RdBu_r',
                                   png_dpi=150, png_sample_interval=1):
    """
    source_posのアライメント情報をtarget_posに適用
    
    Parameters:
    -----------
    source_pos : str
        アライメント情報の参照元Pos（例: "Pos13"）
    target_pos : str
        アライメント適用先Pos（例: "Pos1"）
    wo0_base : str
        wo_0のベースディレクトリ
    wo2_base : str
        wo_2のベースディレクトリ
    save_png : bool
        カラーマップPNG保存
    vmin, vmax : float
        カラーマップ範囲
    cmap : str
        カラーマップ
    png_dpi : int
        PNG解像度
    png_sample_interval : int
        サンプリング間隔
    """
    print("="*80)
    print(f"{source_pos}のアライメント → {target_pos}に適用")
    print("="*80)
    
    # パス設定
    source_json_path = os.path.join(wo2_base, source_pos, "output_phase", "alignment_transform.json")
    wo0_target_phase_dir = os.path.join(wo0_base, target_pos, "output_phase")
    wo2_target_phase_dir = os.path.join(wo2_base, target_pos, "output_phase")
    
    # source_posのアライメント情報を読み込み
    print(f"\n[1/3] {source_pos}のアライメント情報を読み込み中...")
    if not os.path.exists(source_json_path):
        print(f"❌ エラー: アライメント情報が見つかりません: {source_json_path}")
        return None
    
    with open(source_json_path, 'r', encoding='utf-8') as f:
        alignment_info = json.load(f)
    
    warp_matrix = np.array(alignment_info['warp_matrix'], dtype=np.float32)
    shift_y = alignment_info['shift_y']
    shift_x = alignment_info['shift_x']
    correlation = alignment_info['correlation']
    
    print(f"  ✅ アライメント情報読み込み完了")
    print(f"     元Pos: {source_pos}")
    print(f"     シフト量: Y={shift_y:.2f}px, X={shift_x:.2f}px")
    print(f"     相関係数: {correlation:.4f}")
    
    # target_posのディレクトリ確認
    if not os.path.exists(wo0_target_phase_dir):
        print(f"❌ エラー: wo_0のディレクトリが見つかりません: {wo0_target_phase_dir}")
        return None
    
    if not os.path.exists(wo2_target_phase_dir):
        print(f"❌ エラー: wo_2のディレクトリが見つかりません: {wo2_target_phase_dir}")
        return None
    
    # 出力ディレクトリ作成
    aligned_dir = os.path.join(wo2_target_phase_dir, f"aligned_from_{source_pos}")
    subtracted_dir = os.path.join(wo2_target_phase_dir, f"subtracted_from_{source_pos}")
    os.makedirs(aligned_dir, exist_ok=True)
    os.makedirs(subtracted_dir, exist_ok=True)
    
    if save_png:
        colored_dir = os.path.join(wo2_target_phase_dir, f"subtracted_colored_from_{source_pos}")
        os.makedirs(colored_dir, exist_ok=True)
    
    print(f"\n[2/3] {target_pos}の画像にアライメント適用 + 引き算中...")
    
    # target_posの画像リストを取得
    wo0_files = sorted([f for f in os.listdir(wo0_target_phase_dir) 
                        if not f.startswith("._") and (f.endswith("_phase.tif") or f.endswith("_phase.tiff"))])
    wo2_files = sorted([f for f in os.listdir(wo2_target_phase_dir) 
                        if not f.startswith("._") and (f.endswith("_phase.tif") or f.endswith("_phase.tiff"))])
    
    if len(wo0_files) == 0 or len(wo2_files) == 0:
        print(f"❌ エラー: 位相画像が見つかりません")
        return None
    
    print(f"  wo_0画像数: {len(wo0_files)}枚")
    print(f"  wo_2画像数: {len(wo2_files)}枚")
    
    # 画像番号でマッピング
    wo0_file_map = {}
    for filename in wo0_files:
        img_number = get_image_number(filename)
        if img_number is not None:
            wo0_file_map[img_number] = filename
    
    # 処理カウンタ
    processed_count = 0
    skipped_count = 0
    png_saved_count = 0
    
    # 各wo_2画像を処理
    for wo2_filename in tqdm(wo2_files, desc=f"    {target_pos}"):
        try:
            img_number = get_image_number(wo2_filename)
            
            if img_number is None:
                skipped_count += 1
                continue
            
            if img_number not in wo0_file_map:
                if skipped_count < 3:
                    print(f"\n    ⚠️ 警告: wo_0に対応する画像が見つかりません（番号: {img_number}）")
                skipped_count += 1
                continue
            
            wo0_filename = wo0_file_map[img_number]
            
            # wo_2画像を読み込み
            wo2_path = os.path.join(wo2_target_phase_dir, wo2_filename)
            wo2_img = load_tif_image(wo2_path)
            
            # アライメント適用
            h, w = wo2_img.shape
            aligned_img = cv2.warpAffine(
                wo2_img.astype(np.float32),
                warp_matrix,
                (w, h),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            ).astype(np.float64)
            
            # アライメント済み画像を保存
            aligned_path = os.path.join(aligned_dir, wo2_filename)
            io.imsave(aligned_path, aligned_img.astype(np.float32))
            
            # wo_0画像を読み込み
            wo0_path = os.path.join(wo0_target_phase_dir, wo0_filename)
            wo0_img = load_tif_image(wo0_path)
            
            # サイズチェック
            if aligned_img.shape != wo0_img.shape:
                print(f"\n    ⚠️ 警告: 画像サイズが一致しません: {wo2_filename}")
                skipped_count += 1
                continue
            
            # 引き算
            subtracted = aligned_img - wo0_img
            
            # TIF保存
            base_name = wo2_filename.replace("_phase.tif", "").replace("_phase.tiff", "")
            subtracted_path = os.path.join(subtracted_dir, f"{base_name}_subtracted.tif")
            io.imsave(subtracted_path, subtracted.astype(np.float32))
            
            processed_count += 1
            
            # PNG保存
            if save_png and (processed_count % png_sample_interval == 0):
                colored_path = os.path.join(colored_dir, f"{base_name}_subtracted.png")
                
                fig, ax = plt.subplots(figsize=(10, 8))
                norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
                im = ax.imshow(subtracted, cmap=cmap, norm=norm)
                ax.axis('off')
                ax.set_title(f'{target_pos} (align from {source_pos}) - {base_name}\n平均: {np.mean(subtracted):.3f}, 標準偏差: {np.std(subtracted):.3f}')
                plt.colorbar(im, ax=ax, fraction=0.046, label='差分 (rad)')
                plt.tight_layout()
                plt.savefig(colored_path, dpi=png_dpi, bbox_inches='tight')
                plt.close()
                png_saved_count += 1
        
        except Exception as e:
            print(f"\n    ❌ エラー: {wo2_filename} - {e}")
            skipped_count += 1
            continue
    
    # JSON保存（適用情報を記録）
    print(f"\n[3/3] 結果保存中...")
    
    applied_info = {
        'source_pos': source_pos,
        'target_pos': target_pos,
        'source_alignment_info': alignment_info,
        'processed_count': processed_count,
        'skipped_count': skipped_count
    }
    
    json_path = os.path.join(wo2_target_phase_dir, f"alignment_applied_from_{source_pos}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(applied_info, f, indent=2, ensure_ascii=False)
    
    # サマリー
    print(f"\n  ✅ 処理完了")
    print(f"     成功: {processed_count}枚")
    if skipped_count > 0:
        print(f"     スキップ: {skipped_count}枚")
    print(f"     保存先（アライメント済み）: {aligned_dir}")
    print(f"     保存先（引き算TIF）: {subtracted_dir}")
    if save_png:
        print(f"     保存先（引き算PNG）: {colored_dir} ({png_saved_count}枚)")
    print(f"     JSON: {json_path}")
    
    return applied_info


def main():
    """メイン処理"""
    # ========================================
    # 設定
    # ========================================
    SOURCE_POS = "Pos13"  # アライメント情報の参照元
    TARGET_POS = "Pos1"   # アライメント適用先
    
    WO0_BASE = r"F:\wo_0_EMM_1"
    WO2_BASE = r"F:\wo_2_EMM_1"
    
    SAVE_PNG = True
    PNG_DPI = 150
    PNG_SAMPLE_INTERVAL = 1
    VMIN = -0.1
    VMAX = 1.7
    CMAP = 'RdBu_r'
    # ========================================
    
    print("="*80)
    print("他のPosのアライメントを適用")
    print("="*80)
    print(f"\nアライメント参照元: {SOURCE_POS}")
    print(f"適用先: {TARGET_POS}")
    print(f"\nwo_0: {WO0_BASE}")
    print(f"wo_2: {WO2_BASE}")
    
    result = apply_alignment_from_reference(
        SOURCE_POS, TARGET_POS, WO0_BASE, WO2_BASE,
        save_png=SAVE_PNG,
        vmin=VMIN,
        vmax=VMAX,
        cmap=CMAP,
        png_dpi=PNG_DPI,
        png_sample_interval=PNG_SAMPLE_INTERVAL
    )
    
    if result is not None:
        print("\n" + "="*80)
        print("処理完了！")
        print("="*80)
        print(f"\n{SOURCE_POS}のアライメント情報を{TARGET_POS}に適用しました。")
        print(f"結果は以下のフォルダに保存されています：")
        print(f"  {WO2_BASE}\\{TARGET_POS}\\output_phase\\")
        print(f"    - aligned_from_{SOURCE_POS}\\")
        print(f"    - subtracted_from_{SOURCE_POS}\\")
        if SAVE_PNG:
            print(f"    - subtracted_colored_from_{SOURCE_POS}\\")
    else:
        print("\n❌ エラーが発生しました")


if __name__ == "__main__":
    main()

# %%

