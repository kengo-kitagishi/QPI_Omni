# %%
"""
新フォルダ構造対応の引き算処理スクリプト
アライメント済みwo_2画像から元のwo_0画像を引き算
"""
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import os
from pathlib import Path
from tqdm import tqdm

# 日本語フォント設定
plt.rcParams['font.family'] = 'MS Gothic'  # Windowsの場合
plt.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け対策


def load_tif_image(path):
    """TIF画像を読み込んでfloatに変換"""
    img = io.imread(path)
    return img.astype(np.float64)


def get_image_number(filename):
    """
    ファイル名から画像番号（末尾3桁）を抽出
    例: img_000000000_Default_005_phase.tif -> 5
    """
    try:
        # _phase.tif を除去
        base = filename.replace("_phase.tif", "").replace("_phase.tiff", "")
        # 最後のアンダースコア以降の数字を取得
        parts = base.split("_")
        return int(parts[-1])
    except:
        return None


def find_matching_pos_folders(wo0_base, wo2_base):
    """
    wo_0とwo_2で対応するPosフォルダのペアを見つける
    
    Returns:
    --------
    list of tuples: [(pos_name, wo0_path, wo2_path), ...]
    """
    pos_pairs = []
    
    # wo_0のPosフォルダを取得（Pos0を除く）
    wo0_pos_folders = []
    for item in sorted(os.listdir(wo0_base)):
        item_path = os.path.join(wo0_base, item)
        if os.path.isdir(item_path) and item.startswith("Pos") and item != "Pos0":
            wo0_pos_folders.append(item)
    
    # 各Posに対応するwo_2のフォルダがあるか確認
    for pos_name in wo0_pos_folders:
        wo0_pos_path = os.path.join(wo0_base, pos_name)
        wo2_pos_path = os.path.join(wo2_base, pos_name)
        
        # 両方の output_phase フォルダが存在するか確認
        wo0_phase_dir = os.path.join(wo0_pos_path, "output_phase")
        wo2_phase_dir = os.path.join(wo2_pos_path, "output_phase")
        wo2_aligned_dir = os.path.join(wo2_phase_dir, "aligned")
        
        if os.path.exists(wo0_phase_dir) and os.path.exists(wo2_aligned_dir):
            pos_pairs.append((pos_name, wo0_phase_dir, wo2_aligned_dir, wo2_phase_dir))
    
    return pos_pairs


def process_pos_subtraction(pos_name, wo0_phase_dir, wo2_aligned_dir, wo2_phase_dir,
                            save_png=True, vmin=-0.1, vmax=1.7, cmap='RdBu_r', 
                            png_dpi=150, png_sample_interval=1):
    """
    1つのPosペアについて引き算処理を行う
    
    Parameters:
    -----------
    pos_name : str
        Posフォルダ名（例: "Pos1"）
    wo0_phase_dir : str
        wo_0の位相画像ディレクトリ
    wo2_aligned_dir : str
        wo_2のアライメント済み位相画像ディレクトリ
    wo2_phase_dir : str
        wo_2の位相画像ディレクトリ（出力先として使用）
    save_png : bool
        カラーマップPNG画像を保存するか
    vmin, vmax : float
        カラーマップの範囲
    cmap : str
        カラーマップの種類
    png_dpi : int
        PNG保存時の解像度
    png_sample_interval : int
        N枚ごとにPNG保存（1=全部、10=10枚に1枚）
    
    Returns:
    --------
    int: 処理された画像数
    """
    print(f"\n{'='*80}")
    print(f"処理中: {pos_name}")
    print(f"{'='*80}")
    
    # 出力ディレクトリ作成
    subtracted_dir = os.path.join(wo2_phase_dir, "subtracted")
    os.makedirs(subtracted_dir, exist_ok=True)
    
    if save_png:
        colored_dir = os.path.join(wo2_phase_dir, "subtracted_colored")
        os.makedirs(colored_dir, exist_ok=True)
    
    # wo_2のアライメント済み画像を取得
    wo2_files = sorted([f for f in os.listdir(wo2_aligned_dir) 
                        if f.endswith("_phase.tif") or f.endswith("_phase.tiff")])
    
    if len(wo2_files) == 0:
        print(f"  ⚠️ 警告: アライメント済み画像が見つかりません")
        return 0
    
    print(f"  対象画像数: {len(wo2_files)}枚")
    
    # 画像番号でマッピング
    wo0_file_map = {}
    for filename in os.listdir(wo0_phase_dir):
        if filename.endswith("_phase.tif") or filename.endswith("_phase.tiff"):
            img_number = get_image_number(filename)
            if img_number is not None:
                wo0_file_map[img_number] = filename
    
    # 引き算処理
    processed_count = 0
    skipped_count = 0
    png_saved_count = 0
    
    for wo2_filename in tqdm(wo2_files, desc=f"    {pos_name}"):
        try:
            # 画像番号を取得
            img_number = get_image_number(wo2_filename)
            
            if img_number is None:
                print(f"\n    ⚠️ 警告: 画像番号を取得できません: {wo2_filename}")
                skipped_count += 1
                continue
            
            # 対応するwo_0の画像を探す
            if img_number not in wo0_file_map:
                if skipped_count < 3:  # 最初の数枚だけ警告表示
                    print(f"\n    ⚠️ 警告: wo_0に対応する画像が見つかりません（番号: {img_number}）")
                skipped_count += 1
                continue
            
            wo0_filename = wo0_file_map[img_number]
            
            # 画像読み込み
            wo2_path = os.path.join(wo2_aligned_dir, wo2_filename)
            wo0_path = os.path.join(wo0_phase_dir, wo0_filename)
            
            wo2_img = load_tif_image(wo2_path)
            wo0_img = load_tif_image(wo0_path)
            
            # サイズチェック
            if wo2_img.shape != wo0_img.shape:
                print(f"\n    ⚠️ 警告: 画像サイズが一致しません: {wo2_filename}")
                skipped_count += 1
                continue
            
            # 引き算: wo_2 - wo_0
            subtracted = wo2_img - wo0_img
            
            # TIF保存
            base_name = wo2_filename.replace("_phase.tif", "").replace("_phase.tiff", "")
            subtracted_path = os.path.join(subtracted_dir, f"{base_name}_subtracted.tif")
            io.imsave(subtracted_path, subtracted.astype(np.float32))
            
            processed_count += 1
            
            # PNG保存（オプション）
            if save_png and (processed_count % png_sample_interval == 0):
                colored_path = os.path.join(colored_dir, f"{base_name}_subtracted.png")
                
                fig, ax = plt.subplots(figsize=(10, 8))
                norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
                im = ax.imshow(subtracted, cmap=cmap, norm=norm)
                ax.axis('off')
                ax.set_title(f'{pos_name} - {base_name}\n平均: {np.mean(subtracted):.3f}, 標準偏差: {np.std(subtracted):.3f}')
                plt.colorbar(im, ax=ax, fraction=0.046, label='差分 (rad)')
                plt.tight_layout()
                plt.savefig(colored_path, dpi=png_dpi, bbox_inches='tight')
                plt.close()
                png_saved_count += 1
        
        except Exception as e:
            print(f"\n    ❌ エラー: {wo2_filename} - {e}")
            import traceback
            traceback.print_exc()
            skipped_count += 1
            continue
    
    # サマリー
    print(f"\n  ✅ 処理完了")
    print(f"     成功: {processed_count}枚")
    if skipped_count > 0:
        print(f"     スキップ: {skipped_count}枚")
    print(f"     保存先（TIF）: {subtracted_dir}")
    if save_png:
        print(f"     保存先（PNG）: {colored_dir} ({png_saved_count}枚)")
    
    return processed_count


def main():
    """メイン処理"""
    print("="*80)
    print("新フォルダ構造対応の引き算処理")
    print("="*80)
    
    # ベースディレクトリ設定
    WO0_BASE = r"F:\wo_0_EMM_1"
    WO2_BASE = r"F:\wo_2_EMM_1"
    
    print(f"\nwo_0: {WO0_BASE}")
    print(f"wo_2: {WO2_BASE}")
    
    # ディレクトリ存在確認
    if not os.path.exists(WO0_BASE):
        print(f"\n❌ エラー: wo_0ディレクトリが見つかりません: {WO0_BASE}")
        return
    
    if not os.path.exists(WO2_BASE):
        print(f"\n❌ エラー: wo_2ディレクトリが見つかりません: {WO2_BASE}")
        return
    
    # 対応するPosフォルダのペアを探す
    print("\n対応するPosフォルダを検索中...")
    pos_pairs = find_matching_pos_folders(WO0_BASE, WO2_BASE)
    
    if len(pos_pairs) == 0:
        print("\n❌ エラー: 対応するPosフォルダが見つかりません")
        print("   アライメント処理（calc_alignment_new.py）を先に実行してください。")
        return
    
    print(f"検出されたPosペア: {len(pos_pairs)}個")
    for pos_name, _, _, _ in pos_pairs:
        print(f"  - {pos_name}")
    
    # 処理設定
    SAVE_PNG = True  # PNG保存するか（False=高速、True=可視化あり）
    PNG_DPI = 150    # PNG解像度（150=軽い、300=重い）
    PNG_SAMPLE_INTERVAL = 1  # N枚ごとにPNG保存（1=全部、10=10枚に1枚）
    VMIN = -0.1  # カラーマップの最小値
    VMAX = 1.7   # カラーマップの最大値
    CMAP = 'RdBu_r'  # カラーマップ
    
    print(f"\n処理設定:")
    print(f"  PNG保存: {SAVE_PNG}")
    if SAVE_PNG:
        print(f"  PNG解像度: {PNG_DPI} dpi")
        print(f"  サンプリング間隔: {PNG_SAMPLE_INTERVAL}枚ごと")
        print(f"  カラーマップ範囲: [{VMIN}, {VMAX}]")
        print(f"  カラーマップ: {CMAP}")
    
    # 各Posペアを処理
    total_processed = 0
    
    for pos_name, wo0_phase_dir, wo2_aligned_dir, wo2_phase_dir in pos_pairs:
        processed = process_pos_subtraction(
            pos_name, wo0_phase_dir, wo2_aligned_dir, wo2_phase_dir,
            save_png=SAVE_PNG,
            vmin=VMIN,
            vmax=VMAX,
            cmap=CMAP,
            png_dpi=PNG_DPI,
            png_sample_interval=PNG_SAMPLE_INTERVAL
        )
        total_processed += processed
    
    # 全体サマリー
    print("\n" + "="*80)
    print("引き算処理完了")
    print("="*80)
    print(f"\n処理されたPos数: {len(pos_pairs)}")
    print(f"処理された画像総数: {total_processed}枚")


if __name__ == "__main__":
    main()

# %%

