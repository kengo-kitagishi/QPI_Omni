#!/usr/bin/env python3
#%%
"""
位相画像連番マージスクリプト

ph_2の位相画像ファイルを、ph_1の連番の続きとして自動でリネーム・コピーします。
各Posフォルダで自動的に最大連番を検出し、続きの番号でファイルをマージします。
"""

import re
import shutil
from pathlib import Path
from typing import List, Tuple, Optional


def find_pos_folders(base_path: Path) -> List[Path]:
    """
    指定されたパス配下の全Posフォルダを検出し、番号順にソートして返します。
    
    Args:
        base_path: 検索対象のベースパス
        
    Returns:
        ソート済みのPosフォルダのリスト
    """
    if not base_path.exists():
        print(f"エラー: パスが存在しません: {base_path}")
        return []
    
    pos_folders = sorted(
        base_path.glob("Pos*"),
        key=lambda p: int(p.name.replace("Pos", "")) if p.name.replace("Pos", "").isdigit() else 0
    )
    
    return [p for p in pos_folders if p.is_dir()]


def find_max_sequence_number(directory: Path, pattern: str = r"img_(\d+)_ph_000_phase\.tif") -> int:
    """
    指定されたディレクトリ内の最大連番を検出します。
    
    Args:
        directory: 検索対象のディレクトリ
        pattern: ファイル名のパターン（正規表現）
        
    Returns:
        最大連番（ファイルが存在しない場合は-1）
    """
    if not directory.exists():
        return -1
    
    max_num = -1
    pattern_re = re.compile(pattern)
    
    for file_path in directory.glob("img_*_ph_000_phase.tif"):
        match = pattern_re.match(file_path.name)
        if match:
            num = int(match.group(1))
            max_num = max(max_num, num)
    
    return max_num


def get_sorted_phase_files(directory: Path) -> List[Path]:
    """
    指定されたディレクトリ内の位相画像ファイルを連番順に取得します。
    
    Args:
        directory: 検索対象のディレクトリ
        
    Returns:
        ソート済みのファイルパスのリスト
    """
    if not directory.exists():
        return []
    
    files = list(directory.glob("img_*_ph_000_phase.tif"))
    
    # ファイル名から連番を抽出してソート
    def extract_number(path: Path) -> int:
        match = re.search(r"img_(\d+)_", path.name)
        return int(match.group(1)) if match else 0
    
    return sorted(files, key=extract_number)


def generate_new_filename(sequence_number: int) -> str:
    """
    連番から新しいファイル名を生成します。
    
    Args:
        sequence_number: 連番
        
    Returns:
        新しいファイル名
    """
    return f"img_{sequence_number:09d}_ph_000_phase.tif"


def merge_phase_sequences(ph1_dir: Path, ph2_dir: Path, pos_name: str) -> Tuple[int, int, List[str]]:
    """
    ph_2の位相画像ファイルをph_1に連番の続きとしてコピーします。
    
    Args:
        ph1_dir: ph_1のoutput_phaseディレクトリ
        ph2_dir: ph_2のoutput_phaseディレクトリ
        pos_name: Posフォルダ名（ログ用）
        
    Returns:
        (成功数, スキップ数, エラーメッセージリスト)
    """
    success_count = 0
    skip_count = 0
    errors = []
    
    try:
        # ph_1の最大連番を検出
        max_num_ph1 = find_max_sequence_number(ph1_dir)
        next_num = max_num_ph1 + 1
        
        print(f"  ph_1の最大連番: {max_num_ph1 if max_num_ph1 >= 0 else 'なし'}")
        print(f"  次の連番開始: {next_num}")
        
        # ph_2の全ファイルを取得
        ph2_files = get_sorted_phase_files(ph2_dir)
        
        if not ph2_files:
            print(f"  ph_2にファイルが見つかりません")
            return 0, 0, []
        
        print(f"  ph_2のファイル数: {len(ph2_files)}")
        
        # ph_1のディレクトリが存在しない場合は作成
        ph1_dir.mkdir(parents=True, exist_ok=True)
        
        # 各ファイルをコピー
        for i, source_file in enumerate(ph2_files):
            new_sequence = next_num + i
            new_filename = generate_new_filename(new_sequence)
            dest_file = ph1_dir / new_filename
            
            # 既存ファイルのチェック
            if dest_file.exists():
                errors.append(f"    ⚠ スキップ: {new_filename} は既に存在します")
                skip_count += 1
                continue
            
            try:
                # ファイルコピー
                shutil.copy2(source_file, dest_file)
                print(f"    ✓ {source_file.name} -> {new_filename}")
                success_count += 1
                
            except Exception as e:
                error_msg = f"    ✗ エラー: {source_file.name} のコピーに失敗 ({e})"
                errors.append(error_msg)
        
        return success_count, skip_count, errors
        
    except Exception as e:
        errors.append(f"  ✗ 処理エラー: {e}")
        return success_count, skip_count, errors


def main():
    """メイン処理"""
    print("=" * 80)
    print("位相画像連番マージスクリプト")
    print("=" * 80)
    print()
    
    # ベースパスの設定
    base_volume = Path(r"F:\251212")
    ph1_base = base_volume / "ph_1"
    ph2_base = base_volume / "ph_2"
    
    print(f"ph_1ベースパス: {ph1_base}")
    print(f"ph_2ベースパス: {ph2_base}")
    print()
    
    # ボリュームの存在確認
    if not base_volume.exists():
        print(f"エラー: ボリュームがマウントされていません: {base_volume}")
        return
    
    # Posフォルダの検出（ph_2を基準に）
    print("Posフォルダを検出中...")
    pos_folders_ph2 = find_pos_folders(ph2_base)
    
    if not pos_folders_ph2:
        print("エラー: ph_2にPosフォルダが見つかりませんでした")
        return
    
    print(f"検出されたPosフォルダ数: {len(pos_folders_ph2)}")
    print(f"範囲: {pos_folders_ph2[0].name} ～ {pos_folders_ph2[-1].name}")
    print()
    print("-" * 80)
    print()
    
    # 処理結果の記録
    total_success = 0
    total_skip = 0
    total_errors = 0
    pos_processed = 0
    
    # 各Posフォルダに対して処理
    for pos_folder_ph2 in pos_folders_ph2:
        pos_name = pos_folder_ph2.name
        print(f"[{pos_name}]")
        
        # ディレクトリパスの構築
        ph1_output = ph1_base / pos_name / "output_phase"
        ph2_output = ph2_folder_ph2 / "output_phase"
        
        # ph_2のディレクトリが存在しない場合はスキップ
        if not ph2_output.exists():
            print(f"  ⊘ スキップ: ph_2のoutput_phaseが存在しません")
            print()
            continue
        
        # マージ処理
        success, skip, errors = merge_phase_sequences(ph1_output, ph2_output, pos_name)
        
        # エラーメッセージの表示
        for error in errors:
            print(error)
        
        # 結果の集計
        total_success += success
        total_skip += skip
        total_errors += len(errors)
        
        if success > 0:
            pos_processed += 1
        
        print(f"  結果: 成功={success}, スキップ={skip}, エラー={len(errors)}")
        print()
    
    # 処理結果のサマリー
    print("-" * 80)
    print()
    print("処理結果サマリー:")
    print(f"  処理したPosフォルダ: {pos_processed}")
    print(f"  コピー成功: {total_success} ファイル")
    print(f"  スキップ: {total_skip} ファイル")
    print(f"  エラー: {total_errors} 件")
    print()
    print("=" * 80)
    
    if total_errors > 0:
        print("警告: 一部のファイルでエラーが発生しました")
    elif total_success > 0:
        print("すべてのファイルが正常にマージされました")
    else:
        print("マージされたファイルはありません")
    
    print("=" * 80)


if __name__ == "__main__":
    main()


# %%
