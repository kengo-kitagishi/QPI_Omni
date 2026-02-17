#!/usr/bin/env python3
"""
位相画像ファイル一括コピースクリプト

/Volumes/QPI_0_.01_r/251212/wo_cell/ph_2/Pos*/output_phase/img_000000000_ph_021_phase.tif を
/Volumes/QPI_0_.01_r/251212/ph_1/Pos*/output_phase/img_000000000_ph_000_phase.tif に上書きコピーします。
"""

import shutil
from pathlib import Path
from typing import List, Tuple


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


def copy_phase_file(source_path: Path, dest_path: Path) -> Tuple[bool, str]:
    """
    位相ファイルをコピーします。
    
    Args:
        source_path: コピー元ファイルパス
        dest_path: コピー先ファイルパス
        
    Returns:
        (成功フラグ, メッセージ) のタプル
    """
    try:
        # ソースファイルの存在確認
        if not source_path.exists():
            return False, f"ソースファイルが存在しません: {source_path}"
        
        # コピー先ディレクトリが存在しない場合は作成
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # ファイルコピー（メタデータも保持）
        shutil.copy2(source_path, dest_path)
        
        return True, f"コピー成功: {source_path.name} -> {dest_path}"
        
    except PermissionError as e:
        return False, f"権限エラー: {e}"
    except Exception as e:
        return False, f"エラー: {e}"


def main():
    """メイン処理"""
    print("=" * 80)
    print("位相画像ファイル一括コピースクリプト")
    print("=" * 80)
    print()
    
    # ベースパスの設定
    base_volume = Path(r"F:\251212")
    source_base = base_volume / "wo_cell" / "ph_2"
    dest_base = base_volume / "ph_1"
    
    # ファイル名の設定
    source_filename = "img_000000000_ph_021_phase.tif"
    dest_filename = "img_000000000_ph_000_phase.tif"
    
    print(f"ソースベースパス: {source_base}")
    print(f"コピー先ベースパス: {dest_base}")
    print()
    
    # ボリュームの存在確認
    if not base_volume.exists():
        print(f"エラー: ボリュームがマウントされていません: {base_volume}")
        return
    
    # Posフォルダの検出
    print("Posフォルダを検出中...")
    pos_folders = find_pos_folders(source_base)
    
    if not pos_folders:
        print("エラー: Posフォルダが見つかりませんでした")
        return
    
    print(f"検出されたPosフォルダ数: {len(pos_folders)}")
    print(f"範囲: {pos_folders[0].name} ～ {pos_folders[-1].name}")
    print()
    print("-" * 80)
    print()
    
    # 処理結果の記録
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    # 各Posフォルダに対して処理
    for pos_folder in pos_folders:
        pos_name = pos_folder.name
        
        # ソースファイルパスの構築
        source_file = pos_folder / "output_phase" / source_filename
        
        # コピー先ファイルパスの構築
        dest_file = dest_base / pos_name / "output_phase" / dest_filename
        
        print(f"[{pos_name}]", end=" ")
        
        # ファイルコピー
        success, message = copy_phase_file(source_file, dest_file)
        
        if success:
            print(f"✓ {message}")
            success_count += 1
        else:
            if "存在しません" in message:
                print(f"⊘ スキップ: {message}")
                skip_count += 1
            else:
                print(f"✗ 失敗: {message}")
                fail_count += 1
    
    # 処理結果のサマリー
    print()
    print("-" * 80)
    print()
    print("処理結果サマリー:")
    print(f"  成功: {success_count} ファイル")
    print(f"  失敗: {fail_count} ファイル")
    print(f"  スキップ: {skip_count} ファイル")
    print(f"  合計: {len(pos_folders)} フォルダ")
    print()
    print("=" * 80)
    
    if fail_count > 0:
        print("警告: 一部のファイルでエラーが発生しました")
    elif success_count > 0:
        print("すべてのファイルが正常にコピーされました")
    else:
        print("コピーされたファイルはありません")
    
    print("=" * 80)


if __name__ == "__main__":
    main()

