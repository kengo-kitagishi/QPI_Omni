---
name: 固定範囲正規化の実装
overview: to_uint8関数を実際の位相画像値域（-5~2 rad）に基づく固定範囲クリッピングに変更します。
todos:
  - id: replace-function
    content: to_uint8関数を固定範囲版（-5.0~2.0）に置換
    status: completed
---

# 固定範囲正規化の実装

## 実際の位相画像値域

ユーザー確認済み: **-5 ~ 2 rad**

（19_gaussian_backsub.pyの-1.1~1.5はバックグラウンドピークの探索範囲であって画像全体の範囲ではない）

## 現在の問題

```24:32:c:\Users\QPI\Documents\QPI_omni\scripts\21_calc_alignment.py
def to_uint8(img):
    """uint8に変換（OpenCV用）"""
    img_min = np.min(img)
    img_max = np.max(img)
    if img_max - img_min > 0:
        normalized = (img - img_min) / (img_max - img_min)
    else:
        normalized = img
    return (normalized * 255).astype(np.uint8)
```

**問題点**:
- 各画像でmin/maxが異なる → 画像間の対応が取れない
- 外れ値1つで全体のスケールが変わる
- アライメント精度が悪化

## 新しい実装

```python
def to_uint8(img, vmin=-5.0, vmax=2.0):
    """
    固定範囲でuint8に変換（アライメント用）
    
    Parameters:
    -----------
    vmin, vmax : float
        クリッピング範囲（位相画像のrad値）
        デフォルト: -5.0 ~ 2.0 rad（実測値域）
    
    Returns:
    --------
    uint8 : 0-255の範囲に正規化された画像
    
    Note:
    -----
    - 全画像で一貫した変換を保証
    - 外れ値は自動的にクリッピング
    - ECCアルゴリズムには十分な精度
    """
    # クリッピング（外れ値除去）
    clipped = np.clip(img, vmin, vmax)
    
    # 0-255に正規化
    normalized = (clipped - vmin) / (vmax - vmin)
    
    return (normalized * 255).astype(np.uint8)
```

## 実装箇所

[`21_calc_alignment.py`](c:\Users\QPI\Documents\QPI_omni\scripts\21_calc_alignment.py)

### 変更箇所

**24-32行**: 関数定義を置き換え

既存の関数呼び出し（153行、198行付近）はそのまま：
```python
alignment_reference_uint8 = to_uint8(alignment_reference_img)
target_uint8 = to_uint8(target_img)
```

デフォルト引数（-5.0 ~ 2.0）が自動的に適用されます。

## 255の全範囲を使わないメリット

実際の値域が -5 ~ 2（7 rad幅）の場合：
- 0-255の全範囲を使用
- ただし**固定範囲**なので画像間で一貫
- 外れ値の影響を受けない
- ECCには十分な精度

## 期待される効果

1. **一貫性**: 全画像で同じピクセル値が同じuint8値に
2. **外れ値耐性**: 異常値が自動的にクリッピング
3. **精度向上**: 画像間の対応が正確に取れる
4. **安定性**: ノイズや外れ値の影響が小さい

## 実装手順

1. 24-32行の`to_uint8()`関数を新しい実装に置き換え
2. テスト実行
3. 相関値・シフト量を確認

変更は1箇所のみ、関数呼び出し側の変更は不要です。