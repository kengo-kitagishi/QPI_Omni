---
name: 22_ecc_alignmentの精度保存修正
overview: 22_ecc_alignment.pyでも同じ問題を修正：固定範囲正規化と元画像への直接シフト適用
todos:
  - id: 22-fix-to-uint8
    content: "22_ecc: to_uint8関数を固定範囲版に変更"
    status: pending
  - id: 22-fix-warpaffine
    content: "22_ecc: warpAffineを元画像に直接適用"
    status: pending
---

# 22_ecc_alignment.pyの精度保存修正

## 問題

22_ecc_alignment.pyでも21_calc_alignment.pyと**同じ問題**があります：

### 現在の処理（216-232行）
```python
# 画像読み込み
img = load_tif_image(str(target_path))
img_uint8 = to_uint8(img)  # 画像ごとのmin-max → 256段階に圧縮

# 変換行列取得
warp_matrix = np.array(transform_data['warp_matrix'], dtype=np.float32)

# uint8画像を変換
aligned_uint8 = cv2.warpAffine(
    img_uint8, warp_matrix, (w, h),  # uint8を移動
    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
)

# float64に戻す
aligned_img = aligned_uint8.astype(np.float64) / 255.0
aligned_img = aligned_img * (np.max(img) - np.min(img)) + np.min(img)
```

**問題点**:
1. `to_uint8()`が画像ごとのmin-max正規化 → 画像間で不整合
2. uint8画像（256段階）を移動 → 精度が失われる
3. float64に戻すが実質8bit

## 修正内容

### 1. to_uint8関数を固定範囲版に変更（21-29行）

#### 現在
```python
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

#### 修正後
```python
def to_uint8(img, vmin=-5.0, vmax=2.0):
    """
    固定範囲でuint8に変換（アライメント用）
    
    Parameters:
    -----------
    vmin, vmax : float
        クリッピング範囲（位相画像のrad値）
        デフォルト: -5.0 ~ 2.0 rad（実測値域）
    """
    # クリッピング（外れ値除去）
    clipped = np.clip(img, vmin, vmax)
    
    # 0-255に正規化
    normalized = (clipped - vmin) / (vmax - vmin)
    
    return (normalized * 255).astype(np.uint8)
```

### 2. warpAffineを元画像に直接適用（223-232行）

#### 現在
```python
# 変換適用
h, w = img.shape
aligned_uint8 = cv2.warpAffine(
    img_uint8, warp_matrix, (w, h),  # uint8を移動
    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
)

# float64に戻す
aligned_img = aligned_uint8.astype(np.float64) / 255.0
aligned_img = aligned_img * (np.max(img) - np.min(img)) + np.min(img)
```

#### 修正後
```python
# 変換適用（元画像を直接移動）
h, w = img.shape
aligned_img = cv2.warpAffine(
    img.astype(np.float32),  # 元画像を直接移動
    warp_matrix,
    (w, h),
    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
).astype(np.float64)
```

**注意**: 218行の`img_uint8 = to_uint8(img)`は削除可能ですが、将来的に必要になる可能性があるので残しておいても良い

## 実装箇所

[`22_ecc_alignment.py`](c:\Users\QPI\Documents\QPI_omni\scripts\22_ecc_alignment.py)

1. **21-29行**: `to_uint8()`関数を固定範囲版に変更
2. **223-232行**: warpAffineを元画像に直接適用するように変更

## メリット

1. ✅ **精度保存**: 元のbg_corr画像の精度が完全に保存
2. ✅ **一貫性**: 21_calc_alignment.pyと同じ処理
3. ✅ **高精度差分**: subtracted画像の精度が向上
4. ✅ **シンプル**: 余計な変換処理を削除

## 期待される効果

- aligned画像: float64の完全な精度（実質8bitから脱却）
- 微細な位相変化の検出が可能に
- これまでの実験データとの違いが明確に

## 注意

22_ecc_alignment.pyはステップ2なので、21_calc_alignment.pyで保存されたJSONファイルの変換行列を使用します。21の修正後に実行してください。
