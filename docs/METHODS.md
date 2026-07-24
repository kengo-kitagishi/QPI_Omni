# QPI解析手法・アルゴリズム理論

このドキュメントは、QPI解析で使用される手法・アルゴリズムの理論的背景と実装の詳細をまとめたものです。

---

## 目次

1. [QPI基礎理論](#1-qpi基礎理論)
2. [体積推定手法の比較](#2-体積推定手法の比較)
3. [Pomegranate 3D再構成アルゴリズム](#3-pomegranate-3d再構成アルゴリズム)
4. [回転対称体積推定アルゴリズム](#4-回転対称体積推定アルゴリズム)
5. [楕円・Feret径近似](#5-楕円feret径近似)
6. [精度向上テクニック](#6-精度向上テクニック)

---

## 1. QPI基礎理論

### 1.1 位相差と光路長差

光が物質を透過すると、物質の屈折率 n に応じて位相が変化する。真空（または空気、n₀ ≈ 1）中の光と比較した位相差 φ は以下で表される：

```
φ = (2π / λ) × OPD
```

ここで：
- φ: 位相差 [radians]
- λ: 光の波長 [m]
- OPD: 光路長差（Optical Path Difference）[m]

### 1.2 光路長差と屈折率

サンプルの厚さを d、屈折率を n_sample、周囲媒質（培地）の屈折率を n_medium とすると：

```
OPD = (n_sample - n_medium) × d
```

これより：

```
φ = (2π / λ) × (n_sample - n_medium) × d
```

### 1.3 屈折率（RI）の計算

上式を n_sample について解くと：

```
n_sample = n_medium + (φ × λ) / (2π × d)
```

**記号の説明**：
- n_sample: サンプル（細胞）の屈折率 [無次元]
- n_medium: 培地の屈折率 [無次元]（デフォルト: 1.333）
- φ: 位相差 [radians]
- λ: レーザー波長 [m]（663 nm = 663×10⁻⁹ m）
- d: サンプルの厚み [m]

**実装**:
```python
# パラメータ
WAVELENGTH_NM = 663           # レーザー波長 [nm]
N_MEDIUM = 1.333              # 培地の屈折率
lambda_um = WAVELENGTH_NM / 1000  # 波長 [µm]

# RI計算
ri_sample = N_MEDIUM + (phase_radians * lambda_um) / (2 * np.pi * thickness_um)
```

### 1.4 質量濃度の計算

細胞内のタンパク質濃度 C [mg/ml] と屈折率 n の間には、以下の線形関係が成り立つ：

```
n - n_medium = α × C
```

ここで α は**比屈折率増分**（Specific Refractive Index Increment）[ml/mg]。

上式を C について解くと：

```
C = (n - n_medium) / α
```

**パラメータ**：
- C: 質量濃度 [mg/ml]
- n: サンプルの屈折率 [無次元]
- n_medium: 培地の屈折率 [無次元]
- α: 比屈折率増分 [ml/mg]（デフォルト: 0.00018 ml/mg）

**実装**:
```python
ALPHA_RI = 0.00018  # 比屈折率増分 [ml/mg]

# 質量濃度計算
concentration_mg_ml = (ri_map - N_MEDIUM) / ALPHA_RI
```

### 1.5 Total Massの計算

細胞の総質量は、各ピクセルの質量濃度とそのピクセルの体積の積を全体で積算：

```
M_total = Σ C(x, y) × V(x, y)
```

単位換算：
```
1 mg/ml = 1 mg/cm³ = 1 g/L = 10⁶ µg/cm³ = 1 pg/µm³
```

よって：
```
M_total [pg] = Σ C [mg/ml] × V [µm³]
```

**実装**:
```python
# 各ピクセルの体積
pixel_volumes = thickness_um[mask] * pixel_area_um2  # [µm³]

# Total mass計算
total_mass_pg = np.sum(concentration_map[mask] * pixel_volumes)  # [pg]
```

### 1.6 使用した仮定

#### 光学的仮定

1. **薄い物体近似（Thin Object Approximation）**
   - 光がサンプルを直進透過する
   - 回折・散乱の影響は無視

2. **均質な培地**
   - 培地の屈折率 n_medium は一定（1.333）
   - 温度・濃度変化は無視

3. **線形応答**
   - 位相差と光路長差が線形関係
   - 高次の非線形効果は無視

#### 生物学的仮定

1. **比屈折率増分の一定性**
   - 細胞内のすべての領域で α = 0.00018 ml/mg
   - タンパク質組成の違いは無視

2. **2D投影からの3D再構成**
   - 楕円近似、Feret径近似、回転対称などのモデルを使用
   - 実際の形状からのずれは誤差要因

### 1.7 パラメータ一覧

| パラメータ | 記号 | 値 | 単位 | 説明 |
|-----------|------|-----|------|------|
| レーザー波長 | λ | 658 | nm | オフアクシスホログラフィー用光源 |
| 培地屈折率 | n_medium | 1.333 | - | 無指定時のfallback（本番は校正値 wo_2=1.335029 / wo_0=1.332744 を frame ごとに参照） |
| ピクセルサイズ | - | 0.346 | µm | 511×511再構成画像のピクセルサイズ（0.08625×2048/511） |
| 比屈折率増分 | α | 0.00018 | ml/mg | タンパク質の標準値 |

---

## 2. 体積推定手法の比較

### 2.1 手法の概要

| 手法 | 原理 | 特徴 | 精度 | 速度 |
|------|------|------|------|------|
| **楕円近似** | ROIを楕円近似し、回転楕円体として体積計算 | シンプル | ★★★☆☆ | ★★★★★ |
| **Feret径近似** | Feret径で形状を近似 | 細長い細胞に強い | ★★★★☆ | ★★★★☆ |
| **Pomegranate** | Distance Transform + 球体展開 | 複雑な形状に対応 | ★★★★☆ | ★★★☆☆ |
| **回転対称** | 反復的中心線・断面線更新 | 論文準拠、高精度 | ★★★★★ | ★★☆☆☆ |

### 2.2 適用場面

**楕円近似**:
- ✅ 楕円形に近い細胞
- ✅ 高速処理が必要な場合
- ❌ 細長い細胞、不規則な形状

**Feret径近似**:
- ✅ 細長い細胞（分裂酵母など）
- ✅ 楕円近似で精度が低い場合
- ❌ 複雑な形状

**Pomegranate**:
- ✅ 複雑な形状（分岐、不規則）
- ✅ 2D画像から3D再構成
- ❌ XY/Z解像度が大きく異なる場合

**回転対称**:
- ✅ 回転対称に近い細胞
- ✅ 論文準拠の解析が必要な場合
- ✅ 高精度が必要な場合
- ❌ 高速処理が必要な場合

---

## 3. Pomegranate 3D再構成アルゴリズム

### 3.1 アルゴリズムの核心

Pomegranateは、**2Dセグメンテーション画像から3D形状を再構成**するアルゴリズムです。Distance Transform、Skeleton化、球体断面計算を組み合わせた手法により、複雑な細胞形状を3Dで復元します。

### 3.2 アルゴリズムの4つのステップ

#### Step 1: Distance Transform (距離変換)

各前景ピクセルから**最も近い背景ピクセルまでの距離**を計算します。

```
元のバイナリ画像:
█ █ █ █ █
█ ░ ░ ░ █
█ ░ ░ ░ █
█ ░ ░ ░ █
█ █ █ █ █

Distance Map (数値は距離):
█ █ █ █ █
█ 1 1 1 █
█ 1 2 1 █
█ 1 1 1 █
█ █ █ █ █
```

**意味**:
- Distance値 = その位置での「局所的な半径」
- 物体の中心に近いほど値が大きい
- 細い部分は小さく、太い部分は大きくなる

**数式**:
```
D(p) = min{||p - q|| : q ∈ Background}
```

**実装**:
```python
from scipy import ndimage
distance_map = ndimage.distance_transform_edt(binary_image)
```

#### Step 2: Skeleton (骨格化)

物体を**1ピクセル幅の中心線**まで細線化します。

```
元のバイナリ画像:
░ █ █ █ ░
░ █ █ █ ░
█ █ █ █ █
░ █ █ █ ░
░ ░ █ ░ ░

Skeleton:
░ ░ █ ░ ░
░ ░ █ ░ ░
░ █ █ █ ░
░ ░ █ ░ ░
░ ░ █ ░ ░
```

**意味**:
- 物体の「背骨」を抽出
- 形状のトポロジーを保持
- 同じ領域を複数回処理することを防ぐ

**実装**:
```python
from skimage import morphology
skeleton = morphology.skeletonize(binary_image)
```

#### Step 3: Medial Axis Transform

**Skeleton と Distance Map を組み合わせる**ことで、中心軸ピクセルに半径情報を付与します。

```
Medial Axis Transform = Skeleton × Distance Map
```

**意味**:
各中心軸ピクセルが持つ情報:
- 「この位置から半径Rの球体を描く」
- R = Medial Axis値 + 拡張パラメータ

**実装**:
```python
medial_axis = skeleton * distance_map
```

#### Step 4: 3D Reconstruction via Spherical Expansion

各中心軸ピクセルから**球体を上下に展開**します。

**球体の断面半径計算**:

球の方程式から導出:
```
x² + y² + z² = R²
```

z平面で切断すると:
```
x² + y² = R² - z²
```

したがって、断面半径:
```
r(z) = √(R² - z²)
```

**視覚化**:
```
        z
        ↑
        |     ●  ← R₀ = 10 px
        |    ╱│╲
   z=+2 |   ●─┼─●  ← r(2) = √(100-4) ≈ 9.8
        |  ╱  │  ╲
   z=0  | ●───┼───●  ← r(0) = 10
        |  ╲  │  ╱
   z=-2 |   ●─┼─●  ← r(-2) ≈ 9.8
        |    ╲│╱
        |     ●
        └─────────→ xy平面
```

**実装**:
```python
def reconstruct_3d(medial_axis, n_slices, mid_slice, elongation_factor):
    stack_3d = np.zeros((n_slices, height, width))
    
    for y, x in np.argwhere(skeleton):
        r0 = medial_axis[y, x] + radius_enlarge
        
        for z in range(n_slices):
            z_distance = (mid_slice - z) / elongation_factor
            r_squared = r0**2 - z_distance**2
            
            if r_squared > 0:
                segment_radius = np.sqrt(r_squared)
                rr, cc = morphology.disk(int(segment_radius))
                rr, cc = rr + y, cc + x
                stack_3d[z, rr, cc] = 255
    
    return stack_3d
```

### 3.3 Z方向のスライス数の自動推定

```python
max_distance = np.max(distance_map)
elongation_factor = voxel_xy / voxel_z
z_slices = 2 * (ceil(max_distance * elongation_factor) + 2)
```

**理論的根拠**:
1. 最大Distance値 = 物体の最大半径（pixels）
2. Z方向の範囲 = 最大半径 × elongation_factor（slices）
3. 直径分なので ×2
4. バッファとして +2 スライス

**例**:
- 最大半径: 20 px
- voxel_XY: 0.1 µm
- voxel_Z: 0.3 µm
- elongation: 0.333
- Z slices: 2 × (7 + 2) = 18

### 3.4 Elongation Factor（Z方向の距離補正）

XYとZの解像度が異なる場合、**Elongation Factor**で補正:

```
elongation_factor = voxel_XY / voxel_Z
```

実際のZ距離:
```
z_real = (z_slice - z_mid) / elongation_factor
```

### 3.5 アルゴリズムの利点

1. **形状に依存しない**
   - 楕円、不規則形状、複数の突起にも対応

2. **解像度の異方性に対応**
   - Elongation factorで自動補正

3. **計算効率が良い**
   - Skeletonにより処理ピクセル数を削減

4. **物理的に妥当**
   - 球体近似は細胞形状の良いモデル

### 3.6 アルゴリズムの制限事項

1. **球体近似が前提**
   - 凹凸の激しい形状では精度低下

2. **薄い構造の過小評価**
   - 非常に薄い/長い突起は再現しにくい

3. **重なり合う構造**
   - 複数の物体が接触している場合、分離困難

---

## 4. 回転対称体積推定アルゴリズム

### 4.1 アルゴリズムの原理

Odermatt et al. (2021) eLife 10:e64901 に基づく実装。

論文からの引用：
> "Each cell outline was skeletonized using custom Matlab code as follows. First, the closest-fitting rectangle around each cell was used to define the long axis of the cell. Perpendicular to the long axis, sectioning lines at 250 nm intervals and their intersection with the cell contour were computed. The centerline was then updated to run through the midpoint of each sectioning line between the two contour-intersection points."

### 4.2 アルゴリズムの核心ステップ

1. **長軸の決定**: 最小外接矩形
2. **断面線の配置**: 長軸に垂直、250nm間隔
3. **反復的更新**:
   - 各断面線と輪郭の交点を計算
   - 交点の中点を通るように中心線を更新
   - 中心線の局所的な傾きに垂直になるように断面線を更新
4. **体積計算**: 各断面を円形と仮定して回転対称体積を計算

### 4.3 反復的更新アルゴリズム

```python
for iteration in range(max_iterations):
    # 1. 各断面線と輪郭の交点を計算
    for i in range(n_sections):
        intersections = find_intersections(section_line, contour)
        midpoint = (p1 + p2) / 2
        
    # 2. 交点の中点を通るように中心線を更新
    new_centerline = interpolate(midpoints)
    
    # 3. 中心線の局所的な傾きに垂直になるように断面線を更新
    for i in range(n_sections):
        tangent = new_centerline[i] - new_centerline[i-1]
        perpendicular_angle = arctan2(tangent) + π/2
        update_section_line(i, perpendicular_angle)
    
    # 4. 収束判定
    mean_shift = np.mean(np.linalg.norm(new_centerline - old_centerline, axis=1))
    if mean_shift < convergence_tolerance:
        break
```

**パラメータ**:
- `max_iterations`: 最大反復回数（デフォルト: 3）
- `convergence_tolerance`: 収束閾値（デフォルト: 0.5ピクセル）

### 4.4 体積計算

各断面を円形と仮定して体積を計算：

```python
# 各断面の半径を計算
for i in range(n_sections):
    radius[i] = distance(p1, p2) / 2

# 体積計算（円柱の和）
total_volume = sum(π * r² * h for r in radii)
volume_um3 = total_volume * (pixel_size_um ** 3)
```

### 4.5 Z-stack厚みマップ生成

各XYピクセル位置でのZ方向の厚み（スライス数）を計算：

```python
# 回転対称を仮定
for center, radius in zip(centerline_points, radii):
    # 球体の断面: z = 2*sqrt(R² - r²)
    for y in range(height):
        for x in range(width):
            dist_from_center = distance((x, y), center)
            if dist_from_center <= radius:
                z_at_r = 2 * sqrt(radius² - dist_from_center²)
                thickness_map[y, x] = max(thickness_map[y, x], z_at_r)
```

### 4.6 実装の詳細

**長軸の決定**:
```python
rect = cv2.minAreaRect(contour.astype(np.float32))
center, size, angle = rect
```

**断面線の配置**:
```python
n_sections = int(axis_length / section_interval_px)
t = np.linspace(0, 1, n_sections)
section_centers = axis_start + t * (axis_end - axis_start)
```

---

## 5. 楕円・Feret径近似

### 5.1 楕円近似

#### 原理
ROIを楕円で近似し、3D形状を「円柱 + 両端の半球」としてモデル化：

```
       半球    円柱部    半球
        _______________
       /               \
      |                 |
      |                 |
       \_______________ /

      ← r →← h →← r →
```

**パラメータ**：
- r: 半径 = Minor / 2
- h: 円柱部の長さ = Major - 2r
- Major, Minor: ImageJのROI楕円近似パラメータ

#### ピクセルごとの厚み計算

各ピクセル (x, y) について、z方向の厚み d(x, y) を幾何学的に計算：

**円柱部分**:
```
d(x, y) = 2 × sqrt(r² - ρ²)
```

**半球部分**:
```
d(x, y) = 2 × sqrt(r² - s²)
```

#### 体積の数値積分

```
V = Σ d(x, y) × A_pixel
```

**実装**:
```python
PIXEL_SIZE_UM = 0.348  # ピクセルサイズ [µm]
pixel_area_um2 = PIXEL_SIZE_UM ** 2  # [µm²]

# マスク内のピクセルについて積算
mask = (zstack_map > 0)
volume_um3 = np.sum(zstack_map[mask] * pixel_area_um2)
```

### 5.2 Feret径近似

#### 原理
Feret径（Feret diameter）：物体の最大幅と最小幅を使用。

**Feret径の定義**:
- Major Feret: 最大幅
- Minor Feret: 最小幅（Major Feretに垂直な方向）

#### 実装
```python
def create_feret_mask(self, width, height, major, minor, angle, cx, cy):
    """Feret径に基づく3D形状近似"""
    # 回転行列
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    
    # 軸方向の距離計算
    dist_major = abs((px - cx) * cos_a + (py - cy) * sin_a)
    dist_minor = abs(-(px - cx) * sin_a + (py - cy) * cos_a)
    
    # 楕円内判定
    in_ellipse = (dist_major / half_major)**2 + (dist_minor / half_minor)**2 <= 1
    
    # 厚み計算
    # ...
```

#### 適用場面
- ✅ 細長い細胞（分裂酵母など）
- ✅ 楕円近似で精度が低い場合
- ❌ 複雑な形状

---

## 6. 精度向上テクニック

### 6.1 サブピクセルサンプリング

#### 原理
各ピクセルをN×Nのサブピクセルに分割して精度向上。

#### 実装
```python
# サブピクセルオフセット計算
offsets = np.linspace(0.5/N, 1 - 0.5/N, N) - 0.5

# 各サブピクセルで厚みを計算し平均
for dy_offset in offsets:
    for dx_offset in offsets:
        px_sub = px + 0.5 + dx_offset
        py_sub = py + 0.5 + dy_offset
        # 厚み計算...
        
thickness_pixel = thickness_sum / valid_subpixels
```

#### 推奨設定
- N=1: 高速（ピクセル中心のみ）
- N=5: 推奨（精度と速度のバランス）
- N=10: 最高精度

#### 効果
- 2-5%の精度向上
- 実行時間は約N²倍

### 6.2 背景補正

位相差画像の背景を補正：

```python
# 1枚目の画像を全体から減算
background = phase_images[0]
corrected_phases = phase_images - background
```

### 6.3 ROIスムージング

ImageJでROIを前処理：

```imagej
// Gap Closure & Smoothing
run("Enlarge...", "enlarge=" + gap + " pixel");
run("Enlarge...", "enlarge=-" + gap + " pixel");
```

効果：
- 小さな穴や凹みを埋める
- 滑らかな輪郭

---

## 📊 計算フローチャート

```
┌─────────────────────┐
│ オフアクシス        │
│ ホログラム画像      │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ FFTによる位相再構成 │
│ → 位相差 φ [rad]    │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ ROI検出             │
│ (Omnipose)          │
│ → Major, Minor, θ   │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ 体積推定手法選択    │
│ (4つから選択)       │
└──────┬──────────────┘
       │
       ├──────────────────────┐
       │                      │
       ▼                      ▼
┌──────────────┐      ┌───────────────┐
│ RI計算       │      │ 体積計算      │
│ n = n₀ +     │      │ V = Σ(d·A)    │
│  (φλ)/(2πd)  │      └───────┬───────┘
└──────┬───────┘              │
       │                      │
       ▼                      │
┌──────────────┐              │
│ 質量濃度計算 │              │
│ C = (n-n₀)/α │              │
└──────┬───────┘              │
       │                      │
       └──────────┬───────────┘
                  │
                  ▼
         ┌────────────────┐
         │ Total Mass計算 │
         │ M = Σ(C·V)     │
         └────────────────┘
```

---

## 📚 参考文献

### 主要論文

1. **Odermatt, P. D., et al. (2021)**  
   "Variations of intracellular density during the cell cycle arise from tip-growth regulation in fission yeast."  
   *eLife*, 10, e64901.  
   https://doi.org/10.7554/eLife.64901

2. **Park, Y., Depeursinge, C. & Popescu, G. (2018)**  
   "Quantitative phase imaging in biomedicine."  
   *Nature Photonics*, 12, 578–589.  
   https://doi.org/10.1038/s41566-018-0253-x

3. **Barer, R. & Joseph, S. (1954)**  
   "Refractometry of living cells."  
   *Quarterly Journal of Microscopical Science*, 95, 399-423.

4. **Popescu, G. (2011)**  
   "Quantitative Phase Imaging of Cells and Tissues."  
   McGraw-Hill Education.

5. **Mir, M. et al. (2011)**  
   "Optical measurement of cycle-dependent cell growth."  
   *Proceedings of the National Academy of Sciences*, 108, 13124-13129.

### ソフトウェア

6. **Pomegranate**  
   Baybay, E. K. D. (2020). Pomegranate: 3D Cell Segmentation Pipeline.  
   Virginia Tech, Hauf Lab.

7. **Omnipose**  
   Cutler, K. J., et al. (2022). "Omnipose: a high-precision morphology-independent solution for bacterial cell segmentation."  
   *Nature Methods*, 19, 1438-1448.

8. **Felzenszwalb, P. F., & Huttenlocher, D. P. (2012)**  
   "Distance transforms of sampled functions."  
   Theory of Computing, 8(1), 415-428.

9. **Zhang, T. Y., & Suen, C. Y. (1984)**  
   "A fast parallel algorithm for thinning digital patterns."  
   Communications of the ACM, 27(3), 236-239.

---

**最終更新**: 2025-12-24  
**プロジェクト**: QPI_omni  
**著者**: AI Assistant

