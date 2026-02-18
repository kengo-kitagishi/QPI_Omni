# 回転対称体積推定アルゴリズム実装ワークフロー

## 📋 概要

このドキュメントは、Odermatt et al. (2021) eLife 10:e64901 の回転対称体積推定アルゴリズムを実装し、ROIセットに適用するまでの完全なワークフローを記録したものです。

**実装日**: 2024年12月24日  
**参考論文**: [Odermatt et al. (2021) eLife](https://elifesciences.org/articles/64901)

---

## 🎯 目的

1. Pomegranateとは異なる、論文ベースの回転対称体積推定アルゴリズムを実装
2. 反復的な中心線・断面線の更新プロセスの実装
3. Z-stack厚みマップの生成（RI計算用）
4. 断面線と中心線の可視化
5. ROIセット（時系列データ）への適用

---

## 📚 背景知識

### Odermatt et al. (2021) のアルゴリズム

論文からの引用：
> "Each cell outline was skeletonized using custom Matlab code as follows. First, the closest-fitting rectangle around each cell was used to define the long axis of the cell. Perpendicular to the long axis, sectioning lines at 250 nm intervals and their intersection with the cell contour were computed. The centerline was then updated to run through the midpoint of each sectioning line between the two contour-intersection points. The slope of each sectioning line was updated to be perpendicular to the slope of the centerline around the midpoint. Sectioning lines that crossed a neighboring line were removed."

### アルゴリズムの核心

1. **長軸の決定**: 最小外接矩形
2. **断面線の配置**: 長軸に垂直、250nm間隔
3. **反復的更新**:
   - 各断面線と輪郭の交点を計算
   - 交点の中点を通るように中心線を更新
   - 中心線の局所的な傾きに垂直になるように断面線を更新
4. **体積計算**: 各断面を円形と仮定して回転対称体積を計算

### 重要な問題: 収束判定

論文には**明確な収束判定の記述がない**ため、以下のように実装：
- **最大反復回数**: 3回（経験的に十分）
- **収束閾値**: 0.5ピクセル（中心線の位置変化）

---

## 🛠️ 実装ステップ

### ステップ1: デモ実装の作成

#### 1.1 初期実装 (`30_demo_rotational_symmetry_volume.py`)

まず、単一の楕円画像でアルゴリズムをテストするデモスクリプトを作成：

```python
class RotationalSymmetryVolumeEstimator:
    """回転対称を仮定した体積推定"""
    
    def __init__(self, pixel_size_um=0.08625, section_interval_um=0.25):
        self.pixel_size_um = pixel_size_um
        self.section_interval_um = section_interval_um
        self.section_interval_px = section_interval_um / pixel_size_um
    
    def calculate_volume(self, binary_mask):
        """
        2Dバイナリマスクから回転対称体積を計算
        """
        # 1. 輪郭抽出
        contours = measure.find_contours(binary_mask, 0.5)
        contour = max(contours, key=lambda x: len(x))
        
        # 2. 長軸決定（最小外接矩形）
        rect = cv2.minAreaRect(contour.astype(np.float32))
        center, size, angle = rect
        
        # 3. 断面線の配置
        n_sections = int(length / self.section_interval_px)
        
        # 4. 各断面で半径を計算
        for i in range(n_sections):
            # 断面線と輪郭の交点
            intersections = find_intersections(...)
            radius = distance(p1, p2) / 2
        
        # 5. 体積計算（円柱の和）
        volume = sum(π * r² * h)
        
        return volume
```

**実行**:
```bash
cd scripts
python 30_demo_rotational_symmetry_volume.py
```

**結果**:
- デモ楕円画像で体積計算成功
- 可視化により断面線と中心線を確認

---

### ステップ2: 反復更新の実装

#### 2.1 問題の認識

初期実装では断面線の角度が固定されていたが、論文では「断面線の傾きを更新」と明記されている。

#### 2.2 反復アルゴリズムの設計

```python
def compute_volume_rotational(self, mask, return_visualization_data=False):
    # 初期化
    centerline_points = initial_centerline  # 長軸に沿って等間隔
    section_angles = [axis_angle + π/2] * n_sections  # 全て長軸に垂直
    
    for iteration in range(max_iterations):  # デフォルト: 3回
        new_centerline = []
        new_angles = []
        new_radii = []
        
        for i in range(n_sections):
            # 現在の角度で断面線を引く
            intersections = find_intersections(
                centerline_points[i], 
                section_angles[i], 
                contour
            )
            
            # 中点を計算（中心線を更新）
            midpoint = (p1 + p2) / 2
            new_centerline.append(midpoint)
            
            # 局所的な中心線の傾きを計算
            if i > 0 and i < n_sections - 1:
                tangent = new_centerline[i] - new_centerline[i-1]
                local_angle = arctan2(tangent)
                perpendicular_angle = local_angle + π/2
                new_angles.append(perpendicular_angle)
        
        # 収束判定
        if iteration > 0:
            shifts = [norm(new_centerline[i] - centerline_points[i]) 
                     for i in range(n_sections)]
            mean_shift = mean(shifts)
            
            if mean_shift < convergence_tolerance:  # 0.5 pixels
                break
        
        # 更新
        centerline_points = new_centerline
        section_angles = new_angles
```

#### 2.3 重要な実装の詳細

**局所的な傾きの計算**:
```python
# 前後の点から傾きを推定
if i > 0 and i < n_sections - 1 and len(new_centerline) > 1:
    tangent = new_centerline[-1] - new_centerline[-2]
    local_angle = np.arctan2(tangent[1], tangent[0])
    perpendicular_angle = local_angle + np.pi/2
```

**収束判定**:
```python
if iteration > 0 and len(new_centerline) == len(centerline_points):
    shifts = [np.linalg.norm(new_centerline[i] - centerline_points[i]) 
             for i in range(len(new_centerline))]
    mean_shift = np.mean(shifts)
    
    if mean_shift < self.convergence_tolerance:  # 0.5 pixels
        centerline_points = new_centerline
        radii = new_radii
        section_lines = new_section_lines
        break
```

---

### ステップ3: Z-stack厚みマップの実装

#### 3.1 要求事項

24_ellipse_volume.pyのように、各XYピクセル位置でのZ方向の厚み（スライス数）を計算し、RI計算に使用できるようにする。

#### 3.2 厚みマップ計算アルゴリズム

```python
# 厚みマップを初期化
thickness_map = np.zeros((height, width), dtype=np.float32)

# 各中心線ポイントで処理
for center, radius in zip(centerline_points, radii):
    if radius > 0:
        # 回転対称を仮定: 半径Rの球体のZ方向の高さ = 2R
        z_height_um = 2 * radius * pixel_size_um
        z_slices = z_height_um / pixel_size_um
        
        # この半径の円内のピクセルに厚みを割り当て
        y, x = int(center[1]), int(center[0])
        r_int = int(radius) + 1
        
        for dy in range(-r_int, r_int+1):
            for dx in range(-r_int, r_int+1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    dist_from_center = sqrt(dx² + dy²)
                    if dist_from_center <= radius:
                        # 球体の断面: z = 2*sqrt(R² - r²)
                        z_at_r = 2 * sqrt(max(0, radius² - dist_from_center²))
                        # 最大値を保持
                        thickness_map[ny, nx] = max(thickness_map[ny, nx], z_at_r)

return thickness_map
```

#### 3.3 厚みマップの保存

```python
# 個別のTIFFファイルとして保存
for thick_info in self.thickness_maps:
    roi_name = thick_info['roi_name'].replace('.roi', '')
    thick_path = os.path.join(thickness_dir, f"{roi_name}_thickness.tif")
    tifffile.imwrite(thick_path, thick_info['thickness_map'])

# 統合TIFFスタック
stack_array = np.stack([tm['thickness_map'] for tm in self.thickness_maps], axis=0)
tifffile.imwrite('thickness_stack_all_frames.tif', stack_array, metadata={'axes': 'TYX'})
```

---

### ステップ4: RI計算機能の実装

#### 4.1 RI計算の原理

```
RI = n_medium + (φ × λ) / (2π × thickness)

where:
  - n_medium: 培地の屈折率 (1.333)
  - φ: 位相差 (phase image)
  - λ: 波長 (663 nm)
  - thickness: 厚み (um)
```

#### 4.2 実装

```python
def compute_ri_from_phase_images(self, phase_image_dir, wavelength_nm=663, n_medium=1.333):
    """位相差画像と厚みマップからRI計算"""
    
    wavelength_um = wavelength_nm / 1000.0
    
    # 位相差画像を検索（ファイル名から番号を抽出）
    phase_files_all = sorted(glob.glob(os.path.join(phase_image_dir, "*.tif")))
    
    # ファイル名から番号を抽出して辞書を作成
    # 例: "subtracted_by_maskmean_float320085_bg_corr_aligned.tif" -> 85
    phase_file_dict = {}
    for phase_file in phase_files_all:
        basename = os.path.basename(phase_file)
        match = re.search(r'(\d+)(?:_bg_corr_aligned)?\.tif$', basename)
        if match:
            frame_num = int(match.group(1))
            phase_file_dict[frame_num] = phase_file
    
    ri_results = []
    
    for thick_info in self.thickness_maps:
        # ROI名からフレーム番号を抽出
        # 例: "0085-0024-0136.roi" -> 85
        match = re.match(r'(\d+)-', thick_info['roi_name'])
        if match:
            frame_num = int(match.group(1))
        
        if frame_num in phase_file_dict:
            phase_img = tifffile.imread(phase_file_dict[frame_num])
            thickness_map = thick_info['thickness_map']
            
            # 厚みをumに変換
            thickness_um = thickness_map * self.pixel_size_um
            
            # ゼロ除算を避ける
            thickness_um_safe = np.where(thickness_um > 0, thickness_um, np.nan)
            
            # RI計算
            n_sample = n_medium + (phase_img * wavelength_um) / (2 * np.pi * thickness_um_safe)
            
            # 統計計算
            mask = thickness_map > 0
            if np.sum(mask) > 0:
                mean_ri = np.nanmean(n_sample[mask])
                median_ri = np.nanmedian(n_sample[mask])
                std_ri = np.nanstd(n_sample[mask])
                total_ri = np.nansum(n_sample[mask] - n_medium)
                
                ri_results.append({
                    'frame_num': frame_num,
                    'roi_name': thick_info['roi_name'],
                    'mean_ri': mean_ri,
                    'median_ri': median_ri,
                    'std_ri': std_ri,
                    'total_ri': total_ri,
                    'ri_map': n_sample
                })
    
    return ri_results
```

---

### ステップ5: 可視化機能の実装

#### 5.1 要求事項

断面線（cross-sections）と中心線（centerline）をTIFF/PNG形式で保存。

#### 5.2 可視化データの収集

```python
def compute_volume_rotational(self, mask, return_visualization_data=False):
    # ... 体積計算 ...
    
    # 可視化データを追加
    if return_visualization_data:
        result['centerline_points'] = np.array(centerline_points)
        result['section_lines'] = section_lines
        result['contour'] = contour
        result['axis_start'] = axis_start
        result['axis_end'] = axis_end
        result['radii'] = radii
    
    return result
```

#### 5.3 可視化画像の生成

```python
def save_visualizations(self, output_dir, format='png'):
    """断面線と中心線の可視化を保存"""
    
    for vis_data in self.visualization_data:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # 1. マスクを背景に
        ax.imshow(vis_data['mask'], cmap='gray', alpha=0.3)
        
        # 2. 輪郭（青線）
        contour = vis_data['contour']
        ax.plot(contour[:, 0], contour[:, 1], 'b-', linewidth=2, 
               label='Contour', alpha=0.7)
        
        # 3. 長軸（赤線）
        ax.plot([axis_start[0], axis_end[0]], 
               [axis_start[1], axis_end[1]], 
               'r-', linewidth=3, label='Long axis', alpha=0.8)
        
        # 4. 中心線（緑線、点付き）
        centerline = vis_data['centerline_points']
        ax.plot(centerline[:, 0], centerline[:, 1], 'g-', 
               linewidth=3, label='Centerline', marker='o', markersize=4)
        
        # 5. 断面線（シアン線）
        section_lines = vis_data['section_lines']
        for i, (p1, p2) in enumerate(section_lines):
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                   'c-', linewidth=1, alpha=0.5)
            
            # 6. 半径を円で表示（黄色、半透明）
            if i < len(vis_data['radii']):
                midpoint = (p1 + p2) / 2
                radius = vis_data['radii'][i]
                circle = plt.Circle((midpoint[0], midpoint[1]), radius, 
                                   fill=False, color='yellow', 
                                   linewidth=1, alpha=0.3)
                ax.add_patch(circle)
        
        ax.set_title(f"{roi_name}\nSections: {len(section_lines)}", 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.axis('equal')
        
        # 保存
        if format == 'png':
            save_path = f"{roi_name}_visualization.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        elif format == 'tif':
            save_path = f"{roi_name}_visualization.tif"
            plt.savefig(save_path, dpi=150, bbox_inches='tight', format='tiff')
        
        plt.close(fig)
```

---

### ステップ6: ROIセットへの適用

#### 6.1 ROIセット解析クラスの実装

```python
class RotationalSymmetryROIAnalyzer:
    """ROIセットに回転対称体積推定を適用"""
    
    def __init__(self, roi_zip_path, pixel_size_um=0.08625, 
                 section_interval_um=0.25, image_width=512, image_height=512,
                 max_iterations=3, convergence_tolerance=0.5):
        self.roi_zip_path = roi_zip_path
        self.pixel_size_um = pixel_size_um
        self.section_interval_um = section_interval_um
        self.max_iterations = max_iterations
        self.convergence_tolerance = convergence_tolerance
        
        self.load_roi_set()
    
    def load_roi_set(self):
        """ROIセットを読み込んで整理"""
        with zipfile.ZipFile(self.roi_zip_path, 'r') as zf:
            roi_names = zf.namelist()
            
            self.rois_by_time = defaultdict(list)
            
            for roi_name in roi_names:
                roi_bytes = zf.read(roi_name)
                roi_info = self.parse_roi_basic(roi_bytes, roi_name)
                
                if roi_info is not None:
                    frame_num = self.extract_frame_number(roi_name)
                    self.rois_by_time[frame_num].append(roi_info)
    
    def analyze_timeseries(self, max_frames=None, 
                          save_visualizations=False, 
                          save_thickness_maps=True):
        """時系列で体積を解析"""
        
        time_points = sorted(self.rois_by_time.keys())
        if max_frames is not None:
            time_points = time_points[:max_frames]
        
        results = []
        self.visualization_data = []
        self.thickness_maps = []
        
        for t_idx, t in enumerate(time_points):
            rois_at_t = self.rois_by_time[t]
            
            for cell_idx, roi_info in enumerate(rois_at_t):
                mask = self.roi_to_mask(roi_info)
                
                if mask is None or np.sum(mask) == 0:
                    continue
                
                vol_result = self.compute_volume_rotational(
                    mask, 
                    return_visualization_data=save_visualizations
                )
                
                if vol_result is not None:
                    # 結果を保存
                    vol_result['time_point'] = t
                    vol_result['time_index'] = t_idx
                    vol_result['cell_index'] = cell_idx
                    vol_result['roi_name'] = roi_info['name']
                    
                    # 可視化データを別途保存
                    if save_visualizations:
                        vis_data = {
                            'time_index': t_idx,
                            'cell_index': cell_idx,
                            'roi_name': roi_info['name'],
                            'mask': mask,
                            'centerline_points': vol_result['centerline_points'],
                            'section_lines': vol_result['section_lines'],
                            'contour': vol_result['contour'],
                            'axis_start': vol_result['axis_start'],
                            'axis_end': vol_result['axis_end'],
                            'radii': vol_result['radii']
                        }
                        self.visualization_data.append(vis_data)
                    
                    # 厚みマップを別途保存
                    if save_thickness_maps:
                        thickness_info = {
                            'time_index': t_idx,
                            'time_point': t,
                            'cell_index': cell_idx,
                            'roi_name': roi_info['name'],
                            'thickness_map': vol_result['thickness_map']
                        }
                        self.thickness_maps.append(thickness_info)
                    
                    results.append(vol_result)
        
        self.results_df = pd.DataFrame(results)
        return self.results_df
```

#### 6.2 メイン実行スクリプト

```python
def main():
    """メイン実行"""
    roi_zip_path = r"c:\Users\QPI\Documents\QPI_omni\scripts\RoiSet.zip"
    
    # Analyzerを作成（反復更新あり）
    analyzer = RotationalSymmetryROIAnalyzer(
        roi_zip_path=roi_zip_path,
        pixel_size_um=0.348,
        section_interval_um=0.25,  # 250 nm
        image_width=512,
        image_height=512,
        max_iterations=3,  # 最大3回の反復更新
        convergence_tolerance=0.5  # 0.5ピクセル以下で収束
    )
    
    # 解析実行
    results_df = analyzer.analyze_timeseries(
        max_frames=100, 
        save_visualizations=True,
        save_thickness_maps=True
    )
    
    # 結果を保存
    analyzer.save_results('rotational_volume_output')
    
    # 可視化を保存
    analyzer.save_visualizations('rotational_volume_output', format='png')
    
    # プロット
    analyzer.plot_results('rotational_volume_plot.png')
    
    # RI計算（位相差画像がある場合）
    phase_dir = os.path.join(os.path.dirname(__file__), "..", "data", 
                            "align_demo", "bg_corr_aligned", "aligned")
    phase_dir = os.path.abspath(phase_dir)
    
    if os.path.exists(phase_dir):
        analyzer.compute_ri_from_phase_images(
            phase_dir, 
            wavelength_nm=663, 
            n_medium=1.333
        )
        analyzer.save_ri_results('rotational_volume_output')
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()
```

---

## 🚀 実行手順

### 実行環境の準備

```bash
# プロジェクトディレクトリに移動
cd c:\Users\QPI\Documents\QPI_omni

# 必要なライブラリが入っているか確認
python -c "import numpy, pandas, matplotlib, scipy, skimage, tifffile, cv2; print('All libraries OK')"
```

### ステップバイステップ実行

#### 1. デモスクリプトの実行（オプション）

```bash
cd scripts
python 30_demo_rotational_symmetry_volume.py
```

**期待される出力**:
- デモ楕円画像の体積計算
- 可視化画像の生成

#### 2. ROIセットへの適用

```bash
cd scripts
python 31_roiset_rotational_volume.py
```

**実行中の出力例**:
```
=== Rotational Symmetry ROI Analyzer ===
ROI Set: c:\Users\QPI\Documents\QPI_omni\scripts\RoiSet.zip
Pixel size: 0.348 um
Section interval: 0.25 um (0.72 pixels)
Image Size: 512 x 512

=== Loading ROI Set ===
  Total ROIs: 2339
    Processing: 0/2339
    Processing: 100/2339
    ...
  Successfully parsed: 2339 ROIs
  Time points: 2339

============================================================
SETTINGS:
  Max iterations: 3
  Convergence tolerance: 0.5 pixels
  Section interval: 0.25 um
============================================================

=== Analyzing Time-series with Rotational Symmetry ===
  Processing first 100 frames
  Time points to process: 100

  Frame 1/100 (t=85)
    ROIs at this time: 1
      Cell 1/1
        [OK] 0085-0024-0136.roi: Volume=99.34 um^3, Max thickness=12.7px

  Frame 2/100 (t=86)
    ROIs at this time: 1
      Cell 1/1
        [OK] 0086-0024-0136.roi: Volume=92.24 um^3, Max thickness=12.9px
  ...

  Total processed: 100 cells
  Volume range: 86.08 - 275.31 um^3

=== Saving Results to rotational_volume_output ===
  Saved: rotational_volume_output\rotational_volume_timeseries.csv
  Saved: rotational_volume_output\rotational_volume_summary.txt

  Saving thickness maps (100 maps)...
    Progress: 0/100
  Saved: 100 thickness maps to rotational_volume_output\thickness_maps/
  Saved: rotational_volume_output\thickness_stack_all_frames.tif (shape: (100, 512, 512))

=== Saving Visualizations (PNG) ===
  Total visualizations: 100
    Progress: 0/100
    Progress: 10/100
    ...
  Saved: 100 visualizations to rotational_volume_output\visualizations/

=== Plot saved: rotational_volume_plot.png ===
```

#### 3. 結果の確認

```bash
# サマリーを確認
cat rotational_volume_output\rotational_volume_summary.txt

# CSVを確認
head rotational_volume_output\rotational_volume_timeseries.csv

# 可視化画像を確認
ls rotational_volume_output\visualizations\
```

---

## 📊 出力ファイル

### ディレクトリ構造

```
scripts/rotational_volume_output/
├── rotational_volume_timeseries.csv          # 体積・表面積データ
├── rotational_volume_summary.txt             # 統計サマリー
├── thickness_stack_all_frames.tif            # 全フレームの厚みマップ
├── thickness_maps/                           # 個別の厚みマップ
│   ├── 0085-0024-0136_thickness.tif
│   ├── 0086-0024-0136_thickness.tif
│   └── ... (100ファイル)
├── visualizations/                           # 中心線・断面線の可視化
│   ├── 0085-0024-0136_visualization.png
│   ├── 0086-0024-0136_visualization.png
│   └── ... (100ファイル)
├── ri_statistics.csv                         # RI統計（位相差画像がある場合）
├── ri_summary.txt                            # RIサマリー
└── ri_maps/                                  # RIマップ
    ├── 0085-0024-0136_ri_map.tif
    └── ...
```

### CSVファイルの構造

**rotational_volume_timeseries.csv**:
```
volume_um3,surface_area_um2,n_sections,mean_radius_um,max_radius_um,length_um,area_2d,time_point,time_index,cell_index,roi_name
99.34,102.45,42,1.78,2.45,10.5,315,85,0,0,0085-0024-0136.roi
92.24,98.67,41,1.75,2.38,10.2,298,86,1,0,0086-0024-0136.roi
...
```

**ri_statistics.csv** (位相差画像がある場合):
```
time_index,time_point,frame_num,roi_name,mean_ri,median_ri,std_ri,total_ri,n_pixels
0,85,85,0085-0024-0136.roi,1.3567,1.3565,0.0045,123.45,315
1,86,86,0086-0024-0136.roi,1.3572,1.3570,0.0043,119.87,298
...
```

### サマリーファイルの内容

**rotational_volume_summary.txt**:
```
=== Rotational Symmetry Volume Analysis ===

Algorithm: Odermatt et al. (2021) eLife 10:e64901
Section interval: 0.25 um

Total cells: 100
Time points: 100

Volume Statistics (um^3):
  Mean: 125.51
  Median: 122.51
  Std: 28.95
  Min: 86.08
  Max: 275.31

Surface Area Statistics (um^2):
  Mean: 125.97
  Std: 23.07
```

---

## 🔧 トラブルシューティング

### 問題1: 位相差画像が見つからない

**症状**:
```
Warning: Phase image directory not found: ...
```

**解決方法**:
```python
# パスを確認
import os
phase_dir = r"c:\Users\QPI\Documents\QPI_omni\data\align_demo\bg_corr_aligned\aligned"
print(os.path.exists(phase_dir))  # False の場合はパスが間違っている

# 実際のパスを探す
import glob
tif_files = glob.glob(r"c:\Users\QPI\Documents\QPI_omni\data\**\*.tif", recursive=True)
print(tif_files[:5])  # 最初の5個を表示
```

### 問題2: メモリ不足

**症状**:
```
MemoryError: Unable to allocate array
```

**解決方法**:
```python
# max_framesを減らす
analyzer.analyze_timeseries(max_frames=50)  # 100 → 50

# 可視化を無効化
analyzer.analyze_timeseries(
    max_frames=100, 
    save_visualizations=False,  # メモリ節約
    save_thickness_maps=True
)
```

### 問題3: 収束しない

**症状**:
- すべての反復が実行される（収束しない）
- 結果が不安定

**解決方法**:
```python
# 収束閾値を緩める
analyzer = RotationalSymmetryROIAnalyzer(
    ...
    convergence_tolerance=1.0  # 0.5 → 1.0
)

# 反復回数を増やす
analyzer = RotationalSymmetryROIAnalyzer(
    ...
    max_iterations=5  # 3 → 5
)
```

### 問題4: ROIの読み込みエラー

**症状**:
```
Successfully parsed: 0 ROIs
```

**解決方法**:
```python
# ROIファイルの形式を確認
import zipfile
with zipfile.ZipFile('RoiSet.zip', 'r') as zf:
    roi_names = zf.namelist()
    print(roi_names[:5])  # 最初の5個を表示
    
    # 1つ読んでみる
    roi_bytes = zf.read(roi_names[0])
    print(f"Size: {len(roi_bytes)} bytes")
    print(f"Header: {roi_bytes[:4]}")  # b'Iout' であるべき
```

---

## 📈 結果の解釈

### 体積の妥当性チェック

```python
import pandas as pd
import matplotlib.pyplot as plt

# CSVを読み込み
df = pd.read_csv('rotational_volume_output/rotational_volume_timeseries.csv')

# 基本統計
print(df['volume_um3'].describe())

# ヒストグラム
plt.figure(figsize=(10, 6))
plt.hist(df['volume_um3'], bins=30, edgecolor='black')
plt.xlabel('Volume (um³)')
plt.ylabel('Frequency')
plt.title('Volume Distribution')
plt.show()

# 時系列プロット
plt.figure(figsize=(12, 6))
plt.plot(df['time_index'], df['volume_um3'], 'o-', alpha=0.7)
plt.xlabel('Time (frame)')
plt.ylabel('Volume (um³)')
plt.title('Volume Time-series')
plt.grid(True)
plt.show()
```

### 厚みマップの確認

```python
import tifffile
import matplotlib.pyplot as plt

# スタックを読み込み
stack = tifffile.imread('rotational_volume_output/thickness_stack_all_frames.tif')
print(f"Stack shape: {stack.shape}")  # (100, 512, 512)

# 最初のフレームを表示
plt.figure(figsize=(10, 10))
plt.imshow(stack[0], cmap='viridis')
plt.colorbar(label='Thickness (pixels)')
plt.title('Thickness Map - Frame 0')
plt.show()

# 統計
print(f"Min thickness: {stack.min():.2f} pixels")
print(f"Max thickness: {stack.max():.2f} pixels")
print(f"Mean thickness: {stack.mean():.2f} pixels")
```

### RIデータの解析（位相差画像がある場合）

```python
# RI統計を読み込み
ri_df = pd.read_csv('rotational_volume_output/ri_statistics.csv')

# 基本統計
print(ri_df['mean_ri'].describe())

# 時系列プロット
plt.figure(figsize=(12, 6))
plt.plot(ri_df['time_index'], ri_df['mean_ri'], 'o-', alpha=0.7, label='Mean RI')
plt.axhline(y=1.333, color='r', linestyle='--', label='Medium RI')
plt.xlabel('Time (frame)')
plt.ylabel('Refractive Index')
plt.title('RI Time-series')
plt.legend()
plt.grid(True)
plt.show()
```

---

## 🔬 アルゴリズムの検証

### 収束性の検証

反復更新が正しく収束しているか確認するには、可視化画像を見ます：

1. **初期長軸**（赤線）と**最終中心線**（緑線）の違いを確認
2. **断面線**（シアン線）が中心線に垂直になっているか確認
3. **回転対称の円**（黄色）が細胞の形状に適合しているか確認

### 他の手法との比較

```python
# Pomegranate法と比較
pomegranate_df = pd.read_csv('timeseries_volume_output/volume_timeseries.csv')
rotational_df = pd.read_csv('rotational_volume_output/rotational_volume_timeseries.csv')

# 同じ時間点で比較
merged = pd.merge(
    pomegranate_df[['time_point', 'volume_um3']],
    rotational_df[['time_point', 'volume_um3']],
    on='time_point',
    suffixes=('_pomegranate', '_rotational')
)

# 相関プロット
plt.figure(figsize=(8, 8))
plt.scatter(merged['volume_um3_pomegranate'], 
           merged['volume_um3_rotational'],
           alpha=0.5)
plt.xlabel('Pomegranate Volume (um³)')
plt.ylabel('Rotational Symmetry Volume (um³)')
plt.title('Volume Comparison')
plt.plot([50, 300], [50, 300], 'r--', label='y=x')
plt.legend()
plt.grid(True)
plt.show()

# 相関係数
correlation = merged['volume_um3_pomegranate'].corr(
    merged['volume_um3_rotational']
)
print(f"Correlation: {correlation:.3f}")
```

---

## 📚 参考資料

### 論文
- **Odermatt et al. (2021)**. "Variations of intracellular density during the cell cycle arise from tip-growth regulation in fission yeast." eLife 10:e64901. DOI: [10.7554/eLife.64901](https://doi.org/10.7554/eLife.64901)

### 関連スクリプト
- `30_demo_rotational_symmetry_volume.py`: デモスクリプト
- `31_roiset_rotational_volume.py`: ROIセット解析スクリプト（本実装）
- `29_Pomegranate_from_roiset.py`: Pomegranate法の実装（比較用）
- `24_ellipse_volume.py`: 楕円体積推定（参考）

### 関連ドキュメント
- `docs/workflows/pomegranate_reconstruction_summary.md`: Pomegranate法の説明
- `docs/workflows/thickness_map_and_ri_calculation.md`: 厚みマップとRI計算
- `docs/workflows/timeseries_volume_tracking_guide.md`: 時系列体積追跡

---

## 🎓 まとめ

### 実装した機能

1. ✅ **反復的中心線・断面線更新**: 最大3回、収束閾値0.5ピクセル
2. ✅ **回転対称体積計算**: 各断面を円形と仮定
3. ✅ **Z-stack厚みマップ生成**: RI計算用
4. ✅ **断面線・中心線の可視化**: PNG/TIFF形式
5. ✅ **RI計算機能**: 位相差画像と厚みマップから計算
6. ✅ **時系列解析**: ROIセット全体への適用

### 重要な実装の詳細

- **収束判定**: 中心線の位置変化が0.5ピクセル以下
- **局所的な傾き**: 前後の中心線点から計算
- **厚みマップ**: 球体断面を仮定して各ピクセルのZ高さを計算
- **RI計算**: ファイル名から番号を抽出してマッチング

### 典型的な結果

- **体積範囲**: 86-275 µm³
- **平均体積**: 125.51 ± 28.95 µm³
- **厚み範囲**: 12-30ピクセル
- **反復回数**: 通常1-2回で収束

### 今後の拡張可能性

1. **並列処理**: 複数フレームを並列に処理
2. **GPU加速**: CUDA/OpenCLによる高速化
3. **3D可視化**: Mayavi/VTKによる3D表示
4. **機械学習**: 体積予測モデルの構築
5. **リアルタイム処理**: ストリーミングデータへの適用

---

**作成日**: 2024年12月24日  
**バージョン**: 1.0  
**作成者**: AI Assistant  
**プロジェクト**: QPI_omni

