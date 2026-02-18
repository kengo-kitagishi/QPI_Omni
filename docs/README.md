# QPI解析 ドキュメント

このディレクトリには、QPI解析プロジェクトに関するドキュメントが含まれています。

---

## 📚 ドキュメント構成

### 主要ドキュメント

| ドキュメント | 内容 | 対象読者 |
|-------------|------|---------|
| **[USAGE_GUIDE.md](USAGE_GUIDE.md)** | 使い方・クイックスタート・トラブルシューティング | すべてのユーザー |
| **[METHODS.md](METHODS.md)** | 手法・アルゴリズムの理論と実装 | 詳細を知りたいユーザー |
| **[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)** | 時系列での開発・実験記録 | 開発者・履歴を追いたいユーザー |

---

## 🚀 はじめに

### 初めての方

まず **[USAGE_GUIDE.md](USAGE_GUIDE.md)** をご覧ください。

最速でスタート：
```bash
cd scripts
python 24_ellipse_volume.py
```

### 手法の詳細を知りたい方

**[METHODS.md](METHODS.md)** で各手法の理論と実装を確認してください。

### 開発履歴を知りたい方

**[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)** で時系列の開発記録を確認してください。

---

## 📋 クイックリンク

### 使い方

- [クイックスタート](USAGE_GUIDE.md#1-クイックスタート)
- [手法別の使い方](USAGE_GUIDE.md#2-手法別の使い方)
- [パラメータリファレンス](USAGE_GUIDE.md#3-パラメータリファレンス)
- [トラブルシューティング](USAGE_GUIDE.md#5-トラブルシューティング)

### 手法・アルゴリズム

- [QPI基礎理論](METHODS.md#1-qpi基礎理論)
- [体積推定手法の比較](METHODS.md#2-体積推定手法の比較)
- [Pomegranate 3D再構成](METHODS.md#3-pomegranate-3d再構成アルゴリズム)
- [回転対称体積推定](METHODS.md#4-回転対称体積推定アルゴリズム)

### 開発ログ

- [2025-12-23: Total Mass計算実装](EXPERIMENT_LOG.md#実験1-total-mass計算と時系列プロット機能の実装)
- [2025-12-23: Pomegranate実装](EXPERIMENT_LOG.md#実験3-pomegranate-3d再構成アルゴリズムの実装)
- [2025-12-24: 回転対称実装](EXPERIMENT_LOG.md#実験5-回転対称体積推定アルゴリズムの実装)
- [2025-12-24: バッチシステム](EXPERIMENT_LOG.md#実験6-体積推定メソッド比較システムの構築)

---

## 🎯 目的別ガイド

### 体積を計算したい

→ [手法別の使い方](USAGE_GUIDE.md#2-手法別の使い方)

推奨手法：
- **高速**: 楕円近似（24_ellipse_volume.py）
- **高精度**: 回転対称（31_roiset_rotational_volume.py）
- **複雑な形状**: Pomegranate（29_Pomegranate_from_roiset.py）

### RIを計算したい

→ [QPI基礎理論](METHODS.md#13-屈折率riの計算)

必要なもの：
- 位相差画像
- 厚みマップ
- 波長、培地屈折率

### Total Massを計算したい

→ [Total Massの計算](METHODS.md#15-total-massの計算)

計算式：
```
M_total [pg] = Σ C [mg/ml] × V [µm³]
```

### 時系列データを解析したい

→ [Pomegranate 3D再構成](USAGE_GUIDE.md#22-pomegranate-3d再構成)

入力：ImageJ ROIセット（.zip）  
出力：体積の時系列データ（CSV）

### 複数のパラメータで比較したい

→ [バッチ解析](USAGE_GUIDE.md#24-バッチ解析27_compare_volume_estimation_methodspy)

実行条件：
- 2 CSVs × 2 shape_types × 3 subpixel_samplings = 12条件

### エラーが出た

→ [トラブルシューティング](USAGE_GUIDE.md#5-トラブルシューティング)

よくある問題：
- 実行が遅い
- メモリ不足
- ROIが読み込めない
- 位相差画像が見つからない

---

## 📊 体積推定手法の選び方

| 手法 | 特徴 | 精度 | 速度 | 適用場面 |
|------|------|------|------|---------|
| **楕円近似** | シンプル | ★★★☆☆ | ★★★★★ | 楕円形に近い細胞、高速処理 |
| **Feret径近似** | 細長い細胞に強い | ★★★★☆ | ★★★★☆ | 細長い細胞（分裂酵母など） |
| **Pomegranate** | 複雑な形状に対応 | ★★★★☆ | ★★★☆☆ | 複雑な形状、2D→3D再構成 |
| **回転対称** | 論文準拠、高精度 | ★★★★★ | ★★☆☆☆ | 高精度が必要、論文準拠 |

詳細：[体積推定手法の比較](METHODS.md#2-体積推定手法の比較)

---

## 🛠️ 主要スクリプト

| スクリプト | 説明 | 実行時間 |
|-----------|------|---------|
| `24_ellipse_volume.py` | 楕円・Feret径近似 | 約1-2分 |
| `27_compare_volume_estimation_methods.py` | バッチ解析（12条件） | 約1-2時間 |
| `29_Pomegranate_from_roiset.py` | Pomegranate 3D再構成 | 約5-10分 |
| `31_roiset_rotational_volume.py` | 回転対称体積推定 | 約2-3分 |
| `30_plot_filtered_conditions.py` | 時系列プロット | 約30秒 |

---

## 📖 背景知識

### QPI（定量位相イメージング）とは

光の位相差から細胞の屈折率（RI）を測定し、質量濃度や体積を定量する手法。

詳細：[QPI基礎理論](METHODS.md#1-qpi基礎理論)

### 体積推定の原理

2D画像から3D形状を推定する手法：
1. 楕円近似・Feret径近似
2. Distance Transform + 球体展開（Pomegranate）
3. 反復的中心線・断面線更新（回転対称）

詳細：[体積推定手法の比較](METHODS.md#2-体積推定手法の比較)

---

## 📚 参考文献

### 主要論文

1. **Odermatt et al. (2021)** - 回転対称体積推定  
   *eLife*, 10, e64901.

2. **Park et al. (2018)** - QPI基礎  
   *Nature Photonics*, 12, 578–589.

3. **Barer & Joseph (1954)** - 屈折率と質量濃度  
   *Quarterly Journal of Microscopical Science*, 95, 399-423.

詳細：[参考文献](METHODS.md#-参考文献)

---

## 🔄 更新履歴

| 日付 | 内容 |
|------|------|
| 2025-12-24 | ドキュメント統合・整理 |
| 2025-12-24 | 回転対称体積推定実装 |
| 2025-12-24 | バッチ解析システム実装 |
| 2025-12-23 | Total Mass計算実装 |
| 2025-12-23 | Pomegranate 3D再構成実装 |

詳細：[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)

---

## 💡 ヒント

### 最初は少ないフレームで試す

```python
# 最初は10フレームで試して、問題がないか確認
results_df = analyzer.analyze_timeseries(max_frames=10)
```

### 可視化で結果を確認

可視化画像で以下を確認：
- ✅ 中心線が細胞の中央を通っているか
- ✅ 断面線が中心線に垂直か
- ✅ 回転対称円が細胞に適合しているか

### 結果のバックアップ

```bash
# 重要な結果をバックアップ
mkdir backup_2025-12-24
cp -r *_output/ backup_2025-12-24/
```

---

## 📞 サポート

問題が発生した場合：
1. [トラブルシューティング](USAGE_GUIDE.md#5-トラブルシューティング)を確認
2. エラーメッセージで検索
3. 可視化画像で結果を確認

---

**プロジェクトルートに戻る**: [../README.md](../README.md)

**最終更新**: 2025-12-24  
**プロジェクト**: QPI_omni  
**著者**: AI Assistant
