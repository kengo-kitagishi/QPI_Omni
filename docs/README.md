# QPI解析 ドキュメント

このディレクトリには、QPI解析プロジェクトに関するドキュメントとログが含まれています。

## 📂 ディレクトリ構成

```
docs/
├── README.md          # このファイル（ドキュメント索引）
├── workflows/         # 詳細なワークフローログ
│   └── YYYY-MM-DD_description.md
└── notes/             # 簡単なメモ・トラブルシューティング
    └── *.md
```

---

## 📋 ワークフローログ一覧

### 2025年12月

| 日付 | ファイル | 概要 |
|------|---------|------|
| 2025-12-23 | [2025-12-23_timeseries_total_mass.md](workflows/2025-12-23_timeseries_total_mass.md) | Total Mass計算、時系列プロット、Feret径対応、サブピクセルサンプリング、バッチ解析の実装 |

---

## 🔖 主要トピック別索引

### 時系列解析
- [2025-12-23: Total Mass時系列プロット実装](workflows/2025-12-23_timeseries_total_mass.md)

### マスク生成・形状近似
- [2025-12-23: Feret径ベースのマスク生成](workflows/2025-12-23_timeseries_total_mass.md#4-feret径ベースのマスク生成)
- [2025-12-23: サブピクセルサンプリング](workflows/2025-12-23_timeseries_total_mass.md#5-サブピクセルサンプリング)

### バッチ処理
- [2025-12-23: パラメータ網羅的バッチ解析](workflows/2025-12-23_timeseries_total_mass.md#8-バッチ解析の実装)

### ImageJ連携
- [2025-12-23: ROI前処理（縮小＋スムージング）](workflows/2025-12-23_timeseries_total_mass.md#9-imagejでのroi処理)

---

## 🎯 クイックリンク

### 主要スクリプト
- [`24_elip_volume.py`](../scripts/24_elip_volume.py) - メイン解析スクリプト
- [`28_batch_analysis.py`](../scripts/28_batch_analysis.py) - バッチ解析スクリプト

### 計算式・パラメータ
- [物理量計算の原理と計算式](notes/physics_calculations.md) - 完全版（推奨）
- [計算式まとめ](workflows/2025-12-23_timeseries_total_mass.md#計算式まとめ) - クイック版
- [パラメータ一覧](workflows/2025-12-23_timeseries_total_mass.md#パラメータ一覧)

### トラブルシューティング
- [よくある問題と解決策](workflows/2025-12-23_timeseries_total_mass.md#トラブルシューティング)

---

## 📝 新しいログの追加方法

### 1. ワークフローログを追加

```bash
# ファイル作成
code docs/workflows/YYYY-MM-DD_description.md

# 例
code docs/workflows/2025-12-24_gpu_acceleration.md
```

### 2. このREADMEを更新

- 「ワークフローログ一覧」テーブルに行を追加
- 必要に応じて「主要トピック別索引」を更新

### 3. テンプレート

```markdown
# タイトル（YYYY-MM-DD）

## 概要
この日の作業内容を1-2行で説明

## 目次
1. [セクション1](#1-セクション1)
2. [セクション2](#2-セクション2)

---

## 1. セクション1
### 要求
### 実装
### 結果

---

## ファイル一覧
変更したファイル

## パラメータ
使用したパラメータ

---

**End of Log**
最終更新: YYYY-MM-DD HH:MM JST
```

---

## 📌 ノート・メモ一覧

`notes/` ディレクトリの主要ファイル：

- **[quick_reference.md](notes/quick_reference.md)**: よく使うコマンド・パラメータの早見表
- **[physics_calculations.md](notes/physics_calculations.md)**: 物理量計算の原理・仮定・計算式（完全版）

### 新しいメモの追加方法

簡単なメモやTIPSは `notes/` ディレクトリに追加：

```bash
# 例：解析のコツをメモ
code docs/notes/analysis_tips.md

# 例：トラブルシューティング集
code docs/notes/troubleshooting.md
```

---

## 🔍 ログの検索方法

### ファイル名で検索
```bash
ls docs/workflows/*mass*
ls docs/workflows/2025-12*
```

### 内容で検索
```bash
# Total Massに関する記述を検索
grep -r "Total Mass" docs/workflows/

# Feretに関する記述を検索
grep -r "Feret" docs/workflows/
```

---

## 📊 統計情報

- **総ワークフローログ数**: 1
- **最終更新**: 2025-12-23
- **主要機能数**: 5 (Total Mass計算、Feret径、サブピクセルサンプリング、スクリプト統合、バッチ解析)

---

**プロジェクトルートに戻る**: [../README.md](../README.md)
