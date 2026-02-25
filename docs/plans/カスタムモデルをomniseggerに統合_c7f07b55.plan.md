---
name: カスタムモデルをOmniseggerに統合
overview: カスタムトレーニングされたOmniposeモデル（nchan=1, nclasses=3）をOmniseggerのMATLABワークフローに統合し、既存の追跡・解析機能（蛍光解析、カイモグラフなど）を使えるようにします。
todos:
  - id: add_const_fields
    content: loadConstants.mにOmnipose設定フィールドを追加
    status: pending
  - id: modify_gen_command
    content: genOmniposeCommand関数を動的に変更
    status: pending
  - id: update_function_calls
    content: genOmniposeCommandの呼び出しにCONSTを渡す
    status: pending
  - id: update_processexp
    content: processExp.mにカスタムモデル設定例を追加
    status: pending
  - id: update_docs
    content: ドキュメントを更新・作成
    status: pending
---

# カスタムOmniposeモデルをOmniseggerに統合

## 概要

現在、OmniseggerはデフォルトでOmniposeの `bact_phase_omni` モデルを使用していますが、これをカスタムモデルに対応させるため、モデルパスとセグメンテーションパラメータを柔軟に設定できるように修正します。

## 実装内容

### 1. CONST構造体にOmnipose設定を追加

[`omnisegger/SuperSegger-master/settings/loadConstants.m`](omnisegger/SuperSegger-master/settings/loadConstants.m) にOmnipose関連の設定フィールドを追加します:

- `CONST.omnipose.model_path`: モデルのパス（デフォルト: `'bact_phase_omni'`）
- `CONST.omnipose.diameter`: 細胞の直径（デフォルト: `30`）
- `CONST.omnipose.flow_threshold`: フロー閾値（デフォルト: `0`）
- `CONST.omnipose.mask_threshold`: マスク閾値（デフォルト: `1`）
- `CONST.omnipose.cluster`: DBscanクラスタリング使用（デフォルト: `true`）
- `CONST.omnipose.exclude_on_edges`: エッジの細胞を除外（デフォルト: `true`）
- `CONST.omnipose.use_gpu`: GPU使用（デフォルト: `false`）

### 2. genOmniposeCommand関数を動的に変更

[`omnisegger/SuperSegger-master/batch/BatchSuperSeggerOpti.m`](omnisegger/SuperSegger-master/batch/BatchSuperSeggerOpti.m) の `genOmniposeCommand` 関数を修正し、`CONST`構造体からパラメータを読み取ってコマンドを動的に生成するようにします。現在のハードコードされた実装（line 436-444）を、柔軟な実装に置き換えます。

### 3. CONST構造体を関数に渡す

`genOmniposeCommand` の呼び出し箇所（line 331, 362）を修正し、`CONST`を引数として渡すようにします。

### 4. processExpテンプレートを更新

[`omnisegger/SuperSegger-master/batch/processExp.m`](omnisegger/SuperSegger-master/batch/processExp.m) にカスタムモデルの設定例を追加します:

```matlab
%% Omnipose Model Settings (Optional)
% カスタムモデルを使用する場合は、以下のパラメータを設定してください
% デフォルトモデル（bact_phase_omni, bact_fluor_omni）を使用する場合は不要です

% カスタムモデルの例:
% CONST.omnipose.model_path = 'C:\path\to\your\custom_model';
% CONST.omnipose.diameter = 30;  % 細胞の短軸の長さ（ピクセル）
% CONST.omnipose.flow_threshold = 0.11;
% CONST.omnipose.mask_threshold = 0;
% CONST.omnipose.use_gpu = true;  % GPU使用（環境が対応している場合）
```



### 5. ドキュメントを更新

カスタムモデルの使用方法を説明するドキュメントを作成・更新します:

- [`omnisegger/docs/segmentation_options.md`](omnisegger/docs/segmentation_options.md): カスタムモデルのパラメータ設定方法を追記
- 新規ドキュメント `omnisegger/docs/custom_model_guide.md`: カスタムモデルの詳細な使用方法を説明

## 使用方法（実装後）

1. `processExp.m` を編集:
```matlab
CONST.omnipose.model_path = 'C:\Users\QPI\Desktop\archived_train\...\omni_model_2025_11_05_19_14_41.656097';
CONST.omnipose.diameter = 30;
CONST.omnipose.flow_threshold = 0.11;
CONST.omnipose.mask_threshold = 0;
```




2. 通常通り実行:
```matlab
processExp('C:\path\to\images', 0, 1)
```




3. Omniseggerの全機能（細胞追跡、蛍光解析、カイモグラフなど）がカスタムモデルで動作

## メリット

- カスタムモデルを簡単に切り替え可能
- 既存のOmnisegger機能（追跡、解析、可視化）をそのまま利用