import matplotlib.pyplot as plt
import pandas as pd
import scienceplots  # SciencePlotを使用

# =========================================
# === 設定 ===
# =========================================
plt.style.use('science')
plt.rcParams["font.size"] = 18

filepaths = [
    "/Users/kitak/Desktop/251105_QPI_results/pos1_1_Results_1.csv",
    "/Users/kitak/Desktop/251105_QPI_results/pos1_1_Results_2.csv",
    "/Users/kitak/Desktop/251105_QPI_results/pos1_2_Results.csv",
    #"/Users/kitak/Desktop/251105_QPI_results/pos1_3_Results.csv", #最初が変
    "/Users/kitak/Desktop/251105_QPI_results/pos2_1_Results_1.csv",
    "/Users/kitak/Desktop/251105_QPI_results/pos2_2_Results.csv",
    "/Users/kitak/Desktop/251105_QPI_results/pos2_3_Results_1.csv",
    "/Users/kitak/Desktop/251105_QPI_results/pos2_4_Results.csv"
]
highlight_series = [
    #"/Users/kitak/Desktop/251105_QPI_results/pos1_1_Results_1.csv",
    "/Users/kitak/Desktop/251105_QPI_results/pos1_1_Results_2.csv",
    "/Users/kitak/Desktop/251105_QPI_results/pos1_2_Results.csv",
    #"/Users/kitak/Desktop/251105_QPI_results/pos1_3_Results.csv", #最初が変
    #"/Users/kitak/Desktop/251105_QPI_results/pos2_1_Results_1.csv",
    "/Users/kitak/Desktop/251105_QPI_results/pos2_2_Results.csv",
    #"/Users/kitak/Desktop/251105_QPI_results/pos2_3_Results_1.csv",
    "/Users/kitak/Desktop/251105_QPI_results/pos2_4_Results.csv"
]

# =========================================
# === ユーティリティ関数 ===
# =========================================
def get_nearest_slice(df, target):
    """
    指定したSlice番号が存在しない場合、直前のSlice番号を返す。
    """
    if target in df['Slice'].values:
        return target
    else:
        prev_candidates = df[df['Slice'] < target]['Slice']
        if len(prev_candidates) > 0:
            return prev_candidates.max()  # 直前のスライス
        else:
            return df['Slice'].min()  # 最初のスライスを返す


# =========================================
# === Mean補正関数（柔軟スライス対応） ===
# =========================================
def correct_mean_discontinuity(df):
    df = df.copy()

    # --- 1. 1150〜1435 ---
    before_1 = get_nearest_slice(df, 1149)
    after_1  = get_nearest_slice(df, 1150)
    #delta1 = df.loc[df['Slice'] == before_1, 'Mean'].values[0] - df.loc[df['Slice'] == after_1, 'Mean'].values[0]
    delta1 = 0.11
    df.loc[df['Slice'] >= after_1, 'Mean'] += delta1

    # --- 2. 1436〜2013 ---
    before_2 = get_nearest_slice(df, 1435)
    after_2  = get_nearest_slice(df, 1436)
    #delta2 = df.loc[df['Slice'] == before_2, 'Mean'].values[0] - df.loc[df['Slice'] == after_2, 'Mean'].values[0]
    delta2 = 0.01
    df.loc[df['Slice'] >= after_2, 'Mean'] += delta2

    # --- 3. 2014以降 ---
    before_3 = get_nearest_slice(df, 2013)
    after_3  = get_nearest_slice(df, 2014)
    #delta3 = df.loc[df['Slice'] == before_3, 'Mean'].values[0] - df.loc[df['Slice'] == after_3, 'Mean'].values[0]
    delta3 = -0.12
    df.loc[df['Slice'] >= after_3, 'Mean'] += delta3

    return df


# =========================================
# === データ読み込みと補正 ===
# =========================================
area_data = []
mean_data = []

for filepath in filepaths:
    data = pd.read_csv(filepath)
    data = data.sort_values(by='Slice').reset_index(drop=True)
    data['Time'] = data['Slice'] * (1 / 12)  # 時間[h]
    data['Area_um2'] = data['Area'] * (140 / 648) ** 2  # 面積変換

    # === Mean補正 ===
    data = correct_mean_discontinuity(data)

    # === 結果を格納 ===
    area_data.append((data['Time'], data['Area_um2'], filepath))
    mean_data.append((data['Time'], data['Mean'], filepath))

# =========================================
# === プロット ===
# =========================================
fig = plt.figure(figsize=(12, 6))

# --- Areaプロット ---
ax1 = plt.subplot(2, 1, 1)
for time, area, filepath in area_data:
    label = filepath.split('_')[-1].replace('.csv', '')
    if any(hl in filepath for hl in highlight_series):
        ax1.plot(time, area, linewidth=1, label=label, 
                 #color="orange"
                )
    else:
        ax1.plot(time, area, linewidth=0.1, color='gray')
ax1.set_xlabel('Time [h]')
ax1.set_ylabel('Cell Area [$\mu$m$^2$]')
ax1.axvline(x=1145/12, color='gray', linestyle='--')
ax1.axvline(x=1435/12, color='gray', linestyle='--')
ax1.axvline(x=2014/12, color='gray', linestyle='--')
ax1.legend(fontsize=8, loc='upper left', ncol=2)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(top=False, right=False)
ax1.set_ylim(0,30)

# --- Meanプロット ---
ax2 = plt.subplot(2, 1, 2)
for time, mean, filepath in mean_data:
    label = filepath.split('_')[-1].replace('.csv', '')
    if any(hl in filepath for hl in highlight_series):
        ax2.plot(time, mean, linewidth=0.5, label=label, 
                 #color="orange"
                )
    else:
        ax2.plot(time, mean, linewidth=0.1, color='gray')
ax2.set_xlabel('Time [h]')
ax2.set_ylabel('mean RI [rad]')
ax2.axvline(x=1145/12, color='gray', linestyle='--')
ax2.axvline(x=1435/12, color='gray', linestyle='--')
ax2.axvline(x=2014/12, color='gray', linestyle='--')
ax2.legend(fontsize=12, loc='upper left', ncol=2)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.tick_params(top=False, right=False)

plt.tight_layout()
plt.show()
