
import pandas as pd
import matplotlib.pyplot as plt

# ファイルパス
filepath = '/Volumes/asobi-ba/241107_MM/BF50_GFP500_Bin2_1/Pos2/GFP/Roi/241107_MM_Pos2_1.csv'

# データの読み込み
data = pd.read_csv(filepath)

# ラベル列の下4桁を抽出し、整数に変換して新しい列を作成
# data['Label_num'] = data['Label'].apply(lambda x: int(x[-4:]))

# # Label_num列でソート
data = data.sort_values(by='Slice').reset_index(drop=True)
data['Time'] = data['Slice'] * 1/12

# 赤くしたい特定の時間（例: 200, 400, 600, 800 時間に対応する値）
highlight_times = [1144/12, 1200/12,1430/12,1800/12,2230/12]  # 必要に応じて修正
highlight_data = data[data['Time'].isin(highlight_times)]

# Cell Area プロット
plt.figure(figsize=(10, 3))
plt.plot(data['Time'], data['Area'] *(140/648)**2)
plt.scatter(highlight_data['Time'],
            highlight_data['Area'] * (140/648)**2,
            color='red')
plt.xlabel('Time [h]')
plt.ylabel('Cell Area [um^2]')
# plt.axvline(x=1145/12, color='gray', linestyle='--')
# plt.axvline(x=1435/12, color='gray', linestyle='--')
# plt.axvline(x=2020/12, color='gray', linestyle='--')
plt.title('Cell size of a single lineage')
plt.grid(True)
plt.legend()
plt.show()

# Intensity プロット
plt.figure(figsize=(10, 3))
plt.plot(data['Time'], data['Mean'])
plt.scatter(highlight_data['Time'],
            highlight_data['Mean'],
            color='red')
plt.xlabel('Time [h]')
plt.ylabel('Intensity [a.u.]')
# plt.axvline(x=1145/12, color='gray', linestyle='--')
# plt.axvline(x=1435/12, color='gray', linestyle='--')
# plt.axvline(x=2020/12, color='gray', linestyle='--')
plt.title('Intensity of a single lineage')
plt.grid(True)
plt.legend()
plt.show()
