# %%
import matplotlib.pyplot as plt
import pandas as pd
import scienceplots  # Use SciencePlots
from figure_logger import setup_autosave
setup_autosave()

# =========================================
# === Settings ===
# =========================================
plt.style.use('science')
plt.rcParams["font.size"] = 18

filepaths = [
    "/Users/kitak/Desktop/251105_QPI_results/pos1_1_Results_1.csv",
    "/Users/kitak/Desktop/251105_QPI_results/pos1_1_Results_2.csv",
    "/Users/kitak/Desktop/251105_QPI_results/pos1_2_Results.csv",
    #"/Users/kitak/Desktop/251105_QPI_results/pos1_3_Results.csv", #first frame is anomalous
    "/Users/kitak/Desktop/251105_QPI_results/pos2_1_Results_1.csv",
    "/Users/kitak/Desktop/251105_QPI_results/pos2_2_Results.csv",
    "/Users/kitak/Desktop/251105_QPI_results/pos2_3_Results_1.csv",
    "/Users/kitak/Desktop/251105_QPI_results/pos2_4_Results.csv"
]
highlight_series = [
    #"/Users/kitak/Desktop/251105_QPI_results/pos1_1_Results_1.csv",
    "/Users/kitak/Desktop/251105_QPI_results/pos1_1_Results_2.csv",
    "/Users/kitak/Desktop/251105_QPI_results/pos1_2_Results.csv",
    #"/Users/kitak/Desktop/251105_QPI_results/pos1_3_Results.csv", #first frame is anomalous
    #"/Users/kitak/Desktop/251105_QPI_results/pos2_1_Results_1.csv",
    "/Users/kitak/Desktop/251105_QPI_results/pos2_2_Results.csv",
    #"/Users/kitak/Desktop/251105_QPI_results/pos2_3_Results_1.csv",
    "/Users/kitak/Desktop/251105_QPI_results/pos2_4_Results.csv"
]

# =========================================
# === Utility functions ===
# =========================================
def get_nearest_slice(df, target):
    """
    Return the specified slice number if it exists, otherwise return the nearest preceding slice number.
    """
    if target in df['Slice'].values:
        return target
    else:
        prev_candidates = df[df['Slice'] < target]['Slice']
        if len(prev_candidates) > 0:
            return prev_candidates.max()  # nearest preceding slice
        else:
            return df['Slice'].min()  # return the first slice


# =========================================
# === Mean correction function (flexible slice handling) ===
# =========================================
def correct_mean_discontinuity(df):
    df = df.copy()

    # --- 1. 1150 to 1435 ---
    before_1 = get_nearest_slice(df, 1149)
    after_1  = get_nearest_slice(df, 1150)
    #delta1 = df.loc[df['Slice'] == before_1, 'Mean'].values[0] - df.loc[df['Slice'] == after_1, 'Mean'].values[0]
    delta1 = 0.11
    df.loc[df['Slice'] >= after_1, 'Mean'] += delta1

    # --- 2. 1436 to 2013 ---
    before_2 = get_nearest_slice(df, 1435)
    after_2  = get_nearest_slice(df, 1436)
    #delta2 = df.loc[df['Slice'] == before_2, 'Mean'].values[0] - df.loc[df['Slice'] == after_2, 'Mean'].values[0]
    delta2 = 0.01
    df.loc[df['Slice'] >= after_2, 'Mean'] += delta2

    # --- 3. 2014 onwards ---
    before_3 = get_nearest_slice(df, 2013)
    after_3  = get_nearest_slice(df, 2014)
    #delta3 = df.loc[df['Slice'] == before_3, 'Mean'].values[0] - df.loc[df['Slice'] == after_3, 'Mean'].values[0]
    delta3 = -0.12
    df.loc[df['Slice'] >= after_3, 'Mean'] += delta3

    return df


# =========================================
# === Data loading and correction ===
# =========================================
area_data = []
mean_data = []

for filepath in filepaths:
    data = pd.read_csv(filepath)
    data = data.sort_values(by='Slice').reset_index(drop=True)
    data['Time'] = data['Slice'] * (1 / 12)  # time [h]
    data['Area_um2'] = data['Area'] * (140 / 648) ** 2  # area conversion

    # === Mean correction ===
    data = correct_mean_discontinuity(data)

    # === Store results ===
    area_data.append((data['Time'], data['Area_um2'], filepath))
    mean_data.append((data['Time'], data['Mean'], filepath))

# =========================================
# === Plot ===
# =========================================
fig = plt.figure(figsize=(12, 6))

# --- Area plot ---
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

# --- Mean plot ---
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
# %%
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.size"] = 18

filepath = r"C:\Users\QPI\Desktop\Results.csv"

# =========================================
# === Mean correction function ===
# =========================================
def get_nearest_slice(df, target):
    if target in df['Slice'].values:
        return target
    prev = df[df['Slice'] < target]['Slice']
    return prev.max() if len(prev) > 0 else df['Slice'].min()

def correct_mean_discontinuity(df):
    df = df.copy()
    # delta values are fixed as specified
    #df.loc[df['Slice'] >= get_nearest_slice(df, 1150), 'Mean'] += 0.11
    #df.loc[df['Slice'] >= get_nearest_slice(df, 1436), 'Mean'] += 0.1
    #df.loc[df['Slice'] >= get_nearest_slice(df, 2014), 'Mean'] += -0.12
    return df

# =========================================
# === Data loading ===
# =========================================
data = pd.read_csv(filepath)
data = data.sort_values(by="Slice").reset_index(drop=True)

data["Time"] = data["Slice"] / 12           # time [h]
data["Area_um2"] = data["Area"] * (140 / 648)**2   # area conversion

data = correct_mean_discontinuity(data)

# =========================================
# === Plot ===
# =========================================
fig = plt.figure(figsize=(12, 6))

# --- Area ---
ax1 = plt.subplot(2, 1, 1)
ax1.plot(data["Time"], data["Area_um2"], linewidth=1.2)
ax1.set_xlabel("Time [h]")
ax1.set_ylabel("Area [$\mu$m$^2$]")
ax1.axvline(x=1145/12, linestyle="--", color="gray")
ax1.axvline(x=1435/12, linestyle="--", color="gray")
ax1.axvline(x=2014/12, linestyle="--", color="gray")
ax1.set_ylim(0, 30)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# --- Mean ---
ax2 = plt.subplot(2, 1, 2)
ax2.plot(data["Time"], data["Mean"], linewidth=1.2)
ax2.set_xlabel("Time [h]")
ax2.set_ylabel("Phase [rad]")
ax2.axvline(x=1145/12, linestyle="--", color="gray")
ax2.axvline(x=1435/12, linestyle="--", color="gray")
ax2.axvline(x=2014/12, linestyle="--", color="gray")
ax2.set_ylim(0.8,1.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

# %%
