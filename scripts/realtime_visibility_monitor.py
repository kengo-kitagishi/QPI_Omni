import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from qpi import QPIParameters, calc_visibility, _get_dc_ac, _get_visibility
from threading import Lock

WATCH_FOLDER = r"d:\AquisitionData\Kitagishi\basler_image_seq"
WAVELENGHTH=658e-9
NA = 0.95
PIXELSIZE = 3.45e-6/40
OFFAXIS_CENTER = (1642,443)
CROP_REGION = (0,2048,208,2256)

VMIN= .75
VMAX = 0.8
UPDATE_INTERVAL = 500
HIST_BINS=100
HIST_RANGE = (0.3,0.95)

latest_visibility = None
latest_dc = None
latest_ac = None
latest_filename = None
data_lock =Lock()
params = None

class ImageHandlar(FileSystemEventHandler):
    def on_created(self,event):
        global latest_visibility, latest_dc, latest_ac, latest_filename

        if event.is_directory:
            return
        if not (event.src_path.lower().endswith('.tif') or
                event.src_path.lower().endswith('.tiff')):
            return
        time.sleep(0.1)

        try:
            img = np.array(Image.open(event.src_path))

            if len(img.shape)==3:
                img = img[:,:,0]

            if CROP_REGION is not None:
                img = img[CROP_REGION[0]:CROP_REGION[1],
                          CROP_REGION[2]:CROP_REGION[3]]
            print(f"{img.shape}, dtype:{img.dtype}")
            global params
            if params is None:
                params = QPIParameters(
                    wavelength=WAVELENGHTH,
                    NA=NA,
                    img_shape=img.shape,
                    pixelsize=PIXELSIZE,
                    offaxis_center = OFFAXIS_CENTER,
                )
            
            # dc, acを計算
            dc, ac = _get_dc_ac(img, params)
            vis = _get_visibility(dc, ac)
            
            # 絶対値を取得
            dc_abs = np.abs(dc)
            ac_abs = np.abs(ac)

            with data_lock:
                latest_visibility = vis
                latest_dc = dc_abs
                latest_ac = ac_abs
                latest_filename = Path(event.src_path).name

            print(f"mean visibility: {np.mean(vis):.4f}")

        except Exception as e:
            print(f"error : {event.src_path}")
            print(f" {str(e)}")
    
def update_plot(frame):
    global latest_visibility, latest_dc, latest_ac, latest_filename

    with data_lock:
        if latest_visibility is not None:
            # 全てのaxesをクリア
            ax_vis.clear()
            ax_ac.clear()
            ax_dc.clear()
            ax_hist.clear()
            
            # Visibility (左上)
            ax_vis.imshow(latest_visibility, cmap="viridis", vmin=VMIN, vmax=VMAX)
            ax_vis.set_title(f"Visibility\n{latest_filename}", fontsize=10)
            ax_vis.axis("off")
            mean_vis = np.mean(latest_visibility)
            ax_vis.text(0.02, 0.98, f"Mean: {mean_vis:.4f}",
                       transform=ax_vis.transAxes, fontsize=9,
                       verticalalignment="top",
                       bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            
            # 干渉光 |ac| (右上)
            ax_ac.imshow(latest_ac, cmap="hot")
            ax_ac.set_title("Interference light |ac|", fontsize=10)
            ax_ac.axis("off")
            mean_ac = np.mean(latest_ac)
            ax_ac.text(0.02, 0.98, f"Mean: {mean_ac:.2e}",
                      transform=ax_ac.transAxes, fontsize=9,
                      verticalalignment="top",
                      bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            
            # 非干渉光 |dc| (左下)
            ax_dc.imshow(latest_dc, cmap="hot")
            ax_dc.set_title("Non-interference light |dc|", fontsize=10)
            ax_dc.axis("off")
            mean_dc = np.mean(latest_dc)
            ax_dc.text(0.02, 0.98, f"Mean: {mean_dc:.2e}",
                      transform=ax_dc.transAxes, fontsize=9,
                      verticalalignment="top",
                      bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            
            # Visibilityヒストグラム (右下)
            vis_float = latest_visibility.flatten()
            ax_hist.hist(vis_float, bins=HIST_BINS, range=HIST_RANGE,
                        color="steelblue", edgecolor="black", alpha=0.7)
            ax_hist.set_xlabel("Visibility", fontsize=10)
            ax_hist.set_ylabel("Pixel Count", fontsize=10)
            ax_hist.set_title("Visibility Histogram", fontsize=10)
            ax_hist.text(0.02, 0.98, f"Mean: {mean_vis:.4f}",
                        transform=ax_hist.transAxes, fontsize=9,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            
def main():
    global ax_vis, ax_ac, ax_dc, ax_hist

    print("="*70)
    print("Realtime Visibility, Interference & Non-interference Monitor")
    print("="*70)

    if not os.path.exists(WATCH_FOLDER):
        print(f"\n フォルダが見つかりません")
        print(f"{WATCH_FOLDER}")
        return

    event_hander = ImageHandlar()
    observer = Observer()
    observer.schedule(event_hander,WATCH_FOLDER,recursive=False)
    observer.start()
    print(f"file monitoring start")

    # 2×2グリッドの作成
    fig, ((ax_vis, ax_ac), (ax_dc, ax_hist)) = plt.subplots(2, 2, figsize=(14, 12))
    plt.tight_layout()

    ani = FuncAnimation(fig, update_plot, interval=UPDATE_INTERVAL,
                        cache_frame_data=False)
    try:
        plt.show()
    except KeyboardInterrupt:
        print("\n\n中断されました")
    finally:
        observer.stop()
        observer.join()
        print("停止しました")

if __name__ == "__main__":
    main()