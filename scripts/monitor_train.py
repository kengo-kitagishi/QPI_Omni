"""
Real-time training curve monitor.
Usage: conda run -n omnipose python scripts/monitor_train.py
Reads the train_loss.log written by 08_train.py every 5 seconds and plots it.
"""
import re
import matplotlib.pyplot as plt
import matplotlib.animation as animation

LOG_PATH = r"C:\Users\QPI\Desktop\train\train_loss.log"
INTERVAL_MS = 5000  # polling interval (5 seconds)

_pattern = re.compile(
    r"Train epoch:\s*(\d+).*?<Batch Loss>:\s*([\d.]+).*?<Epoch Loss>:\s*([\d.]+)"
)
_run_separator = re.compile(r"Run started:")

def _parse(path):
    epochs, batch_losses, epoch_losses = [], [], []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if _run_separator.search(line):
                    epochs.clear()
                    batch_losses.clear()
                    epoch_losses.clear()
                    continue
                m = _pattern.search(line)
                if m:
                    epochs.append(int(m.group(1)))
                    batch_losses.append(float(m.group(2)))
                    epoch_losses.append(float(m.group(3)))
    except FileNotFoundError:
        pass
    return epochs, batch_losses, epoch_losses

fig, ax = plt.subplots(figsize=(9, 4))
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Omnipose Training Loss (live)")
line_epoch, = ax.plot([], [], "b-", linewidth=1.5, label="Epoch Loss")
line_batch, = ax.plot([], [], "r-", linewidth=0.8, alpha=0.5, label="Batch Loss")
ax.legend()
status_text = ax.text(0.02, 0.95, "", transform=ax.transAxes,
                      fontsize=9, verticalalignment="top", color="gray")

def _update(frame):
    epochs, batch_losses, epoch_losses = _parse(LOG_PATH)
    if epochs:
        line_epoch.set_data(epochs, epoch_losses)
        line_batch.set_data(epochs, batch_losses)
        ax.relim()
        ax.autoscale_view()
        status_text.set_text(f"epoch {epochs[-1]}  |  loss {epoch_losses[-1]:.4f}")
    else:
        status_text.set_text("waiting for log...")
    return line_epoch, line_batch, status_text

from matplotlib.ticker import MaxNLocator
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

ani = animation.FuncAnimation(fig, _update, interval=INTERVAL_MS, cache_frame_data=False)
plt.tight_layout()
plt.show()
