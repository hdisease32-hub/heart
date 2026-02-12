import wfdb
import numpy as np
import matplotlib.pyplot as plt

# Load ECG
record_path = "records100/00000/00001_lr"
record = wfdb.rdrecord(record_path)

signals = record.p_signal
fs = record.fs
leads = record.sig_name

time = np.arange(signals.shape[0]) / fs

fig, axes = plt.subplots(12, 1, figsize=(14, 10), sharex=True)

for i in range(12):
    axes[i].plot(time, signals[:, i], color='black', linewidth=1)

    # Remove ugly ticks
    axes[i].set_yticks([])
    axes[i].set_xticks([])

    # ECG grid
    axes[i].set_xlim(0, time[-1])
    axes[i].set_ylim(-2, 2)

    axes[i].set_xticks(np.arange(0, time[-1], 0.2), minor=True)
    axes[i].set_yticks(np.arange(-2, 2, 0.1), minor=True)

    axes[i].grid(which='minor', color='#ffcccc', linestyle='-', linewidth=0.5)
    axes[i].grid(which='major', color='#ff6666', linestyle='-', linewidth=1)

    # Lead name on left
    axes[i].text(-0.5, 0, leads[i], fontsize=12, verticalalignment='center')

plt.subplots_adjust(hspace=0.2)
plt.suptitle("12-Lead ECG", fontsize=16)
plt.xlabel("Time (seconds)")

plt.savefig("clean_ecg.png", dpi=300, bbox_inches='tight')
plt.show()
