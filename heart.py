import wfdb
import matplotlib.pyplot as plt

# Path WITHOUT extension
record_path = "records100/00000/00001_lr"

record = wfdb.rdrecord(record_path)

print("Leads:", record.sig_name)
print("Sampling rate:", record.fs)

wfdb.plot_wfdb(record=record)
plt.show()
