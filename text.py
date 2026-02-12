import wfdb
import pandas as pd
import ast

# Load CSV
df = pd.read_csv("ptbxl_database.csv")
df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

# Choose ECG id
ecg_id = 39

# Get patient data
patient = df[df["ecg_id"] == ecg_id].iloc[0]
print("Age:", patient.age)
print("Sex:", patient.sex)
print("Report:", patient.report)
print("Diagnosis:", patient.scp_codes)

# Load ECG signal
record_path = "records100/00000/00001_lr"
record = wfdb.rdrecord(record_path)

wfdb.plot_wfdb(record=record)
