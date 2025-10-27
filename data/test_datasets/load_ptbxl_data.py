import os
import wfdb
import pandas as pd
import numpy as np

def load_ptbxl_dataset(base_path="data/test_datasets/ptbxl", sampling_rate="100"):
    """
    Load PTB-XL ECG data from WFDB format (.dat + .hea) instead of .npy
    """
    csv_path = os.path.join(base_path, "ptbxl_database.csv")
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded PTB-XL metadata: {len(df)} records found")

    # Filter only files that exist locally
    df = df[df["filename_lr"].apply(lambda x: os.path.exists(os.path.join(base_path, x + ".dat")))]
    df = df.sample(min(20, len(df)), random_state=42)  # load only a few samples

    data, labels = [], []

    for _, row in df.iterrows():
        file_base = os.path.join(base_path, row["filename_lr"])
        try:
            # Read the WFDB record (use wfdb library)
            record = wfdb.rdrecord(file_base)
            ecg = record.p_signal[:, 0]  # take lead I (first channel)
            data.append(ecg)
            labels.append(list(eval(row["scp_codes"]).keys())[0])  # take first diagnostic label
        except Exception as e:
            print(f"⚠️ Could not load {file_base}: {e}")

    print(f"✅ Loaded {len(data)} ECG signals from PTB-XL")
    return np.array(data, dtype=object), labels
