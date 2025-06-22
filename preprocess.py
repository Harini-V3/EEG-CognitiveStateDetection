import numpy as np
import pandas as pd
import mne
from scipy.signal import butter, filtfilt

file_path = "kanish_con_set2_1.csv" 
df = pd.read_csv(file_path)

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut=0.5, highcut=50, fs=250, order=4):
    """Apply a bandpass filter to EEG data."""
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, data, axis=0)

fs = 250  
filtered_data = apply_bandpass_filter(df.iloc[:, 1:].values, lowcut=0.5, highcut=50, fs=fs)

info = mne.create_info(ch_names=list(df.columns[1:]), sfreq=fs, ch_types="eeg")
raw = mne.io.RawArray(filtered_data.T, info)

ica = mne.preprocessing.ICA(n_components=5, random_state=42)
ica.fit(raw)
raw_ica = ica.apply(raw)

normalized_data = (raw_ica.get_data() - np.mean(raw_ica.get_data(), axis=1, keepdims=True)) / np.std(raw_ica.get_data(), axis=1, keepdims=True)

cleaned_df = pd.DataFrame(normalized_data.T, columns=df.columns[1:])
cleaned_df.to_csv("cleaned_" + file_path, index=False)
print(f" Preprocessed EEG data saved as cleaned_{file_path}")