import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

SAMPLING_RATE = 250  
WINDOW_SIZE = 1 * SAMPLING_RATE  
OVERLAP = WINDOW_SIZE // 2  

def extract_features(data):
    features = {
        'mean': np.mean(data),
        'std': np.std(data),
        'skewness': skew(data),
        'kurtosis': kurtosis(data),
        'min': np.min(data),
        'max': np.max(data),
        'median': np.median(data)
    }
    return features

datasets = [
    "cleaned_kanish_con_set2_1.csv",
]

feature_data = []

for file in datasets:
    df = pd.read_csv(file)
    
    eeg_data = df.iloc[:, 1:].values  
    label = file.split('_')[-1].split('.')[0]  

    for start in range(0, len(eeg_data) - WINDOW_SIZE, OVERLAP):
        window = eeg_data[start: start + WINDOW_SIZE]
        window_features = {f"{col}_{key}": val for col, channel in enumerate(window.T)
                           for key, val in extract_features(channel).items()}
        window_features["label"] = label
        feature_data.append(window_features)

feature_df = pd.DataFrame(feature_data)
feature_df.to_csv("kanish_eeg_feature_extracted.csv", index=False)
print("Feature extraction completed! Saved as 'kanish_eeg_feature_extracted.csv'")