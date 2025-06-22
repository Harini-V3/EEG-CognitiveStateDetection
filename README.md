# EEG-CognitiveStateDetection

PROJECT OVERVIEW : 
This project is about creating a cognitive and emotional state detection system using EEG data from OpenBCI hardware. By placing the electrodes on the specific areas related to attention and emotion i.e., F1, F2 (specifically for concentration) and T3, T4, T5, T6 (for emotional state analysis) we were able to observe brain signals and also captured the patterns for each individuals. The system uses machine learning (ML) models to distinguish between these cognitive and emotional states into five categories like concentration, distraction, happy, sad and angry.

FEATURES : 
Real-time EEG data collection and preprocessing, Machine learning model training and evaluation, Classification of cognitive states (e.g., focused, distracted), Streamlit dashboard to visualize predictions and accuracy metrics.

REQUIREMENTS : 
Programming language : Python, Libraries: Scikit-learn, Pandas, NumPy, Matplotlib, Streamlit, Tools: Google Colab, VS Code, Hardware: EEG device or any compatible EEG headband, electrodes, Cyton biosensing board, Open BCI software (EEG signal acquisition)

MODEL DETAILS : 
Input features : Statistical and frequency-based features extracted from EEG signals
Used Random Forest, KNN and SVM  model for training data and acquired a better accuracy with random forest model.

Brain Signals in Time series
![eeg_signal](https://github.com/user-attachments/assets/7dec1bfd-431f-49b9-a4a4-9dc97d05db8a)
Converting to csv format
![eeg_csv_format](https://github.com/user-attachments/assets/2829a6db-3ecf-415f-bdea-42be7da6eb3e)
Dashboard 
![dashboard](https://github.com/user-attachments/assets/eec30fe7-a867-4090-8a01-8471685cccf6)

