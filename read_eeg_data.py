import pandas as pd
import joblib
import streamlit as st

#  Page config
st.set_page_config(page_title="EEG Cognitive Monitoring", layout="wide")
st.title(" Student Cognitive Monitoring (Feature-Based)")

# Load models
model_con = joblib.load("model_concentration.pkl")
model_emo = joblib.load("model_emotion.pkl")

# Mapping
con_map = {"1": "Concentrated", "des": "Distracted"}
emo_map = {
    "happy": "Happy",
    "sad": "Sad",
    "angry": "Angry",
    "relaxed": "Relaxed"
}

# Pre-loaded student feature files
students = {
    "Kanish": pd.read_csv("kanish_eeg_feature_extracted.csv"),
    "Kamesh": pd.read_csv("kamesh_eeg_features_extracted.csv")
}

# Process and display
for name, features_df in students.items():
    st.markdown(f"###  {name}'s Prediction")

    con_preds = model_con.predict(features_df)
    emo_preds = model_emo.predict(features_df)

    results = pd.DataFrame({
        "Concentration": [con_map.get(str(p), "Unknown") for p in con_preds],
        "Emotion": [emo_map.get(str(p), "Unknown") for p in emo_preds]
    })

    st.dataframe(results)

    distracted_count = results["Concentration"].value_counts().get("Distracted", 0)
    if distracted_count > 0:
        st.warning(f" {distracted_count} instances of distraction detected for {name}")
    else:
        st.success(f" {name} remained focused throughout the session.")