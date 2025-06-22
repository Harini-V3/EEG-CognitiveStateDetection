import pandas as pd
import joblib
from collections import Counter
import matplotlib.pyplot as plt
from streamlit_autorefresh import st_autorefresh
import streamlit as st

st.set_page_config(page_title="EEG Dashboard", layout="wide")

st.markdown("<h1 style='text-align: center;'> EEG Real-Time Emotion & Concentration Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

students = {
    "Kamesh": pd.read_csv("kamesh_eeg_features_extracted.csv"),
    "Kanish": pd.read_csv("kanish_eeg_feature_extracted.csv")
}
for name in students:
    if 'label' in students[name].columns:
        students[name] = students[name].drop(columns=['label'])

model_con = joblib.load("model_concentration.pkl")
model_emo = joblib.load("model_emotion.pkl")
 
con_map = {"1": "Concentrated", "des": "Distracted"}
emo_map = {
    "happy": "Happy",
    "sad": "Sad",
    "angry": "Angry",
    "relaxed": "Relaxed"
}
 
interval = 5000   
count = st_autorefresh(interval=interval, limit=None, key="refresh_counter")
frame_index = count % max(len(df) for df in students.values())
 
st.markdown("<h3 style='text-align: center;'> Live Stream</h3>", unsafe_allow_html=True)
cols = st.columns(len(students))
live_results = []

for idx, (name, df) in enumerate(students.items()):
    with cols[idx]:
        st.markdown(f"  {name}")
        if frame_index < len(df):
            row = df.iloc[frame_index:frame_index + 1]
            con_pred = model_con.predict(row)[0]
            emo_pred = model_emo.predict(row)[0]
            con_label = con_map.get(str(con_pred), "Unknown")
            emo_label = emo_map.get(str(emo_pred), "Unknown")

            live_results.append({
                "Frame": frame_index + 1,
                "Student": name,
                "Concentration": con_label,
                "Emotion": emo_label
            })

            st.markdown(f"Concentration: {con_label}")
            st.markdown(f"Emotion: {emo_label}")
            if con_label == "Distracted":
                st.error(f"{name} is Distracted!", icon="⚠️")
        else:
            st.success("All frames completed.")

if live_results:
    st.markdown("<h3 style='text-align: center;'> Combined Live Predictions Table</h3>", unsafe_allow_html=True)
    df_live = pd.DataFrame(live_results)
    st.dataframe(df_live, use_container_width=True)

if frame_index + 1 >= max(len(df) for df in students.values()):
    st.markdown("---")
    st.subheader("Final Summary Charts")

    df_final = pd.DataFrame(live_results)

    for name in students:
        st.markdown(f"{name}'s Summary")
        df_student = df_final[df_final["Student"] == name]

        con_count = Counter(df_student["Concentration"])
        emo_count = Counter(df_student["Emotion"])

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].bar(con_count.keys(), con_count.values(), color='deepskyblue')
        ax[0].set_title("Concentration", fontsize=14)
        ax[1].bar(emo_count.keys(), emo_count.values(), color='coral')
        ax[1].set_title("Emotion", fontsize=14)
        st.pyplot(fig)