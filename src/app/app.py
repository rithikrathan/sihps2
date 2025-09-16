# app.py
import json
import streamlit as st
import subprocess

with open("./config.json", "r") as f:
    config = json.load(f)

st.title("House rent prediction app")
st.sidebar.write("### configs")
st.sidebar.json(config)

option = st.sidebar.radio(
        "Choose step:",
        ("Preliminary steps", "Model Training", "Model Inference")
        )

if option == "Preliminary steps":
    st.subheader("Preliminary Data Visualization")
    viz_type = st.selectbox(
        "Choose visualization type",
        ["Pie Chart", "Bar Chart", "Histogram", "Scatter Plot"]  # you’ll add more
    )
    result = subprocess.run(
        ["python", "preliminary.py", viz_type],
        capture_output=True,
        text=True
    )
    st.text(result.stdout)

elif option == "Model Training":
    st.subheader("Model Training")
    if st.button("Run Training Script"):
        result = subprocess.run(["python", "model_training.py"], capture_output=True, text=True)
        st.text(result.stdout)

elif option == "Model Inference":
    st.subheader("Model Inference")
    if st.button("Run Inference Script"):
        result = subprocess.run(["python", "model_inference.py"], capture_output=True, text=True)
        st.text(result.stdout)

