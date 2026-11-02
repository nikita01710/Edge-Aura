
import streamlit as st
import librosa
import numpy as np
import joblib
import io
import pandas as pd
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Edge-Aura",
    page_icon="🔊",
    layout="wide"
)

# Load trained SVM model
@st.cache_resource
def load_model():
    return joblib.load("edge_aura_model.pkl")

saved_data = load_model()
model = saved_data["model"]
label_encoder = saved_data["label_encoder"]


# Feature extraction (same as training)
def extract_features(audio_file):

    y, sr = librosa.load(
        audio_file,
        sr=22050,
        duration=5
    )

    if len(y) == 0:
        raise ValueError("Audio file is empty.")

    features = []

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=40
    )

    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    for feature in [
        librosa.feature.spectral_centroid(y=y, sr=sr),
        librosa.feature.spectral_bandwidth(y=y, sr=sr),
        librosa.feature.spectral_rolloff(y=y, sr=sr),
        librosa.feature.zero_crossing_rate(y)
    ]:

        features.extend([
            np.mean(feature),
            np.std(feature)
        ])

    return np.array(features).reshape(1, -1)


# Prediction function
def predict_audio(audio_file):

    features = extract_features(audio_file)

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    category = label_encoder.inverse_transform(
        [prediction]
    )[0]

    confidence = float(np.max(probabilities) * 100)

    # Preliminary low-confidence warning
    status = "Uncertain" if confidence < 60 else "Recognized"

    # Top 3 predictions
    top_indices = np.argsort(probabilities)[::-1][:3]

    top_predictions = [
        (
            label_encoder.classes_[i],
            round(float(probabilities[i] * 100), 2)
        )
        for i in top_indices
    ]

    return category, confidence, probabilities, top_predictions, status


# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

if "result" not in st.session_state:
    st.session_state.result = None


# Dashboard UI
st.title("🔊 Edge-Aura")
st.subheader("Intelligent Environmental Sound Detection")

st.write(
    "Detect and classify environmental sounds using "
    "Machine Learning and audio signal processing."
)

st.caption(
    "Powered by Support Vector Machine (SVM), "
    "MFCC and spectral features."
)

st.divider()

# Audio input options
st.header("Sound Detection")

tab1, tab2 = st.tabs([
    "Upload Audio",
    "Record Audio"
])


# Upload audio
with tab1:

    uploaded_file = st.file_uploader(
        "Upload an audio recording",
        type=["wav", "mp3", "ogg", "flac"]
    )

    if uploaded_file is not None:

        st.audio(uploaded_file)

        if st.button("Detect Sound", key="upload"):

            try:
                category, confidence, probabilities, top_predictions, status = predict_audio(
                    io.BytesIO(uploaded_file.getvalue())
                )

                st.session_state.result = (
                    category, confidence, probabilities,
                    top_predictions, status
                )

                st.session_state.history.append({
                    "Sound": category.replace("_", " ").title(),
                    "Confidence": round(confidence, 2),
                    "Status": status
                })

            except Exception as e:
                st.error(f"Audio processing failed: {e}")


# Microphone input
with tab2:

    recorded_audio = st.audio_input(
        "Record an environmental sound"
    )

    if recorded_audio is not None:

        st.audio(recorded_audio)

        if st.button("Classify Recording", key="record"):

            try:
                category, confidence, probabilities, top_predictions, status = predict_audio(
                    io.BytesIO(recorded_audio.getvalue())
                )

                st.session_state.result = (
                    category, confidence, probabilities,
                    top_predictions, status
                )

                st.session_state.history.append({
                    "Sound": category.replace("_", " ").title(),
                    "Confidence": round(confidence, 2),
                    "Status": status
                })

            except Exception as e:
                st.error(f"Audio processing failed: {e}")


# Display prediction
if st.session_state.result is not None:

    category, confidence, probabilities, top_predictions, status = (
        st.session_state.result
    )

    st.divider()
    st.header("Detection Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Predicted Sound",
            category.replace("_", " ").title()
        )

    with col2:
        st.metric(
            "Prediction Probability",
            f"{confidence:.2f}%"
        )

    if status == "Uncertain":
        st.warning(
            "Prediction is uncertain. "
            "Try uploading a clearer recording."
        )
    else:
        st.success("Sound classification completed.")

    # Top 3 predictions
    st.subheader("Top 3 Predictions")

    for sound, probability in top_predictions:
        st.write(
            f"**{sound.replace('_', ' ').title()}** "
            f"— {probability}%"
        )

    # Probability chart
    st.subheader("Category-wise Prediction Probability")

    probability_df = pd.DataFrame({
        "Sound Category": label_encoder.classes_,
        "Probability (%)": probabilities * 100
    }).sort_values(
        "Probability (%)",
        ascending=True
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.barh(
        probability_df["Sound Category"],
        probability_df["Probability (%)"]
    )

    ax.set_xlabel("Probability (%)")
    ax.set_ylabel("Sound Category")
    ax.set_xlim(0, 100)
    ax.set_title("Edge-Aura - Sound Classification")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# Detection history
st.divider()
st.header("Detection History")

if st.session_state.history:

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(
        history_df,
        use_container_width=True
    )

    if st.button("Clear History"):
        st.session_state.history = []
        st.session_state.result = None
        st.rerun()

else:
    st.info("No sound detections yet.")


# Footer
st.divider()

st.caption(
    "Edge-Aura | Environmental Sound Classification "
    "using SVM and MFCC features"
)
