# Edge-Aura 🔊
### Environmental Sound Classification System

Edge-Aura is a machine learning-based application that identifies environmental sounds from audio recordings using an SVM classifier.

## Features
- Classifies 8 environmental sound categories.
- Supports audio file uploads.
- Supports microphone-based audio input.
- Displays predicted sound category and prediction probabilities.
- Interactive interface built using Streamlit.

## Supported Sound Categories
1. Siren
2. Car Horn
3. Clock Alarm
4. Glass Breaking
5. Door Wood Knock
6. Crying Baby
7. Footsteps
8. Dog

## Technology Stack
- Python
- Librosa
- NumPy
- Pandas
- Scikit-learn
- SVM (RBF Kernel)
- Streamlit
- Joblib

## Model Performance
- Dataset: ESC-50 (selected 8 categories)
- Total audio samples: 320
- Extracted features: 88 per audio sample
- Test samples: 64
- Test accuracy: 90.62%

## Project Structure

```text
Edge-Aura/
├── app.py
├── edge_aura_model.pkl
├── requirements.txt
└── README.md
```

## Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/nikita01710/Edge-Aura
cd Edge-Aura
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

## Limitations
The current model recognizes only the eight supported sound categories. Other sounds may be incorrectly classified.

## Future Scope
- Expand the number of sound categories.
- Explore deep learning-based audio classification.
- Improve recognition of unfamiliar environmental sounds.

**Project:** Edge-Aura — Environmental Sound Classification System

<img width="1267" height="579" alt="image" src="https://github.com/user-attachments/assets/85b6e8b3-06aa-4e5f-9655-8dbd23ee6186" />

<img width="1273" height="563" alt="image" src="https://github.com/user-attachments/assets/23d25502-8964-4c25-8cca-d34f9f1a504a" />

<img width="1306" height="416" alt="image" src="https://github.com/user-attachments/assets/f1a6e73b-7d2a-4bb9-a109-5da0898aa801" />


