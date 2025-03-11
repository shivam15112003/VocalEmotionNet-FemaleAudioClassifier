# 🎙️ Audio Emotion Detection

## 📌 Overview
This project builds an **AI-powered emotion detection system for female audio recordings**, using **deep learning and signal processing**. The model classifies emotions like **happy, sad, angry, neutral**, etc., from voice inputs.

## 🚀 Features
- **Real-time audio emotion detection** using deep learning.
- **Pre-trained VGG16** model for feature extraction.
- Supports **audio recording & file uploads** for emotion classification.
- **Tkinter GUI** for a user-friendly experience.
- Uses **Librosa** for feature extraction (MFCC & Mel Spectrogram).

## 🔧 Technologies Used
- **Python**
- **TensorFlow / Keras** (for deep learning and model training)
- **Librosa** (for audio feature extraction)
- **OpenCV** (for signal processing)
- **Tkinter** (for GUI-based user interface)
- **Pyaudio & Wave** (for audio recording and handling)

## 📂 Installation & Usage
1. Clone the repository:
   ```sh
   git clone https://github.com/shivam15112003/emotion_detection_femalevoice.git
   cd emotion_detection_femalevoice
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```sh
   jupyter notebook emotion_detection_femalevoice_model.ipynb
   ```
4. Use the GUI to:
   - **Upload audio files** for analysis.
   - **Record live audio** and detect emotions.
   - **View results** in a pop-up message.

## 📈 Model Training Details
- Uses **VGG16 pre-trained model** with additional dense layers.
- Extracts **MFCC and Mel Spectrogram features** from audio.
- Achieved **94% accuracy** on the validation dataset.

## 📈 Future Enhancements
- Deploy as a **web-based application**.
- Expand dataset for **better generalization**.
- Optimize for **real-time mobile inference**.
