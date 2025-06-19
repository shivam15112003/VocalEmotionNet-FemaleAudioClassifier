# 🎙️ Female Voice Emotion Detection using VGG16 and GUI

This project implements a deep learning–based audio emotion detection system for **female speakers** using **Mel spectrograms** and **VGG16**, integrated with a real-time GUI for both file upload and voice recording.

---

## 📌 Features

* 🎧 **Supports voice recording** and `.wav/.mp3` upload
* 🔍 **Detects emotions** from female English voice input
* 🧠 **Mel spectrogram** extraction using `librosa`
* 🧩 **Transfer learning** with VGG16 (ImageNet)
* 📊 Achieves \~94% test accuracy on 7-class emotional dataset
* 🖥️ Real-time GUI built with `tkinter`
* ⚠️ **Voice filtering** using pitch-based gender detection

---

## 📂 Dataset Structure

Place your audio files into the following format under a `datasets` folder:

```
datasets/
├── angry/
├── disgust/
├── fear/
├── happy/
├── neutral/
├── pleasant/
└── sad/
```

Each folder should contain `.wav` or `.mp3` files labeled by emotion.

---

## 🚀 How to Run

1. Clone the repo:

```bash
git clone https://github.com/yourusername/female_voice_emotion_classifier.git
cd female_voice_emotion_classifier
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the script:

```bash
python female_voice_emotion_classifier.py
```

4. Use the GUI to upload or record audio

---

## 🎯 Model Details

* Base Model: **VGG16** (frozen)
* Custom head: Flatten → Dense(512) → Dropout → Dense(7)
* Trained for 5 epochs on Mel spectrograms (resized to 224×224×3)
* Preprocessing includes:

  * Mel spectrogram
  * Resizing
  * 3-channel stacking
* Female voice validated by pitch threshold (180–225 Hz)

---

## 📈 Performance

* ✅ Test Accuracy: \~94%
* ✅ Real-time prediction
* ⚠️ Only processes **female** English voice

---

## 🛠 Dependencies

* Python 3.7+
* TensorFlow / Keras
* Librosa
* OpenCV / NumPy / Tkinter
* pyaudio (for mic recording)

---

## 🙋 Author

**Shivam Sharma**
GitHub: [@shivam15112003](https://github.com/shivam15112003)

Feel free to fork or contribute!
