# 🎯 Female English Voice Emotion Detection System (Fine-Tuned)

This project implements a fully automated real-time **audio emotion detection system** designed for **female English voices**, combining:

- 🎙 Audio Processing (Mel Spectrograms)
- 🧠 Deep Learning (VGG16 with Fine-Tuning)
- 🎯 Gender & Language Verification (Pitch + SpeechBrain)
- 🖥 Interactive GUI (Tkinter)
- 🎯 Real-Time Microphone & File Upload Support

---

## 🔧 Project Structure

```
datasets/
    angry/
    disgust/
    fear/
    happy/
    neutral/
    pleasant/
    sad/
emotion_model.h5 (Generated after training)
main.py (Full code)
requirements.txt
README.md
```

---

## ⚙ Key Features

- ✅ Fine-tuned VGG16 using top 20 unfrozen layers on Mel Spectrogram images.
- ✅ Gender filtering using pitch analysis via Librosa.
- ✅ Language detection using SpeechBrain (supports multiple languages).
- ✅ Real-time microphone recording (5 seconds) or file upload.
- ✅ Fully integrated Tkinter GUI for interactive prediction.

---

## 🏗 Model Architecture

- **Base Model:** VGG16 (pretrained on ImageNet)
- **Fine-Tuning:** Top 20 layers unfrozen
- **Classifier Head:** Flatten → Dense(512, relu) → Dropout(0.5) → Dense(7, softmax)
- **Optimizer:** Adam (learning rate = 1e-5)
- **Loss:** categorical_crossentropy

---

## 🏃‍♂️ How to Run

1️⃣ Install dependencies:

```
pip install -r requirements.txt
```

2️⃣ Prepare your dataset following the folder structure mentioned above.

3️⃣ Run the system:

```
python main.py
```

- If `emotion_model.h5` doesn't exist, the system will automatically train it first.

---

## 📈 Accuracy Achieved

- Test Accuracy: ~95% after fine-tuning.
- Increased robustness with gender & language filtering.


---

## 💡 Technologies Used

- TensorFlow / Keras
- Librosa
- SpeechBrain (for language detection)
- Tkinter (GUI)
- Pyaudio (for recording)

---

## 🔒 License

This project is for academic and research purposes only.

---

✅ **Author:** Shivam Sharma (2025)
