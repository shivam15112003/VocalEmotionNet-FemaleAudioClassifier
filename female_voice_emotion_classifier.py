import os
import numpy as np
import librosa
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, messagebox
import pyaudio
import wave
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from tensorflow.keras.applications import VGG16
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model, load_model
import speechbrain as sb
from speechbrain.pretrained import LanguageIdentification
import torchaudio

# ------------- Configuration -------------
DATASET_PATH = 'datasets'
MODEL_PATH = 'emotion_model.h5'
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'pleasant', 'sad']

# ------------- Feature Extraction Function -------------
def extract_features(file_path):
    y, sr = librosa.load(file_path)
    mels = librosa.feature.melspectrogram(y=y, sr=sr)
    mels_resized = tf.image.resize(mels[..., np.newaxis], (224, 224)).numpy()
    mels_rgb = np.repeat(mels_resized, 3, axis=-1)
    return mels_rgb

# ------------- Train Model Function -------------
def train_model():
    print("Loading dataset...")
    audio_data, labels = [], []
    for folder_name in classes:
        folder_path = os.path.join(DATASET_PATH, folder_name)
        for file in os.listdir(folder_path):
            full_path = os.path.join(folder_path, file)
            try:
                features = extract_features(full_path)
                audio_data.append(features)
                labels.append(folder_name)
            except Exception as e:
                print(f"Skipped file: {file} — Error: {e}")

    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    categorical_labels = to_categorical(encoded_labels)

    X = np.array(audio_data)
    y = np.array(categorical_labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False

    x = Flatten()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(len(classes), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print("Training model...")
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {accuracy*100:.2f}%")

    model.save(MODEL_PATH)
    print("✅ Model saved.")

# ------------- Gender & Language Detection Functions -------------
def estimate_gender(file_path):
    y, sr = librosa.load(file_path)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    avg_pitch = np.mean(pitches[np.nonzero(pitches)])

    if avg_pitch < 165:
        return "male"
    elif avg_pitch > 250:
        return "very high (possible child)"
    else:
        return "female"

lang_id_model = LanguageIdentification.from_hparams(
    source="speechbrain/lang-id-commonlanguage_ecapa",
    savedir="pretrained_models/lang-id-commonlanguage_ecapa"
)

def detect_language(file_path):
    signal, fs = torchaudio.load(file_path)
    prediction = lang_id_model.classify_batch(signal)
    language = prediction[3][0]
    return language

# ------------- Full Classification Function -------------
def classify_audio(file_path):
    gender = estimate_gender(file_path)
    if gender != "female":
        messagebox.showwarning("Gender Check", f"Voice detected as {gender}. Only female voices allowed.")
        return

    language = detect_language(file_path)
    if language != "en":
        messagebox.showwarning("Language Check", f"Detected language: {language}. Only English allowed.")
        return

    mels_rgb = extract_features(file_path)
    mels_rgb = np.expand_dims(mels_rgb, axis=0)
    prediction = model.predict(mels_rgb)
    predicted_class = np.argmax(prediction)
    emotion = classes[predicted_class]
    messagebox.showinfo("Result", f"Predicted Emotion: {emotion}")

# ------------- GUI Functions -------------
def browseFiles():
    filename = filedialog.askopenfilename(filetypes=[("Audio files", ".wav .mp3")])
    if filename:
        label_file_explorer.config(text=f"File Opened: {filename}")
        classify_audio(filename)

def record_audio():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 22050
    SECONDS = 5
    OUTPUT = "output.wav"

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = [stream.read(CHUNK) for _ in range(0, int(RATE / CHUNK * SECONDS))]
    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(OUTPUT, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    classify_audio(OUTPUT)

# ------------- Main Execution -------------
if not os.path.exists(MODEL_PATH):
    train_model()

model = load_model(MODEL_PATH)

# Build GUI
window = tk.Tk()
window.title("Audio Emotion Detection (Fully Integrated)")
window.geometry("500x400")

label_file_explorer = tk.Label(window, text="No file selected", wraplength=400)
label_file_explorer.pack(pady=10)

tk.Button(window, text="Upload Audio", command=browseFiles, font=("Arial", 14)).pack(pady=10)
tk.Button(window, text="Record Audio", command=record_audio, font=("Arial", 14)).pack(pady=10)

window.mainloop()