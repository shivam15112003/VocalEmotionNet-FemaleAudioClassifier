import tkinter as tk
from tkinter import filedialog, messagebox
import pyaudio
import wave
import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from tensorflow.keras.applications import VGG16
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model
import tensorflow as tf

# Define emotion classes
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'pleasant', 'sad']

# Feature extraction from audio file
def extract_features(file_path):
    y, sr = librosa.load(file_path)
    mels = librosa.feature.melspectrogram(y=y, sr=sr)
    mels_resized = tf.image.resize(mels[..., np.newaxis], (224, 224)).numpy()
    mels_rgb = np.repeat(mels_resized, 3, axis=-1)
    return mels_rgb

# -------------------- TRAINING PHASE ----------------------

# Load dataset
print("Loading dataset...")
audio_data, labels = [], []
for folder_name in classes:
    folder_path = os.path.join('datasets', folder_name)
    for file in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file)
        try:
            features = extract_features(full_path)
            audio_data.append(features)
            labels.append(folder_name)
        except:
            print(f"Skipped corrupted file: {file}")

# Encode labels
le = LabelEncoder()
encoded_labels = le.fit_transform(labels)
categorical_labels = to_categorical(encoded_labels)

# Prepare data
X = np.array(audio_data)
y = np.array(categorical_labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(len(classes), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
print("Training model...")
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Evaluate model
predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(y_test, axis=1)
print("Test Accuracy:", accuracy_score(y_true, y_pred))

# Save model
model.save("emotion_model.h5")
print("Model saved as emotion_model.h5")

# -------------------- GUI & PREDICTION PHASE ----------------------

# Load trained model
model = tf.keras.models.load_model("emotion_model.h5")

# GUI functions

def classify_audio(file_path):
    y, sr = librosa.load(file_path)
    mels = librosa.feature.melspectrogram(y=y, sr=sr)
    mels_resized = tf.image.resize(mels[..., np.newaxis], (224, 224)).numpy()
    mels_rgb = np.repeat(mels_resized[np.newaxis, ...], 3, axis=-1)

    # Gender filtering using pitch
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    avg_pitch = np.mean(pitches[np.nonzero(pitches)])

    if avg_pitch < 180:
        messagebox.showwarning("Voice Error", "Voice must be female (pitch too low).")
        return
    elif avg_pitch > 225:
        messagebox.showwarning("Language Error", "Audio should be in English (pitch too high).")
        return

    prediction = model.predict(mels_rgb)
    predicted_class = np.argmax(prediction)
    emotion = classes[predicted_class]
    messagebox.showinfo("Result", f"Predicted Emotion: {emotion}")

# Browse file function
def browseFiles():
    filename = filedialog.askopenfilename(filetypes=[("Audio files", ".wav .mp3")])
    if filename:
        label_file_explorer.config(text=f"File Opened: {filename}")
        classify_audio(filename)

# Recording function
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

# Build GUI
window = tk.Tk()
window.title("Audio Emotion Detection (Female English Voice)")
window.geometry("500x400")

label_file_explorer = tk.Label(window, text="No file selected", wraplength=400)
label_file_explorer.pack(pady=10)

tk.Button(window, text="Upload Audio", command=browseFiles).pack(pady=10)
tk.Button(window, text="Record Audio", command=record_audio).pack(pady=10)

window.mainloop()
