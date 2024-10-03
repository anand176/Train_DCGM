import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'  # Directory to save uploaded videos
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to extract frames from a video
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (64, 64))  # Resize frame to (64, 64)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        frames.append(frame)
    cap.release()
    return np.array(frames)

# Function to build the autoencoder model
def build_autoencoder():
    input_img = Input(shape=(64, 64, 1))  # Input shape for grayscale images

    # Encoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.2)(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.2)(x)

    # Decoder
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)

    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')
    return autoencoder

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video_file = request.files['video']
        if video_file:
            video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
            video_file.save(video_path)
            # Start training in the background
            return redirect('/training')
    return render_template('index.html')

@app.route('/training')
def training():
    video_path = os.path.join(UPLOAD_FOLDER, os.listdir(UPLOAD_FOLDER)[0])  # Get the uploaded video path
    frames = extract_frames(video_path)
    frames = frames.astype('float32') / 255.0  # Normalize pixel values
    frames = np.expand_dims(frames, axis=-1)  # Add channel dimension
    X_train, X_test = train_test_split(frames, test_size=0.2)
    autoencoder = build_autoencoder()
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_test, X_test))
    autoencoder.save("autoencoder_video1.h5")
    return "Training has started..."

if __name__ == '__main__':
    app.run(debug=True)
