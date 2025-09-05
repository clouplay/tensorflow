#!/usr/bin/env python3
"""
🎵 TensorFlow Müzik Sınıflandırma Örneği
Gürhan Şen tarafından geliştirilmiştir.
"""

import tensorflow as tf
import librosa
import numpy as np

class MusicClassifier:
    def __init__(self):
        self.genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
                      'jazz', 'metal', 'pop', 'reggae', 'rock']
        self.model = None
    
    def extract_features(self, file_path):
        """MFCC özelliklerini çıkar"""
        audio, _ = librosa.load(file_path, sr=22050, duration=30)
        mfccs = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=13)
        return np.concatenate([np.mean(mfccs.T, axis=0), np.std(mfccs.T, axis=0)])
    
    def build_model(self, input_shape):
        """CNN modeli oluştur"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(len(self.genres), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def predict_genre(self, file_path):
        """Müzik türünü tahmin et"""
        features = self.extract_features(file_path).reshape(1, -1)
        prediction = self.model.predict(features)
        genre_idx = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        return {
            'predicted_genre': self.genres[genre_idx],
            'confidence': float(confidence)
        }

# Kullanım örneği
if __name__ == "__main__":
    print("🎵 TensorFlow Müzik Sınıflandırma")
    classifier = MusicClassifier()
    print(f"Desteklenen türler: {', '.join(classifier.genres)}")