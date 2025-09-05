# 🇹🇷 TensorFlow ile Müzik Sınıflandırma - Türkçe Tutorial

Bu tutorial, TensorFlow kullanarak müzik türü sınıflandırması yapmayı öğretir.

## 📚 İçerik

### 1. Giriş
Müzik sınıflandırması, ses sinyallerini analiz ederek müzik türünü otomatik olarak belirleme işlemidir.

### 2. Veri Hazırlığı
```python
# Ses dosyasını yükle
import librosa
audio, sr = librosa.load('muzik.wav', sr=22050)

# MFCC özelliklerini çıkar
mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
```

### 3. Model Oluşturma
```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

### 4. Eğitim
```python
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(X_train, y_train, epochs=100)
```

### 5. Tahmin
```python
prediction = model.predict(test_features)
genre = genres[np.argmax(prediction)]
```

## 🎯 Sonuç
Bu tutorial ile TensorFlow kullanarak müzik sınıflandırması yapabilirsiniz.

**Geliştirici:** Gürhan Şen  
**GitHub:** @gurhansen  
**Platform:** ClouSound