# 🎵 Müzik Sınıflandırma - TensorFlow Örneği

Bu örnek, TensorFlow kullanarak müzik türü sınıflandırması yapan bir AI modeli gösterir.

## 🎯 Özellikler

- 10 farklı müzik türü sınıflandırması
- MFCC tabanlı özellik çıkarımı  
- CNN (Convolutional Neural Network) mimarisi
- Real-time prediction desteği

## 🚀 Hızlı Başlangıç

```python
from music_classifier import MusicClassifier

# Model oluştur
classifier = MusicClassifier()

# Tahmin yap
result = classifier.predict_genre('test_song.wav')
print(f"Tür: {result['predicted_genre']} ({result['confidence']:.2%})")
```

## 📊 Desteklenen Türler

blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock

## 🔧 Kurulum

```bash
pip install tensorflow librosa numpy
```

## 📁 Veri Yapısı

```
data/
├── genres/
│   ├── blues/
│   ├── classical/
│   ├── jazz/
│   └── ...
```

Geliştirici: Gürhan Şen (@gurhansen)