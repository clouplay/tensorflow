# 📱 Mobil Müzik Sınıflandırma - TensorFlow Lite

TensorFlow Lite kullanarak mobil cihazlarda müzik sınıflandırması.

## 🎯 Özellikler

- TensorFlow Lite optimizasyonu
- Mobil cihaz desteği (Android/iOS)
- Düşük bellek kullanımı
- Real-time inference

## 🚀 Kullanım

```python
import tensorflow as tf

# TFLite model yükle
interpreter = tf.lite.Interpreter(model_path="music_classifier.tflite")
interpreter.allocate_tensors()

# Tahmin yap
input_data = extract_features(audio_file)
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
```

Geliştirici: Gürhan Şen (@gurhansen)