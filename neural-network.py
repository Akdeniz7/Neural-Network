import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Sahte veri oluşturma
np.random.seed(0)
X = np.random.rand(1000, 5)  # 200 örnek, her biri 5 öznitelik
y = (X[:, 0] + X[:, 1] + X[:, 2] > 1.5).astype(int)  # Basit bir kurala göre sınıflandırma

# Veriyi eğitim ve test kümesi olarak bölelim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veriyi ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Yapay sinir ağı modeli oluşturma
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_dim=5),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Modeli derleme
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Modeli eğitme
model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, validation_split=0.1)

# Modelin test verisi üzerinde değerlendirilmesi
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Kredi uygunluk tahminleri
predictions = model.predict(X_test_scaled)
rounded_predictions = np.round(predictions)  # Tahminleri 0 veya 1'e yuvarlayalım

for i in range(len(X_test)):
    result = "Uygun" if rounded_predictions[i] == 1 else "Uygun Değil"
    print(f"Kişi {i+1}: Krediye {result}")

