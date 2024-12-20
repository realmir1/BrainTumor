
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split



data_dir = '/kaggle/input/brain-mri-images-for-brain-tumor-detection'
categories = ['no', 'yes']  

img_size = 128
data = []    
labels = []  


print("Veri yükleniyor...")
for category in categories:
    path = os.path.join(data_dir, category)  
    class_num = categories.index(category)   
    
    for img in os.listdir(path):
        try:
        
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            resized_array = cv2.resize(img_array, (img_size, img_size))
            
          
            data.append(resized_array)
            labels.append(class_num)
        except Exception as e:
            print("Hata:", e)

print("Veri yükleme tamamlandı.")
data = np.array(data).reshape(-1, img_size, img_size, 1) / 255.0 
labels = np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)
model = Sequential([Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Model eğitiliyor...")
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=1)
print("Model eğitimi tamamlandı.")


accuracy = model.evaluate(X_test, y_test)[1]
print(f"Model Test Doğruluğu: {accuracy * 100:.2f}%")


sample_images = X_test[:4]
sample_labels = y_test[:4]
predictions = model.predict(sample_images)


plt.figure(figsize=(12, 6))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(sample_images[i].reshape(img_size, img_size), cmap='gray')
    
    true_label = "Yes" if np.argmax(sample_labels[i]) == 1 else "No"
    predicted_label = "Yes" if np.argmax(predictions[i]) == 1 else "No"
    
    plt.title(f"Gerçek: {true_label}\nTahmin: {predicted_label}")
    plt.axis('off')

plt.tight_layout()
plt.show()



plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.title('Model Doğruluk Grafiği')
plt.legend()
plt.show()
