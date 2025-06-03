import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Carregar e normalizar CIFAR-10
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = y_train.flatten()
y_test = y_test.flatten()

# Classes do CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Mostrar algumas imagens de treino
def show_images(images, labels, n=10):
    plt.figure(figsize=(15, 2))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.xticks([]); plt.yticks([]); plt.grid(False)
        plt.imshow(images[i])
        plt.xlabel(class_names[labels[i]])
    plt.show()

print("Amostras de treino:")
show_images(x_train, y_train)

# Data Augmentation
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)
train_datagen.fit(x_train)

# Definir modelo CNN melhorado
model = models.Sequential([
    layers.Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(32, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(64, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(512),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compilar modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(patience=7, restore_best_weights=True)
lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                patience=3, min_lr=1e-6, verbose=1)

# Treinar modelo com augmentation
history = model.fit(train_datagen.flow(x_train, y_train, batch_size=64),
                    epochs=12,
                    validation_data=(x_test, y_test),
                    callbacks=[early_stop, lr_schedule])

# Avaliação final
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nAcurácia na base de teste: {test_acc:.2f}')

# Visualizar curvas de acurácia
plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()
plt.grid(True)
plt.show()

# Fazer predições
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)

# Mostrar imagens com predições
def show_predictions(images, labels_true, labels_pred, n=10):
    plt.figure(figsize=(15, 2))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.xticks([]); plt.yticks([]); plt.grid(False)
        plt.imshow(images[i])
        label_color = 'green' if labels_true[i] == labels_pred[i] else 'red'
        plt.xlabel(f"Pred: {class_names[labels_pred[i]]}", color=label_color)
    plt.show()

print("Exemplos com predições (verde = correto, vermelho = erro):")
show_predictions(x_test, y_test, predicted_labels)
