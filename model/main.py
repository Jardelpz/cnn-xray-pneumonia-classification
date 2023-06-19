import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import confusion_matrix

# Caminhos para as pastas de treinamento, validação e teste
train_dir = 'input/train'
val_dir = 'input/val'
test_dir = 'input/test'

# Dimensões das imagens de entrada
input_shape = (150, 150, 3)  # Pode ajustar o tamanho conforme necessário

# Pré-processamento e aumento de dados
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
val_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),  # Alterei para o tamanho desejado das imagens de entrada
    batch_size=32,
    class_mode='binary',
    classes=['NORMAL', 'PNEUMONIA']
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),  # Alterei para o tamanho desejado das imagens de entrada
    batch_size=32,
    class_mode='binary',
    classes=['NORMAL', 'PNEUMONIA']
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),  # Alterei para o tamanho desejado das imagens de entrada
    batch_size=32,
    class_mode='binary',
    classes=['NORMAL', 'PNEUMONIA']
)

# Criar o modelo da CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compilar o modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinar o modelo
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=val_generator,
    validation_steps=len(val_generator)
)

# AQUI PRA BAIXO E SO TESTE PODERIA FAZER TUDO NO OUTRO ARQUIVO

# Avaliar o modelo no conjunto de teste
test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
print('Test accuracy:', test_acc)

# Fazer previsões no conjunto de teste
y_pred = model.predict(test_generator)
y_pred = np.round(y_pred).flatten()

# Calcular acurácia
accuracy = np.mean(y_pred == test_generator.classes)

# Calcular recall
true_positives = np.sum((y_pred == 1) & (test_generator.classes == 1))
false_negatives = np.sum((y_pred == 0) & (test_generator.classes == 1))
recall = true_positives / (true_positives + false_negatives)

# Plotar curva de aprendizado
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Curva de Aprendizado')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend(['Treinamento', 'Validação'], loc='upper left')
plt.show()

# Calcular a matriz de confusão
cm = confusion_matrix(test_generator.classes, y_pred)

# Plotar a matriz de confusão
class_names = ['Pneumonia', 'Normal']
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Matriz de Confusão')
plt.xlabel('Previsão')
plt.ylabel('Valor Real')
plt.show()

# Salvar o modelo em formato .h5
model.save('modelo.h5')

