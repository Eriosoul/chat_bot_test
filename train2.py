import json
import pickle
import random

import nltk
import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Cargar los patrones de entrenamiento desde el archivo JSON
with open('intents.json', encoding='utf-8') as file:
    intents = json.load(file)


# Descargar los recursos de NLTK
nltk.download('punkt')
nltk.download('wordnet')

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '¿', '.', ',']

# Clasificar los patrones y las categorías
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern.lower())
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Lematizar las palabras y eliminar letras ignoradas
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

# Guardar las palabras y las categorías en archivos pickle
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Preparar los datos de entrenamiento
training = []
output_empty = [0] * len(classes)
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)

train_x = [item[0] for item in training]
train_y = [item[1] for item in training]

# Crear la red neuronal
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))

# Añadir capas adicionales
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

# Añadir más capas si lo deseas
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))

# Añadir más capas si lo deseas
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))

# Capa de salida
model.add(Dense(len(train_y[0]), activation='softmax'))

# Configurar el optimizador y compilar el modelo
sgd = SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Entrenar el modelo
train_process = model.fit(np.array(train_x), np.array(train_y), epochs=10000, batch_size=5, verbose=1)

# Guardar el modelo entrenado
model.save("chatbot_model.h5", train_process.history)
