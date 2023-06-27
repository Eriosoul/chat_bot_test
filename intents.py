import json
import pickle
import random

import nltk
import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from nltk.stem import WordNetLemmatizer
from numpy import ndarray

# Pasamos por lematizador y lo convertimos a 0 y 1
lemmatizer = WordNetLemmatizer()

# Cargar intents desde el archivo JSON
with open('intents.json') as file:
    intents = json.load(file)

nltk.download('punkt')
nltk.download('wordnet')

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '¿', ',', '.']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Definimos la lista en algo simplificado
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Lista para entrenamiento
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

    # Asegurar que bag y output_row tengan la misma longitud
    if len(bag) != len(output_row):
        continue  # Saltar este documento si no tienen la misma longitud

    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x: ndarray = np.vstack(training[:, 0])
train_y = np.vstack(training[:, 1])

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
train_process = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save("chatbot_model.h5")
