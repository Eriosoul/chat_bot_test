import pickle
import numpy as np
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import nltk
import json
import random
import datetime

lemmatizer = WordNetLemmatizer()

# Cargar las palabras y las categorías desde los archivos pickle
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Cargar el modelo entrenado
model = load_model('chatbot_model.h5')

# Cargar los intents desde el archivo JSON
with open('intents.json') as file:
    intents = json.load(file)


# Preprocesar la oración de entrada
def preprocess_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# Convertir la oración en una bolsa de palabras
def bow(sentence, words):
    sentence_words = preprocess_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)


# Generar respuesta a una oración
def generate_response(sentence):
    bow_input = bow(sentence, words)
    result = model.predict(np.array([bow_input]))[0]
    # Obtener la etiqueta con la probabilidad más alta
    predicted_class_index = np.argmax(result)
    predicted_class = classes[predicted_class_index]
    if result[predicted_class_index] > 0.7:
        # Si la probabilidad supera el umbral de confianza, devolver una respuesta aleatoria de la categoría correspondiente
        for intent in intents['intents']:
            if intent['tag'] == predicted_class:
                response = random.choice(intent['responses'])
                return response
    else:
        # Si la probabilidad es baja, devolver una respuesta por defecto
        return "No estoy seguro de entender. ¿Podrías reformular tu pregunta?"


# Función para registrar la interacción entre el bot y el usuario
def log_interaction(user_input, bot_response):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('interactions.log', 'a') as file:
        file.write(f"[{timestamp}] User: {user_input}\n")
        file.write(f"[{timestamp}] Bot: {bot_response}\n")
        file.write("=" * 50 + "\n")


# Bucle de interacción con el bot
print("¡Hola! Soy un bot de chat. Puedes comenzar a preguntarme o escribir 'salir' para terminar.")
while True:
    input_sentence = input("Tú: ")
    if input_sentence.lower() == "salir":
        break
    response = generate_response(input_sentence)
    log_interaction(input_sentence, response)
    print("Bot: " + response)
