from tensorflow.keras.models import load_model
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy as np
import pickle
import random
import json
from nltk import word_tokenize
import pandas as pd
import re

with open('intents.json') as file:
    data_chatbot = json.load(file)

with open('information.pickle', 'rb') as file:
    vocab, classes, training = pickle.load(file)

with open('dictionary_diseases.pickle', 'rb') as file:
    dic_code = pickle.load(file)

data = pd.read_csv("Training.csv")

def create_symptoms_list(dataframe):
    symptoms_org = dataframe.columns.tolist()[:-1]
    symptoms = []
    for symptom in symptoms_org:
        symptom = symptom.replace('_', ' ')
        symptoms.append(symptom)

    return symptoms

def create_one_hot_symptoms(symptoms_list, user_input):
    o_input = [0 for _ in range(len(symptoms_list))]
    for index, item in enumerate(symptoms_list):
        y = re.findall(item, user_input)
        if y:
            o_input[index] = 1
    return o_input

def create_bag_of_words(sentence):
    bag = [0 for _ in range(len(vocab))]
    word_of_sentence = word_tokenize(sentence)
    stemmed_word = [stemmer.stem(w.lower()) for w in word_of_sentence]
    for i, w in enumerate(vocab):
        if w in stemmed_word:
            bag[i] = 1

    return np.array([bag])

def count_frequence_input(user_input, vocab_list):
    list_of_word = word_tokenize(user_input.lower())
    score = 0
    for w in list_of_word:
        if w in vocab_list:
            score += 1

    freq = score / len(vocab_list) * 100
    return freq

# Load models
chatbot_model = load_model('chatbot.h5')
diagnosing_model = load_model('Diagnosing.h5')

symptoms = create_symptoms_list(data)


def chat(request):
    res = ""
    result = chatbot_model.predict(create_bag_of_words(request))
    tag = classes[np.argmax(result)]

    for content in data_chatbot['intents']:
        if content['tag'] == tag:
            response = content['responses']
    res = random.choice(response)

    if tag == 'diseases' and count_frequence_input(request, vocab) > 0.0:
        o_input = create_one_hot_symptoms(symptoms, request)
        o_input = np.array(o_input)
        o_input = o_input.reshape(1, 132)
        disease = diagnosing_model.predict(o_input)
        res = "I think you have " + dic_code[np.argmax(disease)]

    if tag == 'datetime' and count_frequence_input(request, vocab) > 0.0:
        import datetime
        x = datetime.datetime.now().strftime("%A")
        res = "It is on " + x

    if count_frequence_input(request, vocab) == 0.0:
        try:
            from googlesearch import search
            res = "Sorry, I can't help you, check some sites to know:\n"
            query = request
            num = 0
            for i in search(query,  # The query you want to run
                            tld='com',  # The top level domain
                            lang='en',  # The language
                            num=10,  # Number of results per page
                            start=0,  # First result to retrieve
                            stop=None,  # Last result to retrieve
                            pause=2.0,  # Lapse between HTTP requests
                            ):
                res += i + '\n'
                if num == 5:
                    break
                num += 1
        except:
            res = "Not response due to:\n1.bad connection.\n2.Too many requests in short time (DDOS consideration)"
        
    return res
