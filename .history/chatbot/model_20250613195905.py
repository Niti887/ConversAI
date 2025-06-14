import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import random

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

class ChatbotModel:
    def __init__(self, intents_file='data/intents.json'):
        self.lemmatizer = WordNetLemmatizer()
        self.intents_file = intents_file
        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_words = ['?', '!', '.', ',']
        self.model = None

    def preprocess_data(self):
        # Load and parse the intents file
        with open(self.intents_file, 'r') as file:
            intents = json.load(file)

        # Extract words, classes, and documents
        for intent in intents['intents']:
            for pattern in intent['patterns']:
                # Tokenize each word
                w = nltk.word_tokenize(pattern)
                self.words.extend(w)
                # Add to documents
                self.documents.append((w, intent['tag']))
                # Add to classes
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        # Lemmatize and lower each word
        self.words = [self.lemmatizer.lemmatize(w.lower()) for w in self.words if w not in self.ignore_words]
        self.words = sorted(list(set(self.words)))
        self.classes = sorted(list(set(self.classes)))

        # Create training data
        training = []
        output_empty = [0] * len(self.classes)

        # Create bag of words for each document
        for doc in self.documents:
            bag = []
            word_patterns = doc[0]
            word_patterns = [self.lemmatizer.lemmatize(word.lower()) for word in word_patterns]
            
            for word in self.words:
                bag.append(1) if word in word_patterns else bag.append(0)

            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            training.append([bag, output_row])

        # Shuffle and convert to numpy array
        random.shuffle(training)
        training = np.array(training, dtype=object)

        # Split into X and Y
        train_x = list(training[:, 0])
        train_y = list(training[:, 1])

        return train_x, train_y

    def build_model(self, train_x, train_y):
        # Build the neural network
        self.model = Sequential()
        self.model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(len(train_y[0]), activation='softmax'))

        # Compile the model
        sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    def train_model(self, train_x, train_y, epochs=200, batch_size=5):
        # Train the model
        hist = self.model.fit(np.array(train_x), np.array(train_y), epochs=epochs, batch_size=batch_size, verbose=1)
        return hist

    def save_model(self, model_file='chatbot_model.h5', words_file='words.pkl', classes_file='classes.pkl'):
        # Save the model and necessary files
        self.model.save(model_file)
        pickle.dump(self.words, open(words_file, 'wb'))
        pickle.dump(self.classes, open(classes_file, 'wb'))

    def load_model(self, model_file='chatbot_model.h5', words_file='words.pkl', classes_file='classes.pkl'):
        # Load the model and necessary files
        self.model = tf.keras.models.load_model(model_file)
        self.words = pickle.load(open(words_file, 'rb'))
        self.classes = pickle.load(open(classes_file, 'rb'))

    def clean_up_sentence(self, sentence):
        # Tokenize and lemmatize the sentence
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def bag_of_words(self, sentence):
        # Create bag of words for the sentence
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        for w in sentence_words:
            for i, word in enumerate(self.words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)

    def predict_class(self, sentence):
        # Predict the class of the input sentence
        bow = self.bag_of_words(sentence)
        res = self.model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': self.classes[r[0]], 'probability': str(r[1])})
        return return_list

    def get_response(self, sentence):
        # Get a response based on the predicted intent
        intents_list = self.predict_class(sentence)
        if not intents_list:
            return "I'm not sure I understand. Could you please rephrase that?"
        
        tag = intents_list[0]['intent']
        with open(self.intents_file, 'r') as file:
            intents = json.load(file)
        
        for intent in intents['intents']:
            if intent['tag'] == tag:
                return random.choice(intent['responses']) 