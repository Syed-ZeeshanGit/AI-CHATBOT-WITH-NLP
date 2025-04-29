import random
import json
import nltk
import spacy
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Load intents
intents = {
    "greeting": {
        "patterns": ["hello", "hi", "hey", "good morning", "good evening"],
        "responses": ["Hello!", "Hi there!", "Hey!", "Hi! How can I help you today?"]
    },
    "goodbye": {
        "patterns": ["bye", "see you", "goodbye", "take care"],
        "responses": ["Goodbye!", "See you soon!", "Take care!"]
    },
    "thanks": {
        "patterns": ["thanks", "thank you", "thx", "thank you so much"],
        "responses": ["You're welcome!", "Any time!", "Glad I could help!"]
    },
    "internship": {
        "patterns": ["certificate", "internship", "completion certificate", "end date"],
        "responses": [
            "Your completion certificate will be issued at the end of your internship.",
            "You'll receive the certificate upon successful internship completion."
        ]
    },
    "name": {
        "patterns": ["what is your name", "who are you"],
        "responses": ["I am an AI chatbot built using NLP!", "My name is CodTech Bot."]
    },
    "unknown": {
        "responses": ["I'm sorry, I don't understand that.", "Can you rephrase your question?"]
    }
}

# Preprocess input
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    filtered = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return filtered

# Match intent
def match_intent(user_input):
    processed = preprocess(user_input)
    for intent, data in intents.items():
        for pattern in data.get("patterns", []):
            pattern_tokens = preprocess(pattern)
            if set(pattern_tokens).intersection(set(processed)):
                return intent
    return "unknown"

# Chat function
def chat():
    print("🤖 Hello! I am your AI Chatbot. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            print("Bot:", random.choice(intents["goodbye"]["responses"]))
            break
        intent = match_intent(user_input)
        print("Bot:", random.choice(intents[intent]["responses"]))

# Run chatbot
if __name__ == "__main__":
    chat()
