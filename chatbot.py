import json
import random
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB

# Load intents.json
with open("intents.json", encoding="utf-8") as file:
    data = json.load(file)


# Extract patterns and tags
patterns = []
tags = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(intent["tag"])

# Convert text to numbers (Bag of Words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(patterns)

# Encode labels (tags → numbers)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(tags)

# Train model (Naive Bayes classifier)
model = MultinomialNB()
model.fit(X, y)

print("Chatbot is ready! Type 'quit' to exit.\n")

# Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("Bot: Goodbye 👋")
        break

    # Transform user input
    X_test = vectorizer.transform([user_input])
    tag_index = model.predict(X_test)[0]
    predicted_tag = label_encoder.inverse_transform([tag_index])[0]

    # Pick a random response
    for intent in data["intents"]:
        if intent["tag"] == predicted_tag:
            response = random.choice(intent["responses"])
            print(f"Bot: {response}")
