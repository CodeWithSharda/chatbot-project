import json
import random
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load trained model + tools
model = load_model("chatbot_model.h5")
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Load intents
with open("intents.json", encoding="utf-8") as file:
    data = json.load(file)

print("Chatbot ready! Type 'quit' to exit.\n")

# Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("Bot: Goodbye 👋")
        break

    X_test = vectorizer.transform([user_input]).toarray()
    prediction = model.predict(X_test)
    tag_index = np.argmax(prediction)
    predicted_tag = label_encoder.inverse_transform([tag_index])[0]

    for intent in data["intents"]:
        if intent["tag"] == predicted_tag:
            response = random.choice(intent["responses"])
            print(f"Bot: {response}")
