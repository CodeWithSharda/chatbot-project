import json
import random
import numpy as np
import nltk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.feature_extraction.text import CountVectorizer

# Load intents
with open("intents.json", encoding="utf-8") as file:
    data = json.load(file)

# Prepare data
patterns = []
tags = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(intent["tag"])

# Convert text to vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(patterns).toarray()

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(tags)
y = to_categorical(y)

# Build model
model = Sequential()
model.add(Dense(128, input_shape=(X.shape[1],), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(y.shape[1], activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.01), metrics=["accuracy"])

# Train model
print("Training model...")
model.fit(X, y, epochs=200, batch_size=8, verbose=1)

# Save model + vectorizer + encoder
model.save("chatbot_model.h5")
import pickle
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("Model trained and saved!")

# Chat loop
print("\nChatbot ready! Type 'quit' to exit.\n")
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
