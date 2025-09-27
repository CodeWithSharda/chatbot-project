# 🤖 Chatbot with NLP and Machine Learning

This project implements a **simple chatbot** using Natural Language Processing (NLP) and Machine Learning in Python.  
It uses **NLTK, spaCy, scikit-learn, and TensorFlow** to classify intents and generate responses.

---

## 🚀 Features
- Intent classification using Bag-of-Words and Naive Bayes.
- Preprocessing with tokenization, stopword removal, and lemmatization.
- Dataset stored in `intents.json`.
- Interactive chat loop in terminal.
- Easily extendable with new intents, patterns, and responses.

---

## 📂 Project Structure
chatbot_project/
│
├── intents.json # Dataset of intents
├── chatbot.py # Main chatbot code
├── requirements.txt # Dependencies
└── README.md # Project documentation


---

## ⚡ Installation & Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/chatbot-project.git
   cd chatbot-project
2. Create virtual environment (optional).
3. Install dependencies:
   pip install -r requirements.txt
   ```
4. Run chatbot:
   python chatbot.py
   ```

## ✨ Example
You: Hi
Bot: Hello!

You: Tell me a joke
Bot: Why don’t programmers like nature? Too many bugs 🐛

## 📌 Future Improvements

* Replace Naive Bayes with Deep Learning (TensorFlow/Keras).

* Add context handling for multi-turn conversations.

* Connect to external APIs (e.g., weather, Wikipedia).

* Deploy as a web app using Flask/Streamlit.

## 🧑‍💻 Author

Built as a learning project with Python + NLP + ML.


---
