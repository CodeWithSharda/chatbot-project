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

```
chatbot_project/
│
├── intents.json # Dataset of intents
├── chatbot.py # Main chatbot code
├── requirements.txt # Dependencies
└── README.md # Project documentation
```

---

## ⚡ Installation & Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/chatbot-project.git
   cd chatbot-project
2. Create virtual environment (optional).
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run chatbot:
   ```
   python chatbot.py
   ```

## ✨ Example
You: Hi

Bot: Hello!

You: Tell me a joke

Bot: Why don’t programmers like nature? Too many bugs 🐛

---
---

## 🔥 Deep Learning Version

In addition to the basic Naive Bayes model, this project includes a **Deep Learning upgrade** built with TensorFlow/Keras.

### Files
- `chatbot_dl.py` → trains a neural network (Bag-of-Words → Dense layers → Softmax) and saves the model.
- `chatbot_infer.py` → loads the saved model for chatting instantly without retraining.
- `chatbot_model.h5`, `vectorizer.pkl`, `label_encoder.pkl` → generated after training.

### Training
Run once to train and save the model:
```bash
python chatbot_dl.py
```
### Inference (Chat)
After training, use this for faster startup (no retraining):
```
python chatbot_infer.py
```
### Model

* Input: Bag-of-Words (vectorized user input).

* Hidden layers: Dense layers with ReLU activation + Dropout.

* Output: Softmax over intent classes.

* Optimizer: Adam

* Loss: Categorical Crossentropy
---
## 📌 Future Improvements

* Replace Naive Bayes with Deep Learning (TensorFlow/Keras).

* Add context handling for multi-turn conversations.

* Connect to external APIs (e.g., weather, Wikipedia).

* Deploy as a web app using Flask/Streamlit.

## 🧑‍💻 Author

Built as a learning project with Python + NLP + ML.


---
## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
