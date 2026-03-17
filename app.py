import json
import pickle
import random
from datetime import datetime

from flask import Flask, render_template, request

from utils import tokenize, stem, bag_of_words


app = Flask(__name__)


# Load model
with open("model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
words = data["words"]
tags = data["tags"]


# Load intents
with open("intents.json", "r") as file:
    intents = json.load(file)


def get_response(msg):

    sentence = tokenize(msg)
    X = bag_of_words(sentence, words)
    X = [X]

    probs = model.predict_proba(X)[0]

    max_prob = max(probs)

    tag_index = probs.argmax()

    tag = tags[tag_index]

    if max_prob > 0.6:

        for intent in intents["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])

    return "I don't understand"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get", methods=["POST"])
def chatbot():

    msg = request.form["msg"]

    response = get_response(msg)

    # save chat history
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open("chat_history.txt", "a", encoding="utf-8") as f:
        f.write("Time: " + time + "\n")
        f.write("User: " + msg + "\n")
        f.write("Bot: " + response + "\n")
        f.write("-----------------\n")

    return response


if __name__ == "__main__":
    app.run(debug=True)