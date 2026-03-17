import json
import numpy as np
import nltk
from sklearn.neural_network import MLPClassifier
import pickle

from utils import tokenize, stem, bag_of_words


# Load intents file
with open("intents.json", "r") as file:
    intents = json.load(file)


all_words = []
tags = []
xy = []


# Read patterns
for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)

    for pattern in intent["patterns"]:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))


ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]

all_words = sorted(set(all_words))
tags = sorted(set(tags))


# Training data
X_train = []
y_train = []


for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)


X_train = np.array(X_train)
y_train = np.array(y_train)


# Train model
model = MLPClassifier(hidden_layer_sizes=(8, 8), max_iter=1000)
model.fit(X_train, y_train)


# Save model
data = {
    "model": model,
    "words": all_words,
    "tags": tags
}

with open("model.pkl", "wb") as f:
    pickle.dump(data, f)


print("Model trained and saved!")