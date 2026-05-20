"""
Spam Filter — SMS Message Classifier
-------------------------------------
Dataset : SMS Spam Collection (~5,500 real SMS messages, labeled spam/ham)
Model   : Logistic Regression — fast, interpretable, and strong on text.

What this script does:
  1. Downloads the dataset automatically (no manual steps)
  2. Converts raw text into numbers the model can understand (TF-IDF)
  3. Trains a Naive Bayes classifier
  4. Evaluates it honestly on messages it has never seen
  5. Lets you type your own messages and see predictions in real time

Run it:
    python spam.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)


# ══════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD THE DATASET
# ══════════════════════════════════════════════════════════════════════════
#
# The SMS Spam Collection is a public dataset of 5,574 real SMS messages,
# each labeled as either "spam" or "ham" (not spam).
#
# Source: UCI Machine Learning Repository
# We load it directly from a URL — no download needed.

print("=" * 60)
print("STEP 1: LOADING DATA")
print("=" * 60)

url = (
    "https://raw.githubusercontent.com/justmarkham/"
    "pycon-2016-tutorial/master/data/sms.tsv"
)

df = pd.read_csv(url, sep="\t", header=None, names=["label", "message"])

print(f"Total messages : {len(df)}")
print(f"Spam messages  : {(df['label'] == 'spam').sum()}")
print(f"Ham messages   : {(df['label'] == 'ham').sum()}")
print()
print("Sample messages:")
print(df.sample(5, random_state=1).to_string(index=False))
print()


# ══════════════════════════════════════════════════════════════════════════
# STEP 2: CONVERT TEXT TO NUMBERS (TF-IDF)
# ══════════════════════════════════════════════════════════════════════════
#
# Machine learning models only understand numbers — not words.
# We use TF-IDF to convert each message into a vector of numbers.
#
# TF-IDF stands for Term Frequency–Inverse Document Frequency.
# It answers: "which words are important in THIS message,
#              compared to all other messages?"
#
# Example — the word "free" in a spam message:
#   TF  → "free" appears a lot in this message → high score
#   IDF → "free" is rare across ALL messages → multiplies the score
#   Result → "free" gets a HIGH TF-IDF weight → strong spam signal
#
# Common words like "the", "is", "a" appear everywhere → low IDF → low weight
# Spam words like "WIN", "FREE", "PRIZE" → high weight when they appear

X = df["message"]                                # the raw text
y = (df["label"] == "spam").astype(int)          # 1 = spam, 0 = ham

# Split BEFORE fitting the vectorizer — this prevents data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% held out for testing
    random_state=42,
)

# Fit TF-IDF only on training messages (test messages are "unseen")
vectorizer = TfidfVectorizer(
    stop_words="english",   # ignore common words like "the", "a", "is"
    lowercase=True,         # treat "FREE" and "free" as the same word
    max_features=5000,      # keep only top 5000 most informative words
)

X_train_tfidf = vectorizer.fit_transform(X_train)   # learn vocab + transform
X_test_tfidf  = vectorizer.transform(X_test)        # ONLY transform (no learning)

print("=" * 60)
print("STEP 2: TEXT CONVERTED TO NUMBERS")
print("=" * 60)
print(f"Vocabulary size     : {len(vectorizer.vocabulary_)} unique words")
print(f"Training matrix     : {X_train_tfidf.shape}  (messages × words)")
print(f"Each message is now a row of {X_train_tfidf.shape[1]} numbers")
print()


# ══════════════════════════════════════════════════════════════════════════
# STEP 3: TRAIN THE MODEL
# ══════════════════════════════════════════════════════════════════════════
#
# Why Logistic Regression for text?
#   It learns a weight for every word in the vocabulary.
#   A message's score is the weighted sum of its TF-IDF values.
#   If the score crosses a threshold → spam. Simple, fast, and accurate.

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

print("=" * 60)
print("STEP 3: MODEL TRAINED ✓")
print("=" * 60)
print(f"Learned from {len(X_train)} messages")
print()


# ══════════════════════════════════════════════════════════════════════════
# STEP 4: EVALUATE — HOW GOOD IS IT?
# ══════════════════════════════════════════════════════════════════════════

y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("=" * 60)
print("STEP 4: RESULTS ON UNSEEN MESSAGES")
print("=" * 60)
print(f"Accuracy : {accuracy * 100:.1f}%")
print()

# Unpacking the confusion matrix
#
#                     PREDICTED
#                   Ham   | Spam
#   ACTUAL  Ham  [  TN   |  FP  ]  ← FP = real message wrongly blocked (annoying)
#           Spam [  FN   |  TP  ]  ← FN = spam slips through (filter failure)

tn, fp, fn, tp = cm.ravel()
print("Confusion Matrix:")
print(f"  Correctly let through HAM     : {tn}")
print(f"  Correctly blocked SPAM        : {tp}")
print(f"  Real messages wrongly blocked : {fp}  ← false alarm (bad!)")
print(f"  Spam that slipped through     : {fn}  ← filter missed it")
print()
print("Detailed Report:")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

# Precision tells you: of messages flagged spam, how many really were?
# Recall tells you:    of all actual spam, how many did we catch?
# In spam filtering, high Precision matters — blocking real emails is costly.


# ══════════════════════════════════════════════════════════════════════════
# STEP 5: THE SPAMMIEST WORDS
# ══════════════════════════════════════════════════════════════════════════
#
# Logistic Regression assigns each word a coefficient — positive means
# the word pushes the prediction toward spam, negative means ham.
# Reading these coefficients directly is a form of model explainability.

import numpy as np

feature_names = vectorizer.get_feature_names_out()
spam_coefs     = model.coef_[0]                      # weight of each word for spam
top_spam_indices = np.argsort(spam_coefs)[-15:]      # top 15 spammiest words

print("=" * 60)
print("STEP 5: WORDS MOST ASSOCIATED WITH SPAM")
print("=" * 60)
for idx in reversed(top_spam_indices):
    word = feature_names[idx]
    bar  = "█" * int(spam_coefs[idx] * 10)
    print(f"  {word:<20} {bar}")
print()


# ══════════════════════════════════════════════════════════════════════════
# STEP 6: REAL-TIME PREDICTION — TYPE YOUR OWN MESSAGES
# ══════════════════════════════════════════════════════════════════════════

def predict_message(text):
    """Convert a message to TF-IDF and predict spam or ham."""
    vec  = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    label = "🚨 SPAM" if pred == 1 else "✅ HAM (not spam)"
    confidence = prob[pred] * 100
    return label, confidence

print("=" * 60)
print("STEP 6: TRY SOME EXAMPLE MESSAGES")
print("=" * 60)

examples = [
    "Congratulations! You've won a FREE iPhone. Click here now!!!",
    "Hey, are we still meeting for lunch tomorrow?",
    "URGENT: Your account has been compromised. Verify now to avoid suspension.",
    "Can you pick up some milk on the way home?",
    "Win £1000 cash prize! Text WIN to 87121 now. T&Cs apply.",
    "The meeting has been rescheduled to 3pm.",
]

for msg in examples:
    label, confidence = predict_message(msg)
    short = msg[:55] + "..." if len(msg) > 55 else msg
    print(f"  {label}  ({confidence:.1f}%)")
    print(f"  \"{short}\"")
    print()

print("=" * 60)
print("NOW TRY YOUR OWN MESSAGES (type 'quit' to exit)")
print("=" * 60)

while True:
    msg = input("\nYour message: ").strip()
    if msg.lower() in ("quit", "exit", "q", ""):
        break
    label, confidence = predict_message(msg)
    print(f"  → {label}  (confidence: {confidence:.1f}%)")

print("\nDone! Next step: try adding bigrams (ngram_range=(1,2)) to the vectorizer and see if accuracy improves.")
