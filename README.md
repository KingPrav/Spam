# SMS Spam Filter

A text classification project that detects spam SMS messages using machine learning. Built as a learning exercise covering the full pipeline from raw text to real-time predictions.

## What It Does

- Loads the [SMS Spam Collection dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) (~5,500 labeled messages) directly from the web — no manual download required
- Converts raw text into numerical features using **TF-IDF** vectorization
- Trains a **Logistic Regression** classifier to distinguish spam from ham
- Evaluates the model on a held-out test set and reports accuracy, precision, recall, and a confusion matrix
- Displays the words most strongly associated with spam
- Lets you type your own messages and get live predictions

## Requirements

- Python 3.8+
- `pandas`
- `scikit-learn`
- `numpy`

Install dependencies:

```bash
pip install pandas scikit-learn numpy
```

## Usage

```bash
python spam.py
```

The script will walk through each step with printed output, then drop into an interactive prompt where you can test your own messages. Type `quit` to exit.

## How It Works

| Step | What Happens |
|------|-------------|
| 1 | Dataset is fetched from GitHub and loaded into a DataFrame |
| 2 | Messages are split into train/test sets, then converted to TF-IDF vectors |
| 3 | A Logistic Regression model is trained on the training set |
| 4 | The model is evaluated on the unseen test set |
| 5 | The top 15 words by Logistic Regression coefficient are printed with a visual bar chart |
| 6 | Example messages are classified, then you can enter your own |

### Why TF-IDF?

TF-IDF (Term Frequency–Inverse Document Frequency) weights words by how often they appear in a message relative to how common they are across all messages. This makes spammy words like "FREE" or "WIN" stand out, while common words like "the" or "is" are down-weighted automatically.

### Why Logistic Regression?

Fast, interpretable, and effective for text classification. The model's coefficients directly reveal which words push a message toward spam or ham.

## What's Next

A natural next experiment: add **bigrams** to the vectorizer and see if accuracy improves.

```python
vectorizer = TfidfVectorizer(
    stop_words="english",
    lowercase=True,
    max_features=5000,
    ngram_range=(1, 2),   # include two-word phrases like "click here", "win prize"
)
```

Bigrams capture phrases that single words miss — "not bad" means something very different from "bad" alone.

## Dataset

**SMS Spam Collection** — 5,574 real SMS messages collected for research purposes.

- `ham` — 4,827 legitimate messages
- `spam` — 747 spam messages

Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
