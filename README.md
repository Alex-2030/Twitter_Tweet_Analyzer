# Twitter Tweet Analyzer

A machine learning project that analyzes tweets in two ways:

- **Supervised learning:** classify the sentiment of a tweet as Positive, Negative, Neutral, or Irrelevant.
- **Unsupervised learning:** discover the themes in a large collection of tweets with topic modeling and clustering.

The project also includes a full exploratory data analysis (EDA): sentiment distribution, text length, word and n-gram frequency, word clouds, hashtags, mentions, and engagement-style metrics.

Everything lives in one notebook, `Final_project_ML.ipynb`, and the datasets are included, so you can clone the repo and run it straight away.

## Quick start

```bash
git clone https://github.com/Alex-2030/Twitter_Tweet_Analyzer.git
cd Twitter_Tweet_Analyzer

python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install pandas numpy scikit-learn imbalanced-learn nltk matplotlib seaborn wordcloud networkx pyldavis joblib jupyter

jupyter notebook Final_project_ML.ipynb
```

Run all cells in order. The NLTK data (`wordnet` and the POS tagger) downloads automatically in the first cells.

**Requirements:** Python 3.10 or newer (developed on 3.13) and Jupyter.

> The notebook is committed with its outputs cleared, so you will see the plots and metrics only after you run it.

## Dataset

Two CSV files in the project root, with no header row:

| File | Tweets |
|---|---|
| `twitter_training.csv` | 74,682 |
| `twitter_validation.csv` | 1,000 |

| Column | Description |
|---|---|
| `id` | Tweet ID |
| `topic` | Entity or topic the tweet refers to (for example, Borderlands) |
| `label` | Sentiment: `Positive`, `Negative`, `Neutral`, or `Irrelevant` |
| `text` | Tweet text |

The data comes from the **Twitter Entity Sentiment Analysis** dataset on Kaggle, published by [jp797498e](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis). It is an entity-level sentiment dataset: given a tweet and an entity, the task is to judge the sentiment of the tweet about that entity. See the [Acknowledgements](#acknowledgements) section for credit and license details.

## Pipeline

1. **Preprocessing:** remove URLs, mentions, hashtags, emojis, and special characters, then tokenize with NLTK's `TweetTokenizer` and lemmatize using part-of-speech tags (`WordNetLemmatizer`).
2. **Vectorization:** TF-IDF with unigrams and bigrams (`ngram_range=(1, 2)`, `min_df=5`, `max_df=0.95`, sublinear term frequency). Labels are encoded with `LabelEncoder`.
3. **EDA:** sentiment distributions, text length analysis, top words and n-grams per sentiment, word clouds, hashtag and mention analysis (including a mention network graph), and engagement metrics.
4. **Supervised models:** Logistic Regression, Random Forest, and a calibrated Linear SVM. The notebook checks class imbalance and applies SMOTE only if the imbalance ratio is above 2.
5. **Unsupervised models:** LDA (15 and 20 topics), NMF (15 topics), and KMeans clustering (k=10) on the TF-IDF vectors, with pyLDAvis for interactive topic exploration.
6. **Prediction:** `predict_sentiment()` takes a raw tweet and returns the predicted sentiment with a confidence score.

## Results

Accuracy on the 1,000-tweet validation set from a full run of the notebook:

| Model | Accuracy |
|---|---|
| Logistic Regression | 96.9% |
| Linear SVM (calibrated) | 97.8% |
| Random Forest (150 trees, max depth 20) | 56.5% |

The training data's class imbalance ratio is about 1.74, so SMOTE is not triggered. Exact numbers may vary slightly with library versions.

## Using the saved models

The "Save Models" cells write these files to the project folder:

- `logistic_regression_model.pkl`
- `random_forest_model.pkl`
- `svm_model.pkl`
- `tfidf_vectorizer.pkl`
- `label_encoder.pkl`
- `lda_model.pkl`
- `nmf_model_model.pkl`

They are not stored in the repo (`*.pkl` is git-ignored), so run the notebook once to generate them. To reuse them without retraining:

```python
import joblib

model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")
```

Run each tweet through the notebook's preprocessing function before calling `vectorizer.transform(...)`.

## Project structure

```
.
├── Final_project_ML.ipynb   # Full pipeline: preprocessing, EDA, supervised and unsupervised ML
├── twitter_training.csv     # Training data
├── twitter_validation.csv   # Validation data
└── .gitignore
```

## Tech stack

Python, pandas, NumPy, scikit-learn, imbalanced-learn, NLTK, matplotlib, seaborn, WordCloud, NetworkX, pyLDAvis, joblib

## Acknowledgements

- **Dataset:** [Twitter Entity Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis) by jp797498e on Kaggle. `twitter_training.csv` and `twitter_validation.csv` are the training and validation files from that dataset, included here unmodified for convenience. The dataset is listed on Kaggle under the CC0-1.0 license; check its Kaggle page for the current license terms.
