import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = nltk.word_tokenize(text)
    
    cleaned = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in stop_words and len(t) > 2
    ]
    return " ".join(cleaned)

def sentiment_from_rating(r):
    if r >= 4:
        return 1  # Positive
    elif r <= 2:
        return 0  # Negative
    else:
        return np.nan  # Neutral → removed

if __name__ == "__main__":
    df1 = pd.read_csv("data/raw/amazon_shopping_reviews.csv")
    df2 = pd.read_csv("data/raw/samsung_members_reviews.csv")

    df = pd.concat([df1, df2], ignore_index=True)

    # Drop NA text
    df.dropna(subset=["review_text"], inplace=True)

    # Label sentiment
    df["sentiment"] = df["rating"].apply(sentiment_from_rating)
    df = df.dropna(subset=["sentiment"])
    df["sentiment"] = df["sentiment"].astype(int)

    # Clean text
    df["clean_text"] = df["review_text"].astype(str).apply(clean_text)

    # Word count for insights
    df["word_count"] = df["clean_text"].apply(lambda x: len(x.split()))

    # Review month
    df["review_month"] = pd.to_datetime(df["review_date"]).dt.to_period("M").astype(str)

    df.to_csv("data/processed/reviews_processed.csv", index=False)
    print(df.head())
    print(f"Saved {len(df)} cleaned & labeled reviews")
