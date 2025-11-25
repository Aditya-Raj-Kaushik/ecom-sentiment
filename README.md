🎧 E-Commerce & Device Review Sentiment Analysis

Hybrid ML + Rule-Based Model | FastAPI + Docker Deployment

This project builds a complete end-to-end automated sentiment analysis pipeline using real user reviews from:

Amazon Shopping App

Samsung Members App

The goal is to analyze customer satisfaction across the entire buyer journey, covering:

Purchase & delivery experience (Amazon app)

Post-purchase device performance (Samsung Members)

The system includes:

✔ Scraping → Cleaning → EDA → Model Training
✔ TF-IDF + Logistic Regression
✔ Rule-Boosting for critical complaints (“battery drain”, “heating”, “refund”, etc.)
✔ FastAPI deployment
✔ Docker containerization
✔ Business Insights + Visualizations

📌 1. Project Overview

Modern e-commerce customer satisfaction involves multiple touchpoints:

Before & during purchase – ordering, payment, delivery

After purchase – device experience, updates, performance

This project combines both perspectives by analyzing 1,476 real app reviews scraped from the Play Store.
We train a hybrid sentiment classifier and deploy it as a scalable API.

📁 2. Project Structure
ecom-sentiment/
│
├── data/
│   ├── raw/                  # Raw scraped reviews (Play Store)
│   └── processed/            # Cleaned + labeled CSV
│
├── models/
│   └── sentiment_model.pkl   # Saved ML model + TF-IDF vectorizer
│
├── reports/
│   ├── model_performance.txt # Accuracy, Precision, Recall, F1
│   └── plots/
│       └── confusion_matrix.png
│
├── src/
│   ├── scrape_playstore.py   # Scraper for Play Store reviews
│   ├── preprocess.py         # Cleaning, tokenization, lemmatization
│   ├── train_model.py        # TF-IDF + Logistic Regression
│   └── api_main.py           # FastAPI inference API (with rule boosting)
│
├── Dockerfile                # Full container setup
├── requirements.txt
└── README.md

🧹 3. Data Processing Pipeline
✔ Text Cleaning Steps

Lowercasing

Removing punctuation

Removing stopwords

Lemmatization

Tokenization

Word count extraction

✔ Sentiment Label Mapping
Rating ≥ 4 → Positive
Rating ≤ 2 → Negative
(Neutral reviews removed)

✔ Final Processed Dataset:

Rows: 1,476

Columns: app_name, review_text, clean_text, rating, sentiment, date, word_count

📊 4. Exploratory Data Analysis (EDA)

Generated insights include:

⭐ Common Pain Points:

Battery drain

Heating issues

Refund & delivery delays

Lag and performance drop after updates

⭐ Delight Factors:

Smooth UI/UX

Fast delivery

Good features

Helpful customer service

⭐ Visuals Saved:

Confusion Matrix → reports/plots/confusion_matrix.png

Word Clouds (optional)

Monthly trend charts (optional)

🤖 5. Model Training
Selected Model:

Logistic Regression + TF-IDF (n-grams up to 3)

Vocabulary size: 8,000 features

Why Logistic Regression?

✔ Fast
✔ Lightweight
✔ Highly interpretable
✔ Works extremely well with TF-IDF text vectors

Model Performance (Saved in model_performance.txt)
Metric	Score
Accuracy	~0.97
Precision	~0.96
Recall	~0.99
F1 Score	~0.98

The confusion matrix is saved automatically.

⚡ 6. Hybrid Rule-Boosted Sentiment Correction

ML alone sometimes misses device-specific negative patterns.
So we added rule boosting:

Hardcoded negative signals:
battery, drain, heating, overheating, lag, slow,
refund, fake, scam, not working, worst, useless


If any appear → model forces sentiment as Negative with confidence 0.99.

This dramatically increases production reliability.

🌐 7. FastAPI Deployment

The API:

Loads model + TF-IDF vectorizer

Applies rule-boosted sentiment logic

Returns probability scores

Includes interactive Swagger UI

Run locally:
uvicorn src.api_main:app --host 0.0.0.0 --port 8000 --reload


Swagger Docs:
👉 http://localhost:8000/docs

Example Request
{
  "review_text": "Battery drains out quickly after update"
}

Example Response
{
  "predicted_label": "Negative",
  "confidence": 0.99,
  "note": "Rule-boosted (keyword hit)"
}

🐳 8. Docker Deployment
Build Image
docker build -t sentiment-api .

Run Container
docker run -d -p 8000:8000 --name sentiment sentiment-api

Test API inside Docker:

👉 http://localhost:8000/docs

📝 9. How to Reproduce Full Pipeline
1️⃣ Preprocess Data
python src/preprocess.py

2️⃣ Train Model
python src/train_model.py


Outputs:

models/sentiment_model.pkl

reports/model_performance.txt

reports/plots/confusion_matrix.png

3️⃣ Run API
uvicorn src.api_main:app --reload

4️⃣ Build & Run Docker
docker build -t sentiment-api .
docker run -d -p 8000:8000 sentiment-api

📦 10. Business Value Delivered

This project provides:

✔ Real-time sentiment monitoring
✔ Product issue detection (battery drain, heating, etc.)
✔ Delivery and refund issue identification
✔ Insights usable by customer support, quality, and operations teams
✔ Scalable deployment ready for cloud (AWS, GCP, Azure)
🙌 11. Contributors

Aditya Raj Kaushik — Data Analyst & ML Engineer