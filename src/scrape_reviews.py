import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
}

def scrape_trustpilot_reviews(base_url, max_pages=15):
    all_reviews = []

    for page in range(1, max_pages + 1):
        url = f"{base_url}?page={page}"
        print(f"[INFO] Scraping Trustpilot page {page}")

        resp = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(resp.text, "html.parser")

        reviews = soup.find_all("article")
        if not reviews:
            print("[INFO] No more reviews found.")
            break

        for r in reviews:
            title_tag = r.find("h2")
            title = title_tag.text.strip() if title_tag else None

            text_tag = r.find("p")
            text = text_tag.text.strip() if text_tag else None

            rating_tag = r.find("img", {"class": "star-rating_image"})
            rating = None
            if rating_tag:
                try:
                    rating = float(rating_tag["alt"].split()[0])
                except:
                    rating = None

            date_tag = r.find("time")
            date = date_tag["datetime"] if date_tag else None

            all_reviews.append({
                "source": "Trustpilot",
                "review_title": title,
                "review_text": text,
                "rating": rating,
                "review_date_raw": date
            })

        time.sleep(random.uniform(1, 2))

    df = pd.DataFrame(all_reviews)
    return df


if __name__ == "__main__":
    url = "https://www.trustpilot.com/review/amazon.in"
    df = scrape_trustpilot_reviews(url, max_pages=20)
    print(f"Total reviews scraped: {len(df)}")
    df.to_csv("data/raw/trustpilot_amazon_reviews.csv", index=False)
