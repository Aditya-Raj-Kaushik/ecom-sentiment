from google_play_scraper import reviews, Sort
import pandas as pd

def scrape_reviews(app_id, app_name, max_reviews=600):
    all_reviews = []
    count = 0
    batch_size = 200

    while count < max_reviews:
        result, _ = reviews(
            app_id,
            lang="en",
            country="in",
            sort=Sort.NEWEST,
            count=batch_size,
            filter_score_with=None
        )
        if not result:
            break

        for r in result:
            all_reviews.append({
                "app_name": app_name,
                "review_text": r.get("content", ""),
                "rating": r.get("score"),
                "review_date": r.get("at"),
            })

        count += len(result)
        print(f"[INFO] Fetched {len(result)} reviews... Total: {count}")

    return pd.DataFrame(all_reviews)


if __name__ == "__main__":
    apps = [
        ("in.amazon.mShop.android.shopping", "Amazon Shopping"),
        ("com.samsung.android.voc", "Samsung Members")
    ]

    for app_id, app_name in apps:
        df = scrape_reviews(app_id, app_name, max_reviews=700)
        df.to_csv(f"data/raw/{app_name.replace(' ', '_').lower()}_reviews.csv", index=False)
        print(f"[SUCCESS] Saved {app_name} ({len(df)} reviews)")
