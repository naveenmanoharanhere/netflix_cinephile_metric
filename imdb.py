import requests
import pandas as pd
import time
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

TMDB_API_KEY = "nice try lol"

INPUT_FILE = Path(
    r"whatever man"
)

OUTPUT_FILE = Path(
    r"or woman"
)

BASE_URL = "https://api.themoviedb.org/3"
REQUEST_DELAY = 60 

def get_session():
    retry = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False
    )

    adapter = HTTPAdapter(max_retries=retry)

    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update({
        "User-Agent": "Mozilla/5.0 (cinephile-metric/1.0)",
        "Accept": "application/json"
    })

    return session


SESSION = get_session()

def search_movie(title, year=None):
    params = {
        "api_key": TMDB_API_KEY,
        "query": title
    }
    if year and not pd.isna(year):
        params["year"] = int(year)

    try:
        r = SESSION.get(
            f"{BASE_URL}/search/movie",
            params=params,
            timeout=20
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        return results[0]["id"] if results else None

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] TMDB search failed for '{title}': {e}")
        return None


def fetch_reviews(movie_id):
    try:
        r = SESSION.get(
            f"{BASE_URL}/movie/{movie_id}/reviews",
            params={"api_key": TMDB_API_KEY},
            timeout=20
        )
        r.raise_for_status()
        return r.json().get("results", [])

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Review fetch failed for movie_id={movie_id}: {e}")
        return []

def main():
    print("Loading Netflix titles...")
    df = pd.read_csv(INPUT_FILE)

    all_reviews = []

    for idx, row in df.iterrows():
        title = row["title"]
        year = row.get("year")

        print(f"\n({idx + 1}/{len(df)}) Searching TMDB: {title}")
        movie_id = search_movie(title, year)

        if not movie_id:
            print(f"[SKIP] No TMDB match for '{title}'")
            time.sleep(REQUEST_DELAY)
            continue

        reviews = fetch_reviews(movie_id)
        print(f"Found {len(reviews)} reviews")

        for r in reviews:
            all_reviews.append({
                "title": title,
                "author": r.get("author"),
                "rating": r.get("author_details", {}).get("rating"),
                "content": r.get("content"),
                "created_at": r.get("created_at")
            })

        time.sleep(REQUEST_DELAY)

    if not all_reviews:
        print("No reviews fetched. Check connectivity or API limits.")
        return

    out_df = pd.DataFrame(all_reviews)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved {len(out_df)} reviews to:")
    print(f"{OUTPUT_FILE}")

if __name__ == "__main__":
    main()
