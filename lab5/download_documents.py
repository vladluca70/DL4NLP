import os
import re
import time
import requests

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "TopicModelingHomework/1.0 (student project; contact: your_email@example.com)"
}

TOPICS = {
    "music": [
        "Music", "Jazz", "Rock music", "Classical music", "Hip hop music",
        "Pop music", "Blues", "Opera", "Heavy metal music", "Electronic music",
    ],
    "football": [
        "Association football", "FIFA World Cup", "UEFA Champions League",
        "Lionel Messi", "Cristiano Ronaldo", "Real Madrid CF", "FC Barcelona",
        "Premier League", "Offside (association football)",
        "Penalty kick (association football)",
    ],
    "school": [
        "School", "Primary school", "Secondary school", "Middle school",
        "Education", "Teacher", "Curriculum", "Boarding school",
        "Homeschooling", "Kindergarten",
    ],
}

DELAY = 2.0        # secunde între cereri
MAX_RETRIES = 6

def fetch_article(title: str, session: requests.Session) -> str:
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "titles": title,
    }
    for attempt in range(MAX_RETRIES):
        r = session.get(API, params=params, headers=HEADERS, timeout=30)
        if r.status_code == 429:
            # respectă Retry-After dacă există, altfel backoff exponențial
            wait = int(r.headers.get("Retry-After", 5 * 2 ** attempt))
            print(f"    429 pentru '{title}', aștept {wait}s "
                  f"(încercarea {attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait)
            continue
        r.raise_for_status()
        pages = r.json()["query"]["pages"]
        page = next(iter(pages.values()))
        return page.get("extract", "")
    raise RuntimeError(f"Eșec după {MAX_RETRIES} încercări: {title}")

def safe_name(title: str) -> str:
    return re.sub(r"[^\w]+", "_", title).strip("_").lower()

def main(out_dir="data"):
    total = 0
    with requests.Session() as session:
        for topic, titles in TOPICS.items():
            folder = os.path.join(out_dir, topic)
            os.makedirs(folder, exist_ok=True)
            for title in titles:
                path = os.path.join(folder, f"{safe_name(title)}.txt")

                # sare peste ce există deja
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    print(f"[skip] {topic:9s} {title}")
                    total += 1
                    continue

                text = fetch_article(title, session)
                if not text:
                    print(f"[!] Gol: {title}")
                    continue
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
                total += 1
                print(f"[ok] {topic:9s} {title:40s} {len(text):>8d} caractere")
                time.sleep(DELAY)

    print(f"\nDocumente disponibile: {total} în '{out_dir}/'")

if __name__ == "__main__":
    main()