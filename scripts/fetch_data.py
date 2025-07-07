import threading
import time
import sqlite3
import tweepy
import os
from datetime import datetime

BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAD%2Fw2gEAAAAAz9EqrUB4Bmg1rEvlRIFvIxGZbn8%3DFnfbqrxb3HivEBe4REjdyyZubO2D0aDduU4OwBfWhNkwl7BvOc"
DB_PATH = "db/new_data.db"
os.makedirs("db", exist_ok=True)
lock = threading.Lock()

client = tweepy.Client(bearer_token=BEARER_TOKEN)

def fetch_tweets(keyword, results, max_results=20):
    query = f"{keyword} -is:retweet lang:en"
    try:
        response = client.search_recent_tweets(query=query, max_results=max_results, tweet_fields=["created_at"])
        if response.data:
            with lock:
                for tweet in response.data:
                    results.append((tweet.id, keyword, tweet.text, tweet.created_at))
            print(f"Fetched {len(response.data)} for {keyword}")
    except Exception as e:
        print(f"Error fetching tweets for {keyword}: {e}")

def save_to_db(tweets):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS tweets (
        id TEXT PRIMARY KEY,
        keyword TEXT,
        text TEXT,
        created_at TEXT
    )""")
    for tweet in tweets:
        c.execute("INSERT OR IGNORE INTO tweets (id, keyword, text, created_at) VALUES (?, ?, ?, ?)", tweet)
    conn.commit()
    conn.close()

def fetch_periodically():
    while True:
        print(f"\n=== Fetching tweets at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
        keywords = ["AI", "movies", "climate", "education"]
        threads = []
        results = []

        for kw in keywords:
            t = threading.Thread(target=fetch_tweets, args=(kw, results))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        save_to_db(results)
        print(f"Saved {len(results)} tweets\nSleeping for 24 hours...\n")
        time.sleep(86400) 

if __name__ == "__main__":
    fetch_periodically()

