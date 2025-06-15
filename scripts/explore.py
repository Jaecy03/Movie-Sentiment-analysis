import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

df = pd.read_csv('data/cleaned_reviews.csv')

sns.countplot(data=df, x='label')
plt.title("Class Distribution")
plt.xlabel("Label (0 = Negative, 1 = Positive)")
plt.ylabel("Count")
plt.show()

df['text_length'] = df['clean_text'].apply(len)
df['text_length'].plot.hist(bins=30, title="Text Length Distribution")
plt.xlabel("Number of characters")
plt.ylabel("Frequency")
plt.show()

df['word_count'] = df['clean_text'].apply(lambda x: len(str(x).split()))
print("\nWord Count Stats:\n", df['word_count'].describe())

all_words = " ".join(df['clean_text'].astype(str)).lower()
vocab = set(re.findall(r'\b\w+\b', all_words))
print(f"\nVocabulary Size: {len(vocab)}")

def get_top_words(texts, n=20):
    """Return top n most frequent words from a list of text."""
    words = " ".join(texts).lower()
    words = re.findall(r'\b\w+\b', words)
    return Counter(words).most_common(n)

top_words = get_top_words(df['clean_text'])
print("\nTop 20 Words:")
for word, freq in top_words:
    print(f"{word}: {freq}")

missing = df['clean_text'].isnull().sum()
empty = (df['clean_text'].str.strip() == '').sum()
print(f"\nMissing entries: {missing}")
print(f"Empty entries: {empty}")

