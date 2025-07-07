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
plt.figure(figsize=(8, 1))
sns.boxplot(x=df['word_count'], color='lightgreen')
plt.title("Boxplot of Word Counts")
plt.xlabel("Number of words")
plt.show()

all_words = " ".join(df['clean_text'].astype(str)).lower()
all_words_list = re.findall(r'\b\w+\b', all_words)
total_word_count = len(all_words_list)
unique_word_count = len(set(all_words_list))
print(f"\nVocabulary Size: {unique_word_count}")

plt.figure(figsize=(6, 4))
plt.bar(['Total Words', 'Unique Words'], [total_word_count, unique_word_count], color=['orange', 'purple'])
plt.title("Total vs Unique Words in Dataset")
plt.ylabel("Count")
plt.show()


def get_top_words(texts, n=20):
    """Return top n most frequent words from a list of text."""
    words = " ".join(texts).lower()
    words = re.findall(r'\b\w+\b', words)
    return Counter(words).most_common(n)

top_words = get_top_words(df['clean_text'])
print("\nTop 20 Words:")
for word, freq in top_words:
    print(f"{word}: {freq}")
top_words_df = pd.DataFrame(top_words, columns=['word', 'frequency'])

plt.figure(figsize=(10, 6))
sns.barplot(data=top_words_df, x='frequency', y='word', palette='viridis')
plt.title("Top 20 Most Frequent Words")
plt.xlabel("Frequency")
plt.ylabel("Word")
plt.show()

missing = df['clean_text'].isnull().sum()
empty = (df['clean_text'].str.strip() == '').sum()
print(f"\nMissing entries: {missing}")
print(f"Empty entries: {empty}")

