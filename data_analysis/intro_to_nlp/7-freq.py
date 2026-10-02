#!/usr/bin/env python3
"""Find and plot most frequent words in the cleaned corpus."""
import nltk
import matplotlib.pyplot as plt


def plot_top_n_frequencies(corpus_tokens, n=20):
    """Count every token in the corpus and draw bar chart of n-most
    common ones. Returns full frequency distribution."""
    # one big list with all tokens
    words = []
    for tokens in corpus_tokens:
        for token in tokens:
            words.append(token)

    # count them
    freq = nltk.FreqDist(words)

    # the n-most common words & their counts
    x = []
    y = []
    for word, count in freq.most_common(n):
        x.append(word)
        y.append(count)

    # bar chart
    plt.figure(figsize=(12, 5))
    plt.bar(x, y)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Top {n} Most Frequent Words")
    plt.xlabel("Word")
    plt.ylabel("Frequency")
    plt.tight_layout()

    # Return all counts so callers can inspect more than the plotted words.
    return freq
