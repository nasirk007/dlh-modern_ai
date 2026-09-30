#!/usr/bin/env python3
"""Draw word cloud from the cleaned corpus."""
import wordcloud
import matplotlib.pyplot as plt


def generate_wordcloud(corpus_tokens, max_words=200, label=None):
    """Join all tokens into one text and draw it as a word cloud.
    Returns the fitted WordCloud object."""
    # one big string with all tokens
    words = []
    for tokens in corpus_tokens:
        for token in tokens:
            words.append(token)
    text = " ".join(words)

    # build word cloud
    WC = wordcloud.WordCloud(max_words=max_words, background_color="white",
                             width=800, height=400, random_state=42)
    WC.generate(text)

    # show it
    plt.figure(figsize=(10, 5))
    plt.imshow(WC, interpolation="bilinear")
    plt.axis("off")
    if label:
        plt.title(f"WordCloud — {label}")
    else:
        plt.title("WordCloud")
    plt.tight_layout()

    return WC
