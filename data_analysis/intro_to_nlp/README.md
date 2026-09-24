Introduction to Natural Language Processing

Natural Language Processing (NLP) is the area of artificial intelligence that
enables computers to work with human language: reading it, organizing it,
extracting meaning from it, and generating useful responses. This module uses
Python to turn raw text into data that can be explored, measured, and used by
machine learning models.

Why NLP Matters

Text is one of the largest sources of human-generated data, but it is
unstructured and full of variation. The same idea can be written with
different words, spellings, formats, or levels of context. NLP provides the
methods needed to clean that text, represent it numerically, identify useful
patterns, and build systems that can make reliable predictions from language.

NLP, Machine Learning, and Deep Learning

NLP is a problem domain, while machine learning (ML) and deep learning (DL)
are approaches that can be used to solve NLP problems. Traditional ML often
depends on features designed or selected from text, such as word counts,
TF-IDF scores, or n-grams, and then applies models such as Naive Bayes or
logistic regression. Deep learning learns richer representations from data
using neural networks, including word embeddings, recurrent networks, and
transformers. In short, NLP describes what kind of data and tasks we are
working with; ML and DL describe how a system learns from that data.

Real-World Applications

- Spam detection and email filtering
- Search engines and document retrieval
- Sentiment analysis for reviews and social media
- Chatbots, virtual assistants, and question answering
- Machine translation and speech-enabled applications
- Named Entity Recognition for people, places, organizations, and dates
- Text summarization, topic discovery, and document classification
- Content moderation, recommendation, and fraud or threat detection

Learning Objectives

By the end of this module, you should be able to explain these concepts and
apply them to a text dataset:

- NLP: Understand how computers process and learn from human language.
- Text exploration: Inspect text length, labels, vocabulary, duplicates,
	and other patterns before modeling.
- Normalization: Clean inconsistent casing, punctuation, spacing, and
	symbols so equivalent text is treated consistently.
- spaCy and NLTK: Choose between a production-oriented NLP toolkit and a
	flexible learning and experimentation toolkit.
- Tokenization: Split text into meaningful units such as words, sentences,
	or subwords.
- Stemming and lemmatization: Reduce related word forms, using a faster
	heuristic stemmer or a linguistically informed lemma.
- Vectorization: Convert text into numerical features that algorithms can
	process.
- Bag of Words and TF-IDF: Compare simple word-count features with scores
	that reduce the importance of common words.
- Word embeddings: Represent words as dense vectors where useful semantic
	and contextual relationships can be learned.
- Word2Vec and GloVe: Learn or use vector representations based on the
	words that appear around one another.
- Named Entity Recognition: Detect and classify real-world entities in
	text, such as names, locations, and organizations.
- Sentiment analysis: Estimate the opinion or emotional polarity expressed
	in a piece of text.
- Topic modeling: Discover recurring themes in a collection of documents.
- Text classification: Build a complete pipeline from raw text to a
	trained classifier and evaluation metrics.
- What's next in NLP: Connect these foundations to modern neural language
	models, transformers, and generative AI.

Tasks

The practical tasks move from understanding the dataset to creating useful
text representations:

0. Basic Exploration
1. Text Normalization
2. Tokenization
3. Stopword Removal
4. Filtering
5. Lemma vs Stem
6. N-gram
7. Word Frequency Distribution
8. Wordclouds
9. Bag of Words (BoW)
10. TF-IDF
11. Word2Vec
12. FastText

Resources

Read or Watch

- [Text Preprocessing in Python](https://intranet-dlh.hbtn.io/rltoken/tdus7BkhzkonGMMQztMRvw)
- [NLTK Tokenization Tutorial](https://intranet-dlh.hbtn.io/rltoken/uMpF7zV5RwCTpSrgNu3Y5g)
- [Stemming vs Lemmatization](https://intranet-dlh.hbtn.io/rltoken/1teA3x8Dwitxrrq0qC6yVw)
- [Understanding N-grams](https://intranet-dlh.hbtn.io/rltoken/73T5DItQGFWqOWAWbhpm0w)
- [Bag of Words](https://intranet-dlh.hbtn.io/rltoken/h2aEPlXHGcD2ebwBrg4ZlA)
- [TF-IDF Explained Simply](https://intranet-dlh.hbtn.io/rltoken/Dm2-fjQFB2aXwUO9y0BPSA)
- [Illustrated Word2Vec](https://intranet-dlh.hbtn.io/rltoken/DeMBRwxexW6txJkAP_a-Vw)
- [FastText](https://intranet-dlh.hbtn.io/rltoken/poMsjQQ9AxEbDdnMAr1msw)
- [Naive Bayes for Text Classification](https://intranet-dlh.hbtn.io/rltoken/sd_s65T2SXs8pGAlGfQwoA)
- [Precision, Recall, F1](https://intranet-dlh.hbtn.io/rltoken/3F_gbJ99EqdawifLx4P_Ug)
- [Text Classification Pipeline - NLTK](https://intranet-dlh.hbtn.io/rltoken/hBALggNPvpcIVJyQQPIAdA)
- [Word2Vec](https://intranet-dlh.hbtn.io/rltoken/M-tgHMtgFBN4QGqMDeXm6A)

References

- [NLTK Documentation](https://intranet-dlh.hbtn.io/rltoken/VsB2X6lUqChJJuW9Ns96Pg)
- [Emoji](https://intranet-dlh.hbtn.io/rltoken/6Z9Kdlaj6d1r9LjRNgc-iQ)
- [scikit-learn Text Feature Extraction](https://intranet-dlh.hbtn.io/rltoken/4AFWOvI6r-0pclBfw_c8IQ)
- [Vectorization and Word Embeddings - Gensim](https://intranet-dlh.hbtn.io/rltoken/rE02cyOwFLjtCq-m1wBWBw)
- [Gensim Word2Vec](https://intranet-dlh.hbtn.io/rltoken/hmXOhTQdM1OIxKK8bm_Hjw)
- [Gensim FastText](https://intranet-dlh.hbtn.io/rltoken/jsM2EEC8HzaEwYPWBim9Xg)
- [WordCloud Library](https://intranet-dlh.hbtn.io/rltoken/n-q1tgmWL4fs36a-LL05sA)
- [scikit-learn Naive Bayes](https://intranet-dlh.hbtn.io/rltoken/QQISLvDgWYNQTcETCM0mFw)
- [scikit-learn Metrics](https://intranet-dlh.hbtn.io/rltoken/XTf97TLalxdFfUDuMEFY-w)
- [Regular Expressions in Python](https://intranet-dlh.hbtn.io/rltoken/RsJ32lro6rT2Ti18PqY_OA)

Requirements

- Ubuntu 20.04 LTS with `python3` 3.11
- Python files must begin with `!/usr/bin/env python3`, end with a newline,
	and be executable.
- Code must follow `pycodestyle` 2.14.0.
- Modules, classes, and functions must include documentation strings.
- Required packages and versions: `numpy` 2.0.2, `regex` 2024.11.6,
	`pandas` 2.2.2, `scikit-learn` 1.6.1, `matplotlib` 3.10.0,
	`seaborn` 0.13.2, `wordcloud` 1.9.4, `nltk` 3.9.1, `emoji` 2.15.0,
	and `gensim` 4.4.0.

NLTK Setup

Download the language resources used by the exercises:

```python
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

Dataset

The exercises use the [SMSSpamCollection dataset](https://intranet-dlh.hbtn.io/rltoken/zM-EWesZt1b9dHC7p3yHWg),
a labelled collection of SMS messages classified as `ham` or `spam`.

The original file contains malformed quotation marks and duplicate messages.
When loading it with pandas, use `quoting=csv.QUOTE_NONE` so embedded quotes do
not merge rows, strip only the unwanted surrounding quotes, and remove
duplicates before saving the cleaned dataset. This preparation matters because
bad parsing or repeated examples can produce misleading row counts and model
evaluation results.

```python
import csv
import pandas as pd

df = pd.read_csv(
		'SMSSpamCollection_original',
		sep='\t',
		names=['label', 'message'],
		quoting=csv.QUOTE_NONE,
)
df['message'] = df['message'].str.strip('"')
df = df.drop_duplicates(ignore_index=True)
df.to_csv('SMSSpamCollection', sep='\t', index=False, header=False)
```

The cleaned dataset contains 5,114 unique messages and is used throughout the
subsequent tasks.
