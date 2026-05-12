# Practical No: 7
# Title: Text Analytics
#
# 1. Extract Sample document and apply following document preprocessing methods:
#    Tokenization, POS Tagging, stop words removal, Stemming and Lemmatization.
# 2. Create representation of documents by calculating Term Frequency (TF)
#    and Inverse Document Frequency (IDF).

import nltk
import re
import math
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# ── Sample Text ──────────────────────────────────────────────
text = "How to remove stop words with NLTK library in Python?"

# ── 1. Sentence Tokenization ─────────────────────────────────
print("\nSentence Tokenization:")
print(sent_tokenize(text))

# ── 2. Word Tokenization ─────────────────────────────────────
print("\nWord Tokenization:")
print(word_tokenize(text))

# ── 3. POS Tagging ───────────────────────────────────────────
data = "The pink sweater fit her perfectly"
print("\nPOS Tagging:")
tokens_pos = word_tokenize(data)
print(nltk.pos_tag(tokens_pos))

# ── 4. Stop Words Removal ────────────────────────────────────
print("\nHow to remove stop words with NLTK library in Python?")
stop_words = set(stopwords.words('english'))
clean_text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
tokens = word_tokenize(clean_text)
filtered = [w for w in tokens if w not in stop_words]
print("Filtered Words:", filtered)

# ── 5. Stemming ──────────────────────────────────────────────
ps = PorterStemmer()
words = ['wait', 'waiting', 'waited', 'waits']
print("\nStemming:")
for w in words:
    print(w, '->', ps.stem(w))

# ── 6. Lemmatization ─────────────────────────────────────────
lemmatizer = WordNetLemmatizer()
text2 = "studies studying cries cry"
print("\nLemmatization:")
for w in word_tokenize(text2):
    print(w, '->', lemmatizer.lemmatize(w))

# ── 7. TF-IDF Representation ─────────────────────────────────

# Sample documents
docA = "Jupiter is the largest planet"
docB = "Mars is the fourth planet from the sun"

bagA = docA.split()
bagB = docB.split()

unique_words = set(bagA).union(set(bagB))

# Word frequency dictionaries
wordDictA = dict.fromkeys(unique_words, 0)
wordDictB = dict.fromkeys(unique_words, 0)

for w in bagA:
    wordDictA[w] += 1

for w in bagB:
    wordDictB[w] += 1

# ── TF ───────────────────────────────────────────────────────
def computeTF(wordDict, bag):
    tf = {}
    total = len(bag)
    for word in wordDict:
        tf[word] = wordDict[word] / total
    return tf

tfA = computeTF(wordDictA, bagA)
tfB = computeTF(wordDictB, bagB)

# ── IDF ──────────────────────────────────────────────────────
def computeIDF(docs):
    N = len(docs)
    idf = dict.fromkeys(docs[0].keys(), 0)
    for doc in docs:
        for word in doc:
            if doc[word] > 0:
                idf[word] += 1
    for word in idf:
        idf[word] = math.log(N / idf[word])
    return idf

idf = computeIDF([wordDictA, wordDictB])

# ── TF-IDF ───────────────────────────────────────────────────
def computeTFIDF(tf, idf):
    tfidf = {}
    for word in tf:
        tfidf[word] = tf[word] * idf[word]
    return tfidf

tfidfA = computeTFIDF(tfA, idf)
tfidfB = computeTFIDF(tfB, idf)

df = pd.DataFrame([tfidfA, tfidfB])
print("\nTF-IDF MATRIX:")
print(df)
