import os
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

for pkg in ("stopwords", "wordnet", "omw-1.4"):
    nltk.download(pkg, quiet=True)

DATA_DIR = "data"
USE_LEMMATIZATION = True   # False => stemming (Porter)

# ---------- 1. Încărcare documente ----------
def load_documents(data_dir=DATA_DIR):
    texts, labels, names = [], [], []
    for topic in sorted(os.listdir(data_dir)):
        folder = os.path.join(data_dir, topic)
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            with open(os.path.join(folder, fname), encoding="utf-8") as f:
                texts.append(f.read())
            labels.append(topic)
            names.append(f"{topic}/{fname}")
    return texts, labels, names

# ---------- 2. Curățare ----------
CUT_SECTIONS = re.compile(
    r"\n==\s*(See also|References|Notes|Footnotes|External links|"
    r"Further reading|Sources|Bibliography|Citations)\s*==.*",
    flags=re.DOTALL | re.IGNORECASE,
)

def clean_wikipedia(text):
    text = CUT_SECTIONS.sub("", text)          # elimină secțiunile de final
    text = re.sub(r"={2,}.*?={2,}", " ", text)  # elimină titlurile == ... ==
    return text

# ---------- 3. Preprocesare ----------
STOP = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocess(text):
    text = clean_wikipedia(text).lower()
    tokens = re.findall(r"[a-z]+", text)                       # doar litere
    tokens = [t for t in tokens if len(t) > 2 and t not in STOP]
    if USE_LEMMATIZATION:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    else:
        tokens = [stemmer.stem(t) for t in tokens]
    tokens = [t for t in tokens if t not in STOP]              # a doua trecere
    return " ".join(tokens)

# ---------- 4. Vectorizare ----------
def main():
    texts, labels, names = load_documents()
    print(f"Documente: {len(texts)}")

    # verificare duplicate
    assert len(set(texts)) == len(texts), "Ai documente duplicate!"

    docs = [preprocess(t) for t in texts]
    print(f"Exemplu ({names[0]}): {docs[0][:200]}...\n")

    # Bag-of-Words
    bow_vec = CountVectorizer(min_df=2, max_df=0.8)
    X_bow = bow_vec.fit_transform(docs)

    # TF-IDF
    tfidf_vec = TfidfVectorizer(min_df=2, max_df=0.8)
    X_tfidf = tfidf_vec.fit_transform(docs)

    print(f"BoW    : {X_bow.shape}  (documente x termeni)")
    print(f"TF-IDF : {X_tfidf.shape}")
    print(f"Sparsitate BoW: {1 - X_bow.nnz / (X_bow.shape[0] * X_bow.shape[1]):.2%}")

    # cei mai frecvenți termeni pe fiecare temă
    import numpy as np
    terms = np.array(bow_vec.get_feature_names_out())
    for topic in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == topic]
        top = np.asarray(X_bow[idx].sum(axis=0)).ravel().argsort()[::-1][:10]
        print(f"{topic:9s}: {', '.join(terms[top])}")

    # salvare pentru pașii următori
    with open("vectors.pkl", "wb") as f:
        pickle.dump({
            "docs": docs, "labels": labels, "names": names,
            "bow_vec": bow_vec, "X_bow": X_bow,
            "tfidf_vec": tfidf_vec, "X_tfidf": X_tfidf,
        }, f)
    print("\nSalvat în vectors.pkl")

if __name__ == "__main__":
    main()
