# Task 2 - Topic modeling: LSA, NMF, LDA pe BoW si TF-IDF (k=3)
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.metrics import adjusted_rand_score

K = 3          # numar de topicuri
TOP_N = 10     # cate cuvinte afisam pe topic
SEED = 0
METHODS = ["LSA", "NMF", "LDA"]
os.makedirs("plots", exist_ok=True)


def load_data(path="vectors.pkl"):
    # datele salvate de task1.py
    with open(path, "rb") as f:
        return pickle.load(f)


def make_model(method, k):
    if method == "LSA":
        return TruncatedSVD(n_components=k, random_state=SEED)
    if method == "NMF":
        return NMF(n_components=k, init="nndsvd", max_iter=500, random_state=SEED)
    return LatentDirichletAllocation(n_components=k, max_iter=50, random_state=SEED)


def fit_model(method, X, k):
    model = make_model(method, k)
    doc_topic = model.fit_transform(X)   # documente x topicuri
    comp = model.components_             # topicuri x cuvinte
    return doc_topic, comp


def top_idx(comp, n=TOP_N):
    # indicii celor mai importanti cuvinte pe topic (valoare absoluta, pt. LSA cu semne mixte)
    return np.argsort(-np.abs(comp), axis=1)[:, :n]


def top_words(comp, terms, n=TOP_N):
    return [[terms[i] for i in row] for row in top_idx(comp, n)]


def assign_topics(doc_topic):
    # topicul cu cel mai mare scor absolut (cerinta pentru LSA; pentru NMF/LDA nu schimba nimic)
    return np.abs(doc_topic).argmax(axis=1)


def purity(labels, assigned):
    # cat de "pure" sunt topicurile fata de domeniile reale
    table = pd.crosstab(assigned, labels)
    return table.max(axis=1).sum() / len(labels)


def ari(labels, assigned):
    return adjusted_rand_score(labels, assigned)


# Pentru grafice: LSA sare peste prima componenta (e mai mult "frecventa generala")
DIMS = {"LSA": (1, 2), "NMF": (0, 1), "LDA": (0, 1)}


def plot_words(comp, terms, method, title, path):
    d = DIMS[method]
    emb = comp.T[:, list(d)]                       # fiecare cuvant = un punct
    plt.figure(figsize=(8, 6))
    plt.scatter(emb[:, 0], emb[:, 1], s=3, c="lightgray")
    colors = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple"]
    for t, row in enumerate(top_idx(comp)):        # etichetam doar top cuvintele
        for i in row:
            plt.scatter(emb[i, 0], emb[i, 1], c=colors[t % 5], s=15)
            plt.annotate(terms[i], emb[i], fontsize=7)
    plt.title(title + " - cuvinte")
    plt.xlabel(f"dimensiunea {d[0]}")
    plt.ylabel(f"dimensiunea {d[1]}")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_docs(doc_topic, labels, assigned, method, title, path):
    d = DIMS[method]
    emb = doc_topic[:, list(d)]
    labels = np.array(labels)
    colors = {l: c for l, c in zip(sorted(set(labels)), ["tab:red", "tab:blue", "tab:green"])}
    markers = ["o", "s", "^", "D", "v"]            # forma = topicul gasit
    plt.figure(figsize=(8, 6))
    for l in colors:
        for t in sorted(set(assigned)):
            m = (labels == l) & (assigned == t)
            if m.any():
                plt.scatter(emb[m, 0], emb[m, 1], c=colors[l], marker=markers[t % 5],
                            s=50, label=f"{l} / topic {t}")
    plt.legend(fontsize=7)
    plt.title(title + " - documente (culoare = domeniu real)")
    plt.xlabel(f"dimensiunea {d[0]}")
    plt.ylabel(f"dimensiunea {d[1]}")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    data = load_data()
    labels = data["labels"]
    terms = np.array(data["bow_vec"].get_feature_names_out())  # acelasi vocabular la ambele matrici
    matrices = {"BoW": data["X_bow"], "TF-IDF": data["X_tfidf"]}

    for rep, X in matrices.items():
        for method in METHODS:
            title = f"{method} pe {rep} (k={K})"
            print("\n" + "=" * 60 + f"\n{title}\n" + "=" * 60)

            doc_topic, comp = fit_model(method, X, K)

            # top 10 cuvinte pe topic
            for t, words in enumerate(top_words(comp, terms)):
                print(f"Topic {t}: {', '.join(words)}")

            # asignarea documentelor + verificare grupare
            assigned = assign_topics(doc_topic)
            print("\nDomeniu real vs topic asignat:")
            print(pd.crosstab(pd.Series(labels, name="domeniu"),
                              pd.Series(assigned, name="topic")))
            print(f"Purity: {purity(labels, assigned):.2f} | ARI: {ari(labels, assigned):.2f}")

            # grafice
            name = f"{method}_{rep}_k{K}".replace("-", "")
            plot_words(comp, terms, method, title, f"plots/{name}_words.png")
            plot_docs(doc_topic, labels, assigned, method, title, f"plots/{name}_docs.png")

    print("\nGraficele sunt salvate in folderul plots/")


if __name__ == "__main__":
    main()