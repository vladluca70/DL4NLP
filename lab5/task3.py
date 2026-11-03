# Task 3 - Coerenta topicurilor cu Gensim (k=3 si k=5)
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from task2 import load_data, fit_model, top_words, assign_topics, purity, ari, METHODS

KS = [3, 5]
METRICS = ["c_v", "u_mass", "c_uci", "c_npmi"]   # metricile de coerenta din Gensim


def diversity(topics):
    # cuvinte unice / cuvinte totale in top-N ale tuturor topicurilor (formula din curs)
    all_words = [w for t in topics for w in t]
    return len(set(all_words)) / len(all_words)


def coherence_scores(topics, texts, dictionary):
    scores = {}
    for m in METRICS:
        cm = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary,
                            coherence=m, processes=1)   # c_v foloseste fereastra de 110 cuvinte
        scores[m] = cm.get_coherence()
    return scores


def evaluate_all(data, ks=KS, verbose=False):
    labels = data["labels"]
    texts = [d.split() for d in data["docs"]]       # documentele preprocesate, tokenizate
    dictionary = Dictionary(texts)
    terms = np.array(data["bow_vec"].get_feature_names_out())
    matrices = {"BoW": data["X_bow"], "TF-IDF": data["X_tfidf"]}

    rows = []
    for k in ks:
        for rep, X in matrices.items():
            for method in METHODS:
                doc_topic, comp = fit_model(method, X, k)
                topics = top_words(comp, terms)     # top 10 cuvinte / topic
                assigned = assign_topics(doc_topic)
                row = {"k": k, "reprezentare": rep, "metoda": method}
                row.update(coherence_scores(topics, texts, dictionary))
                row["diversity"] = diversity(topics)
                row["purity"] = purity(labels, assigned)
                row["ARI"] = ari(labels, assigned)
                rows.append(row)
                if verbose:
                    print(f"\n[k={k}] {method} pe {rep}")
                    for t, words in enumerate(topics):
                        print(f"  Topic {t}: {', '.join(words)}")
                    print("  " + ", ".join(f"{m}={row[m]:.3f}" for m in METRICS))
    return pd.DataFrame(rows)


def main():
    data = load_data()
    # afisam topicurile ca sa le putem compara cu scorurile (interpretare calitativa)
    df = evaluate_all(data, verbose=True)

    pd.set_option("display.width", 150)
    print("\n" + "=" * 60 + "\nREZULTATE COERENTA\n" + "=" * 60)
    for k in KS:
        print(f"\nk = {k}")
        print(df[df.k == k].drop(columns="k").round(3).to_string(index=False))

    df.to_csv("coherence_results.csv", index=False)
    print("\nSalvat in coherence_results.csv")
    print("\nNota: c_v, c_uci, c_npmi: mai mare = mai bine; "
          "u_mass este negativ, mai aproape de 0 = mai bine.")


if __name__ == "__main__":
    main()