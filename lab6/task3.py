import gensim.downloader as api
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

# 1. Încărcăm modelele
print("Se încarcă modelele GloVe și Word2Vec...")
glove_model = api.load("glove-wiki-gigaword-50")
w2v_model = api.load("word2vec-google-news-300")

# 2. Definim un set de ~25 de cuvinte împărțite pe categorii clare
# (Animale, Tehnologie, Fructe, Țări) pentru a vedea clusterele în dendrogramă
words = [
    # Animale
    "dog", "cat", "lion", "tiger", "horse",
    # Tehnologie
    "computer", "software", "keyboard", "mouse", "internet",
    # Fructe / Mâncare
    "apple", "banana", "orange", "fruit", "food",
    # Țări / Orașe
    "france", "germany", "london", "paris", "rome",
    # Emoții / Diverse
    "happy", "sad", "king", "queen", "car"
]

# Filtrăm doar cuvintele care există în ambele modele
valid_words = [w for w in words if w in glove_model and w in w2v_model]
print(f"Cuvinte folosite pentru clustering ({len(valid_words)}): {valid_words}")

# Extragem matricea de vectori
glove_vectors = np.array([glove_model[w] for w in valid_words])
w2v_vectors = np.array([w2v_model[w] for w in valid_words])


# 3. Funcție pentru generarea și salvarea dendrogramei
def plot_dendrogram(vectors, word_list, title, filename):
    # Calculăm matricea de legătură (linkage matrix) folosind distanța cosinus sau Euclideană
    # Pentru embedding-uri, 'ward' sau 'average' cu metric='cosine' funcționează foarte bine.
    # Aici folosim linkage cu metoda Ward pe date normalizate sau distanță euclidiană pe vectori.

    # O variantă simplă și robustă în SciPy pentru clustering ierarhic:
    linked = linkage(vectors, method='ward')

    plt.figure(figsize=(12, 7))
    dendrogram(
        linked,
        labels=word_list,
        orientation='top',
        leaf_rotation=90,
        leaf_font_size=11
    )
    plt.title(title, fontsize=14)
    plt.xlabel("Cuvinte")
    plt.ylabel("Distanța")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"✔ S-a salvat dendrograma: {filename}")


# Generăm dendrogramele pentru ambele modele
plot_dendrogram(glove_vectors, valid_words, "Dendrogramă Ierarhică - GloVe (50d)", "dendrogram_glove.png")
plot_dendrogram(w2v_vectors, valid_words, "Dendrogramă Ierarhică - Word2Vec (300d)", "dendrogram_w2v.png")

print("\nToate sarcinile au fost finalizate! Verifică imaginile generate (dendrogram_glove.png și dendrogram_w2v.png).")