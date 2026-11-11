import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gensim.downloader as api
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

# ==========================================
# 1. ÎNCĂRCAREA MODELELOR ȘI PREGĂTIREA DATELOR
# ==========================================
print("Se încarcă modelele GloVe și Word2Vec...")
glove_model = api.load("glove-wiki-gigaword-50")  # Vectori de 50 dimensiuni
w2v_model = api.load("word2vec-google-news-300")  # Vectori de 300 dimensiuni

# Lista noastră de 20 de cuvinte împărțite pe categorii (animale, tehnologie, fructe, emoții)
words = [
    "king", "queen", "man", "woman",
    "dog", "cat", "animal", "pet",
    "computer", "software", "keyboard", "mouse",
    "apple", "banana", "fruit", "food",
    "happy", "sad", "angry", "joy"
]

# Filtrăm doar cuvintele care există în ambele modele
valid_words = [w for w in words if w in glove_model and w in w2v_model]
print(f"Cuvinte folosite ({len(valid_words)}): {valid_words}")

# Extragem matricea de vectori pentru cuvintele alese
glove_vectors = np.array([glove_model[w] for w in valid_words])
w2v_vectors = np.array([w2v_model[w] for w in valid_words])

# ==========================================
# 2. METODA 1: 3 DIMENSIUNI ALEATORII
# ==========================================
random.seed(42)  # Setăm un seed pentru reproductibilitate (să primim aceleași dimensiuni mereu)

# GloVe are 50 dimensiuni (alegem 3 random între 0 și 49)
glove_rand_dims = random.sample(range(50), 3)
# Word2Vec are 300 dimensiuni (alegem 3 random între 0 și 299)
w2v_rand_dims = random.sample(range(300), 3)

# Extragem doar coloanele alese aleatoriu din matrice
glove_rand_coords = glove_vectors[:, glove_rand_dims]
w2v_rand_coords = w2v_vectors[:, w2v_rand_dims]

# ==========================================
# 3. METODA 2: PCA (Principal Component Analysis)
# ==========================================
# Reduce dimensionalitatea proiectând datele pe axele cu cea mai mare variație
pca_glove = PCA(n_components=3, random_state=42)
pca_w2v = PCA(n_components=3, random_state=42)

glove_pca_coords = pca_glove.fit_transform(glove_vectors)
w2v_pca_coords = pca_w2v.fit_transform(w2v_vectors)

# ==========================================
# 4. METODA 3: t-SNE (t-Distributed Stochastic Neighbor Embedding)
# ==========================================
# Reduce dimensionalitatea păstrând distanțele locale (foarte bun pentru clustere/grupuri)
# Notă: 'perplexity' trebuie să fie mai mic decât numărul de cuvinte (20)
tsne_glove = TSNE(n_components=3, perplexity=5, random_state=42)
tsne_w2v = TSNE(n_components=3, perplexity=5, random_state=42)

glove_tsne_coords = tsne_glove.fit_transform(glove_vectors)
w2v_tsne_coords = tsne_w2v.fit_transform(w2v_vectors)


# ==========================================
# 5. FUNCȚIE PENTRU SALVAREA GRAFICELOR 3D
# ==========================================
def save_3d_plot(coords, word_list, title, filename):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')

    xs = coords[:, 0]
    ys = coords[:, 1]
    zs = coords[:, 2]

    # Desenăm punctele
    ax.scatter(xs, ys, zs, c='purple', marker='o')

    # Scriem cuvântul lângă fiecare punct
    for i, word in enumerate(word_list):
        ax.text(xs[i], ys[i], zs[i], word, fontsize=9)

    ax.set_title(title)
    plt.savefig(filename)
    plt.close()
    print(f"✔ S-a salvat graficul: {filename}")


# Generăm și salvăm toate cele 6 fișiere imagini
save_3d_plot(glove_rand_coords, valid_words, "GloVe - Dimensiuni Aleatorii", "glove_random.png")
save_3d_plot(w2v_rand_coords, valid_words, "Word2Vec - Dimensiuni Aleatorii", "w2v_random.png")

save_3d_plot(glove_pca_coords, valid_words, "GloVe - PCA (Reducere 3D)", "glove_pca.png")
save_3d_plot(w2v_pca_coords, valid_words, "Word2Vec - PCA (Reducere 3D)", "w2v_pca.png")

save_3d_plot(glove_tsne_coords, valid_words, "GloVe - t-SNE (Reducere 3D)", "glove_tsne.png")
save_3d_plot(w2v_tsne_coords, valid_words, "Word2Vec - t-SNE (Reducere 3D)", "w2v_tsne.png")

print("\nFelicitări! Toate vizualizările din Partea I au fost generate cu succes!")