import gensim.downloader as api

# 1. Încărcăm modelele
print("Se încarcă modelele GloVe și Word2Vec...")
glove_model = api.load("glove-wiki-gigaword-50")
w2v_model = api.load("word2vec-google-news-300")

# 2. Definim perechile de cuvinte cerute de laborator:
# - 3 perechi de cuvinte înrudite (synonyms / same semantic field / antonyms)
# - 2 perechi de cuvinte neregulate / neînrudite (unrelated)
related_pairs = [
    ("king", "queen"),  # Același domeniu semantic (reg rede / regalitate)
    ("dog", "animal"),  # Relație hiperonim (câine - animal)
    ("happy", "joy")  # Sinonime / Emoții pozitive
]

unrelated_pairs = [
    ("computer", "banana"),  # Tehnologie vs Fruct (fără legătură)
    ("king", "keyboard")  # Regalitate vs Hardware (fără legătură)
]

print("\n" + "=" * 60)
print(f"{'Pereche de cuvinte':<20} | {'GloVe Similarity':<18} | {'Word2Vec Similarity':<18}")
print("=" * 60)

# 3. Calculăm pentru perechile înrudite
print("--- PERECHI ÎNRUDITE (Related Words) ---")
for w1, w2 in related_pairs:
    # Verificăm dacă ambele cuvinte există în ambele modele
    if w1 in glove_model and w2 in glove_model:
        sim_glove = glove_model.similarity(w1, w2)
    else:
        sim_glove = "N/A"

    if w1 in w2v_model and w2 in w2v_model:
        sim_w2v = w2v_model.similarity(w1, w2)
    else:
        sim_w2v = "N/A"

    print(f"{w1} - {w2}:".ljust(20) + f" | {str(sim_glove):<18} | {str(sim_w2v):<18}")

print("\n--- PERECHI NEÎNRUDITE (Unrelated Words) ---")
# 4. Calculăm pentru perechile neînrudite
for w1, w2 in unrelated_pairs:
    sim_glove = glove_model.similarity(w1, w2) if (w1 in glove_model and w2 in glove_model) else "N/A"
    sim_w2v = w2v_model.similarity(w1, w2) if (w1 in w2v_model and w2 in w2v_model) else "N/A"

    print(f"{w1} - {w2}:".ljust(20) + f" | {str(sim_glove):<18} | {str(sim_w2v):<18}")

print("=" * 60)