import gensim.downloader as api

print("Se descarcă și se încarcă modelul GloVe (50 dimensiuni)...")
# glove-wiki-gigaword-50 este cea mai mică variantă pentru GloVe
glove_model = api.load("glove-wiki-gigaword-50")
print(f"GloVe încărcat cu succes! Dimensiune vectori: {glove_model.vector_size}, Cuvinte în vocabular: {len(glove_model):,}")

print("\nSe descarcă și se încarcă modelul Word2Vec (300 dimensiuni)...")
# word2vec-google-news-300 este modelul standard pre-antrenat disponibil
w2v_model = api.load("word2vec-google-news-300")
print(f"Word2Vec încărcat cu succes! Dimensiune vectori: {w2v_model.vector_size}, Cuvinte în vocabular: {len(w2v_model):,}")