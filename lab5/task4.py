# Task 4 - Analiza comparativa: rezumat al rezultatelor + raspunsuri la intrebari
import pandas as pd
from task2 import load_data
from task3 import evaluate_all

pd.set_option("display.width", 150)


def main():
    df = evaluate_all(load_data(), ks=[3, 5])
    k3, k5 = df[df.k == 3], df[df.k == 5]

    print("=" * 60 + "\nTABEL REZUMAT\n" + "=" * 60)
    print(df.round(3).to_string(index=False))

    # ---------------------------------------------------------------
    print("\n" + "=" * 60 + "\n1. BoW vs TF-IDF\n" + "=" * 60)
    print("Cv mediu (k=3 si k=5) pe metoda si reprezentare:")
    print(df.pivot_table(index="metoda", columns="reprezentare", values="c_v").round(3))
    print("Purity medie:")
    print(df.pivot_table(index="metoda", columns="reprezentare", values="purity").round(3))
    print("""
Interpretare (verifica cu cifrele de mai sus si cu topicurile din task 3):
- BoW foloseste frecvente brute: documentele lungi (ex. 'Music', 'Education') domina,
  iar cuvintele foarte frecvente (also, one, player, school) apar des in topicuri.
- TF-IDF reduce greutatea termenilor prezenti in multe documente si le creste pe cele
  specifice, deci topicurile tind sa fie mai distincte. Efectul e vizibil mai ales la
  LSA (prima componenta pe BoW e doar 'frecventa generala') si la NMF.
- LDA este un model generativ pentru numere (counts); pe TF-IDF (valori fractionare)
  functioneaza tehnic, dar nu mai respecta ipoteza modelului, deci BoW e alegerea naturala.""")

    # ---------------------------------------------------------------
    print("\n" + "=" * 60 + "\n2. Cea mai interpretabila metoda\n" + "=" * 60)
    print("Medie pe metoda (k=3):")
    print(k3.groupby("metoda")[["c_v", "diversity", "purity", "ARI"]].mean().round(3))
    print(f"\nCel mai mare Cv mediu (k=3): {k3.groupby('metoda').c_v.mean().idxmax()}")
    print(f"Cea mai mare purity medie (k=3): {k3.groupby('metoda').purity.mean().idxmax()}")
    print("""
Interpretare:
- LSA: componentele au semne mixte (+/-), deci sunt greu de interpretat ca 'topicuri'.
- NMF: toate valorile sunt non-negative, topicurile sunt aditive, de obicei foarte clare.
- LDA: topicuri ca distributii de probabilitate, interpretabile, dar pe corpus mic (30 de
  documente) pot iesi mai zgomotoase / mai putin stabile.""")

    # ---------------------------------------------------------------
    print("\n" + "=" * 60 + "\n3. Cv mare = topicuri mai bune?\n" + "=" * 60)
    print(f"Corelatia Cv vs purity: {df.c_v.corr(df.purity):.2f}")
    print(f"Corelatia Cv vs diversity: {df.c_v.corr(df.diversity):.2f}")
    best_cv = df.loc[df.c_v.idxmax()]
    best_pur = df.loc[df.purity.idxmax()]
    print(f"Cel mai mare Cv: {best_cv.metoda} / {best_cv.reprezentare} / k={int(best_cv.k)} "
          f"(Cv={best_cv.c_v:.3f}, purity={best_cv.purity:.2f})")
    print(f"Cea mai mare purity: {best_pur.metoda} / {best_pur.reprezentare} / k={int(best_pur.k)} "
          f"(Cv={best_pur.c_v:.3f}, purity={best_pur.purity:.2f})")
    print("""
Interpretare:
- Nu neaparat. Cv masoara doar cat de des apar impreuna top cuvintele unui topic.
  Un topic poate avea Cv mare fiindca repeta cuvinte generice sau foarte apropiate
  (redundant), sau poate fi coerent dar inutil (ex. un topic doar despre un jucator).
- Cv nu masoara diversitatea intre topicuri si nici potrivirea cu domeniile reale.
  Scorul trebuie combinat cu citirea topicurilor, diversity si purity.""")

    # ---------------------------------------------------------------
    print("\n" + "=" * 60 + "\n4. Efectul cresterii numarului de topicuri\n" + "=" * 60)
    cmp = pd.DataFrame({
        "Cv k=3": k3.groupby("metoda").c_v.mean(),
        "Cv k=5": k5.groupby("metoda").c_v.mean(),
        "diversity k=3": k3.groupby("metoda").diversity.mean(),
        "diversity k=5": k5.groupby("metoda").diversity.mean(),
    })
    print(cmp.round(3))
    d_cv = k5.c_v.mean() - k3.c_v.mean()
    d_div = k5.diversity.mean() - k3.diversity.mean()
    print(f"\nSchimbare medie Cv (5 vs 3): {d_cv:+.3f}")
    print(f"Schimbare medie diversity (5 vs 3): {d_div:+.3f}")
    print("""
Interpretare:
- Cu mai multe topicuri, un domeniu poate fi spart in subteme (ex. football: jucatori vs
  competitii vs reguli; music: genuri) - topicurile devin mai specifice.
- Dar, cum sunt doar 3 domenii reale, apar topicuri apropiate intre ele: diversity scade
  (mai multe cuvinte repetate intre topicuri) = redundanta mai mare.
- Cv poate creste sau scadea; nu spune singur care k e potrivit.""")

    # ---------------------------------------------------------------
    print("\n" + "=" * 60 + "\n5. Corespondenta cu cele 3 domenii (k=3)\n" + "=" * 60)
    print(k3[["metoda", "reprezentare", "purity", "ARI"]].round(3).to_string(index=False))
    print("""
Interpretare:
- Purity/ARI apropiate de 1 inseamna ca topicurile corespund domeniilor music/football/school.
- Diferentele apar la articole de granita: 'Education' si 'Teacher' sunt mai generale,
  'Music' si 'Opera' pot atrage vocabular de educatie muzicala, iar articolele despre
  Messi/Ronaldo contin multe nume proprii (cluburi, ani, competitii) care pot forma
  un topic separat. Vezi si tabelele 'domeniu real vs topic' din task 2.""")

    df.to_csv("final_results.csv", index=False)
    print("\nSalvat in final_results.csv")


if __name__ == "__main__":
    main()