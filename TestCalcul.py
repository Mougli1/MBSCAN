# test_me_dissim_module.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from scipy.io import loadmat

from MBSCAN_Vpubliée.CalculMBD import MeDissimilarity  # Importer la classe optimisée

def main():
    # Définir les exemples de données manuellement
    exemples = np.array([
        [1, 2],
        [1, 2],  # Point dupliqué
        [1.0000001, 2.0000001],  # Point très proche
        [2, 3],
        [21, 3.2],  # Point dupliqué
    ])
    data_mat = loadmat('../data.mat')
    exemples = data_mat['data']

    print(f"Nombre d'exemples : {exemples.shape[0]}")
    print(f"Nombre de caractéristiques : {exemples.shape[1]}")

    # Initialiser MeDissimilarity avec les exemples
    diss = MeDissimilarity(exemples)
    num_arbres = 10 # Nombre d'arbres aléatoires
    fonction_diss = diss.get_dissim_func(num_arbres)

    num_exemples = exemples.shape[0]
    matrice_dissimilarite = np.zeros((num_exemples, num_exemples))

    # Définir une fonction pour calculer une ligne de la matrice en utilisant fonction_diss
    def calculer_ligne(i):
        return fonction_diss(exemples[i], exemples)

    print("Calcul de la matrice de dissimilarité avec parallélisation...")
    try:
        matrice_dissimilarite = np.array(
            Parallel(n_jobs=-1)(
                delayed(calculer_ligne)(i) for i in range(num_exemples)
            )
        )
    except Exception as e:
        print("Erreur lors du calcul de la matrice de dissimilarité :", e)
        return
    print("Calcul terminé.")

    # Visualiser la matrice de dissimilarité
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrice_dissimilarite, cmap='viridis', annot=True, fmt=".2f")
    plt.title("Matrice de Dissimilarité")
    plt.xlabel("Exemples")
    plt.ylabel("Exemples")
    plt.show()

    print("Matrice de Dissimilarité :")
    print(matrice_dissimilarite)

    # Sauvegarder la matrice de dissimilarité
    np.save('../matrice_dissimilarite.npy', matrice_dissimilarite)
    print("Matrice de dissimilarité sauvegardée dans 'matrice_dissimilarite.npy'.")


if __name__ == "__main__":
    main()
