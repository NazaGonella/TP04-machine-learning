import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class k_means():
    def __init__(self, X : np.ndarray, k : int):
        self.k = k
        self.X = X
    
    def initialize_centroids(self) -> np.ndarray:
        mu : np.ndarray = np.array([self.X[np.random.randint(len(self.X))] for _ in range(self.k)])
        return mu
    
    def calculate_distances(self, mu : np.ndarray) -> np.ndarray:
        # forma vectorial de calcular la distancia de cada muestra al centroide
        return np.linalg.norm(mu[:, np.newaxis] - self.X, axis=2)
    
    def fit_centroids(self, max_iterations : int = 250) -> np.ndarray:
        mu : np.ndarray = self.initialize_centroids()
        r : np.ndarray = np.zeros(shape=(self.X.shape[0], self.k))
        for i in range(max_iterations):
            r_old : np.ndarray = r.copy()
            # for n in range(self.X.shape[0]):
            #     k : int = np.argmin((np.linalg.norm(mu - self.X[n], axis=1)))
            #     r[n] = np.zeros_like(range(self.k))
            #     r[n][k] = 1
            distancias : np.ndarray = self.calculate_distances(mu)
            k = np.argmin(distancias, axis=0)
            r[:] = np.zeros_like(r)
            r[np.arange(len(k)), k] = 1

            # recalculo centroides
            number_of_points_per_cluster = np.sum(r, axis=0)
            for k in range(self.k):
                if number_of_points_per_cluster[k] == 0:
                    continue
                mu[k] = np.sum(self.X[r[:, k] == 1], axis=0) / number_of_points_per_cluster[k]
            # analizo si converge
            if np.array_equal(r, r_old):
                print("done at iteration ", i)
                break
        return


# project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
# np.random.seed(42)

# clustering_df : pd.DataFrame = pd.read_csv(f"{project_root}/TP04/data/raw/clustering.csv").drop(columns=['index'])
# model : k_means = k_means(clustering_df.to_numpy(), 4)
# model.fit_centroids()