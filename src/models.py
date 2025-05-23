import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class KMeans():
    def __init__(self, X : np.ndarray, k : int):
        self.k : int = k
        self.X : np.ndarray = X
        self.mu : np.ndarray = self.initialize_centroids()
    
    def initialize_centroids(self) -> np.ndarray:
        mu : np.ndarray = np.array([self.X[np.random.randint(len(self.X))] for _ in range(self.k)])
        return mu
    
    def calculate_distances(self) -> np.ndarray:
        # forma vectorial de calcular la distancia de cada muestra al centroide
        return np.linalg.norm(self.mu[:, np.newaxis] - self.X, axis=2)
    
    def calculate_distance_squared_error(self) -> float:
        k : np.ndarray = np.argmin(self.calculate_distances(), axis=0)
        loss : float = np.sum(np.linalg.norm(self.X - self.mu[k], axis=1) ** 2)
        return loss
    
    def fit_centroids(self, max_iterations : int = 250, runs : int = 1) -> np.ndarray:
        best_loss = np.inf
        best_mu = None
        for t in range(runs):
            self.mu = self.initialize_centroids()
            r : np.ndarray = np.zeros(shape=(self.X.shape[0], self.k))
            for i in range(max_iterations):
                r_old : np.ndarray = r.copy()
                # for n in range(self.X.shape[0]):
                #     k : int = np.argmin((np.linalg.norm(mu - self.X[n], axis=1)))
                #     r[n] = np.zeros_like(range(self.k))
                #     r[n][k] = 1
                k = np.argmin(self.calculate_distances(), axis=0)
                r = np.zeros_like(r)
                r[np.arange(self.X.shape[0]), k] = 1

                # recalculo centroides
                number_of_points_per_cluster = np.sum(r, axis=0)
                for k in range(self.k):
                    if number_of_points_per_cluster[k] == 0:
                        continue
                    self.mu[k] = np.sum(self.X[r[:, k] == 1], axis=0) / number_of_points_per_cluster[k]
                # veo si converge
                if np.array_equal(r, r_old):
                    print("done at iteration ", i)
                    break
            loss : float = self.calculate_distance_squared_error()
            if loss < best_loss:
                best_loss = loss
                best_mu = self.mu.copy()
        self.mu = best_mu
        return self.mu
    
    def plot_clusters_2d(self):
        k = np.argmin(self.calculate_distances(), axis=0)
        plt.scatter(self.X[:, 0], self.X[:, 1], c=k, cmap='tab10', s=10, alpha=1)
        plt.scatter(self.mu[:, 0], self.mu[:, 1], c='red', marker='x', s=50)
        plt.title(f'K-Means (k = {self.k})')
        plt.xlabel('A')
        plt.ylabel('B')



# project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
# np.random.seed(42)

# clustering_df : pd.DataFrame = pd.read_csv(f"{project_root}/TP04/data/raw/clustering.csv").drop(columns=['index'])
# model : k_means = k_means(clustering_df.to_numpy(), 4)
# model.fit_centroids()