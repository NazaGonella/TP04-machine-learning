import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.cm import get_cmap
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
from scipy.spatial import KDTree


class KMeans():
    def __init__(self, X : np.ndarray, k : int):
        self.k : int = k
        self.X : np.ndarray = X
        self.mu : np.ndarray
    
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
    
    def fit_centroids(
            self,
            max_iterations : int = 250,
            runs : int = 1,
            print_iterations : bool = False,
        ) -> np.ndarray:
        best_loss : float = np.inf
        best_mu : np.ndarray = np.array([])
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
                    if print_iterations:
                        print("K-MEANS: done at iteration ", i)
                    break
            loss : float = self.calculate_distance_squared_error()
            if loss < best_loss:
                best_loss = loss
                best_mu = self.mu.copy()
        self.mu = best_mu
        return self.mu
    
    def plot_clusters_2d(self):
        k = np.argmin(self.calculate_distances(), axis=0)
        plt.scatter(self.X[:, 0], self.X[:, 1], c=k, cmap='tab20', s=10, alpha=1)
        plt.scatter(self.mu[:, 0], self.mu[:, 1], c='red', marker='x', s=50)
        # plt.title(f'K-Means (k = {self.k})')
        plt.xlabel('A')
        plt.ylabel('B')

class GMM():
    def __init__(self, X : np.ndarray, k : int):
        self.k : int = k
        self.X : np.ndarray = X
        self.mu : np.ndarray
        self.coef : np.ndarray
        self.cov : np.ndarray
    
    def initialize_centroids(self) -> np.ndarray:
        mu : np.ndarray = np.array([self.X[np.random.randint(len(self.X))] for _ in range(self.k)])
        return mu
    
    def initialize_global_covariance(self) -> np.ndarray:
        diff : np.ndarray = self.X - np.mean(self.X, axis=0)
        global_cov = np.dot(diff.T, diff) / self.X.shape[0]
        return np.array([global_cov for _ in range(self.k)])
    
    def log_likelihood(self):
        pdfs = np.array([
            multivariate_normal.pdf(self.X, mean=self.mu[k], cov=self.cov[k])
            for k in range(self.k)
        ])
        weighted_pdfs = self.coef[:, None] * pdfs
        likelihoods = np.sum(weighted_pdfs, axis=0)
        return np.sum(np.log(likelihoods + 1e-10))
    
    def fit_gaussians(
            self, 
            use_k_means_centroids : bool = False,
            max_iterations : int = 250,
            runs : int = 1,
            print_iterations : bool = False,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        best_log_likelihood = -np.inf
        best_mu : np.ndarray = np.array([])
        best_cov : np.ndarray = np.array([])
        best_mu : np.ndarray = np.array([])

        for run in range(runs):
            if use_k_means_centroids:
                k_means_model : KMeans = KMeans(self.X, self.k)
                k_means_model.fit_centroids(max_iterations=max_iterations, runs=runs)
                self.mu = k_means_model.mu
            else:
                self.mu = self.initialize_centroids()
            self.coef = np.array([1/self.k for _ in range(self.k)])
            self.cov = self.initialize_global_covariance()


            # EXPECTATION - MAXIMIZATION
            # falta runs
            r_nk : np.ndarray = np.zeros((len(self.X), self.k))
            r_nk_old : np.ndarray = np.array([])
            for i in range(max_iterations):
                # E step
                # for n in range(len(self.X)):
                #     for k in range(self.k):
                #         denominador = sum(
                #             self.coef[j] * multivariate_normal.pdf(self.X[n], self.mu[j], self.cov[j])
                #             for j in range(self.k)
                #         )
                #         r_nk[n][k] = self.coef[k] * multivariate_normal.pdf(x=self.X[n], mean=self.mu[k], cov=self.cov[k]) / denominador
                # Vectorized E-step
                pdfs = np.array([
                    multivariate_normal.pdf(self.X, mean=self.mu[k], cov=self.cov[k])
                    for k in range(self.k)
                ])  # shape (k, n_samples)

                weighted_pdfs = self.coef[:, None] * pdfs  # shape (k, n_samples)
                denominator = np.sum(weighted_pdfs, axis=0)  # shape (n_samples,)

                r_nk = (weighted_pdfs / denominator).T  # shape (n_samples, k)
                # M step
                N_K : float
                mu_new : np.ndarray = np.zeros_like(self.mu)
                cov_new : np.ndarray = np.zeros_like(self.cov)
                coef_new : np.ndarray = np.zeros_like(self.coef)
                for k in range(self.k):
                    N_k = np.sum(r_nk[:, k])
                    # mu_new[k] = (1 / N_k) * np.sum(r_nk[:, k] * self.X, axis=0) 
                    mu_new[k] = (1 / N_k) * np.sum(r_nk[:, k][:, None] * self.X, axis=0)
                    diff = self.X - mu_new[k]
                    # sigma_k = (1/ N_k) * np.sum(r_nk[:, k] *((self.X - mu_k) @ (self.X - mu_k).T))
                    diff = self.X - mu_new[k]
                    # cov_new[k] = (1 / N_k) * np.einsum('ni,nj->ij', r_nk[:, k][:, None] * diff, diff)
                    cov_new[k] = (1 / N_k) * np.einsum('ni,nj->ij', diff * r_nk[:, k][:, None], diff)
                    # diff = self.X - mu_k
                    # weighted_diff = r_nk[:, k][:, None] * diff
                    # sigma_k = np.dot(weighted_diff.T, diff) / N_k
                    coef_new[k] = N_k / self.X.shape[0]
                self.mu, self.cov, self.coef = mu_new, cov_new, coef_new

                # veo si converge
                if len(r_nk_old) > 0:
                    # if np.array_equal(r_nk, r_nk_old):
                    if np.allclose(r_nk, r_nk_old, atol=1e-4):
                        if print_iterations:
                            print("GMM: done at iteration", i)
                        break
                r_nk_old = r_nk.copy()

            ll = self.log_likelihood()
            if ll > best_log_likelihood:
                best_log_likelihood = ll
                best_mu, best_cov, best_coef = self.mu.copy(), self.cov.copy(), self.coef.copy()
        self.mu, self.cov, self.coef = best_mu, best_cov, best_coef
        return self.mu, self.cov, self.coef
    
    def plot_gmm(self):
        plt.scatter(self.X[:, 0], self.X[:, 1], s=5, c='gray', alpha=0.5)

        for k in range(len(self.mu)):
            plt.scatter(self.mu[k][0], self.mu[k][1], marker='x', color='red', s=100)
            eigenvalues, eigenvectors = np.linalg.eigh(self.cov[k])
            order = eigenvalues.argsort()[::-1]
            eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
            angle = np.degrees(np.arctan2(*eigenvectors[:,0][::-1]))
            width, height = 2 * np.sqrt(eigenvalues)
            ellipse = Ellipse(xy=self.mu[k], width=width, height=height, angle=angle, edgecolor='blue', fc='None', lw=2)
            plt.gca().add_patch(ellipse)
        plt.axis('equal')
        plt.xlabel('A')
        plt.ylabel('B')
        plt.show()

class DBScan():
    def __init__(self, X : np.ndarray):
        self.X : np.ndarray = X
        self.labels : dict[int, int]
    
    # def get_closest_points(self, x : np.ndarray, epsilon : float, min_points : int) -> set[tuple[float]]:
    #     tree : cKDTree = cKDTree(self.X)
    #     indices : list[int] = tree.query_ball_point(x, r=epsilon)
    #     if len(indices) == 0:
    #         return set()
    #     candidates : np.ndarray = self.X[indices]
    #     distances : np.ndarray = np.linalg.norm(candidates - x, axis=1)
    #     sorted_idx : np.ndarray = np.argsort(distances)
    #     closest_neighbors : np.ndarray = candidates[sorted_idx[:min_points]]
    #     return set(map(tuple, closest_neighbors))

    def range_query(self, x_i : int, epsilon : float) -> set[int]:
        tree : KDTree = KDTree(self.X)
        indices : list[int] = tree.query_ball_point(self.X[x_i], r=epsilon)
        if len(indices) == 0:
            set()
        return set(indices)


    def fit_labels(
        self, 
        epsilon : float ,
        min_points : int,
        max_iterations : int = 1000,
        runs : int = 1,
        print_iterations : bool = False,
    ) -> dict[int, int]:
        # LABELS
        #  0 : undefined
        # -1 : noise
        # >0 : clusters 1
        self.labels = { i : 0 for i in range(len(self.X)) }
        c : int = 0
        for x_i in range(len(self.X)):
            if self.labels[x_i] != 0:
                continue
            neighbors : set[int] = self.range_query(x_i, epsilon)
            if len(neighbors) < min_points:
                self.labels[x_i] = -1
                continue
            c += 1
            self.labels[x_i] = c
            s : set[int] = neighbors.copy()
            s.discard(x_i)
            # for q in s:
            while len(s) > 0:
                q = s.pop()
                if self.labels[q] == -1:
                    self.labels[q] = c
                if self.labels[q] != 0:
                    continue
                self.labels[q] = c
                new_neighbors = self.range_query(q, epsilon)
                if len(new_neighbors) >= min_points:
                    s.update(new_neighbors)
        return self.labels
    
    def plot_dbscan(self):
        unique_labels = set(self.labels.values())
        colors = get_cmap("tab20", len(unique_labels))

        for label in unique_labels:
            indices = [i for i, l in self.labels.items() if l == label]
            cluster_points = self.X[indices]
            if label == -1:
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=5, c='k', alpha=0.2, label='noise')
            else:
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=10, color=colors(label - 1))
        
        plt.axis('equal')
        plt.xlabel('A')
        plt.ylabel('B')
        plt.legend()
        plt.show()
    
    def plot_k_distance(self, min_points : int):
        tree : KDTree = KDTree(self.X)
        k_distances = []
        for point in self.X:
            distances, _ = tree.query(point, k=min_points + 1)
            k_distances.append(distances[-1])
        k_distances = np.sort(k_distances)
        plt.plot(k_distances)
        plt.grid(True)
        plt.xlabel("Puntos ordenados")
        plt.ylabel(f"{min_points}-ésima distancia")
        plt.axhspan(0.05, 0.10, color='red', alpha=0.3, label='Rango ε')
        plt.legend()
        plt.show()

class PCA:
    def __init__(self, X : np.ndarray):
        self.X : np.ndarray = X
        self.mean : float = np.mean(X, axis=0, keepdims=True)
        self.X_low_dim : np.ndarray
        self.X_reconstruction : np.ndarray
    
    # k: Dimensiones asociadas a los autovalores más bajos para quitar.
    def reduce_dimensionality(self, k : int) -> None:
        if k > self.X.shape[1]:
            print("Se quitan más componentes de las que hay")
            return
        X_centered : np.ndarray = self.X - self.mean
        X_centered_covariance : np.ndarray = (X_centered.T @ X_centered) / len(self.X)
        X_cov_eigvec : np.ndarray
        X_cov_eigval : np.ndarray
        X_cov_eigval, X_cov_eigvec = np.linalg.eigh(X_centered_covariance)
        greatest_k_indices : np.ndarray = np.argsort(X_cov_eigval)[k:]
        X_low_dim : np.ndarray = X_centered @ X_cov_eigvec[:, greatest_k_indices]
        self.X_low_dim = X_low_dim
        self.X_reconstruction = (X_low_dim @ X_cov_eigvec[:, greatest_k_indices].T) + self.mean
    
    def get_reconstruction_MSE(self) -> float:
        squared_errors : np.ndarray = (self.X - self.X_reconstruction) ** 2
        return np.mean(squared_errors, dtype=float)

# project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))

# clustering_df : pd.DataFrame = pd.read_csv(f"{project_root}/TP04/data/raw/clustering.csv").drop(columns=['index'])
# GMM(clustering_df, 10)