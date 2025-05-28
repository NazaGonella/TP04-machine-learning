import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
from scipy.spatial import cKDTree


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
        plt.scatter(self.X[:, 0], self.X[:, 1], c=k, cmap='tab10', s=10, alpha=1)
        plt.scatter(self.mu[:, 0], self.mu[:, 1], c='red', marker='x', s=50)
        plt.title(f'K-Means (k = {self.k})')
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
        best_mu, best_cov, best_coef = None, None, None

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
        plt.show()

class DBScan():
    def __init__(self, X : np.ndarray):
        self.X : np.ndarray = X
    
    def get_closest_points(self, x: np.ndarray, epsilon: float, min_points: int) -> np.ndarray:
        tree : cKDTree = cKDTree(self.X)
        indices : list[int] = tree.query_ball_point(x, r=epsilon)
        if len(indices) == 0:
            return np.array([], dtype=int)
        candidates : np.ndarray = self.X[indices]
        distances : np.ndarray = np.linalg.norm(candidates - x, axis=1)
        sorted_idx : np.ndarray = np.argsort(distances)
        closest_indices : np.ndarray = np.array(indices, dtype=int)[sorted_idx[:min_points]]
        return closest_indices

    def fit(
        self, 
        epsilon : float ,
        min_points : int,
        max_iterations : int = 1000,
        runs : int = 1,
        print_iterations : bool = False,
    ) -> None:
        # LABELS
        #  0 : undefined
        # -1 : noise
        # >0 : clusters 1
        labels : dict[np.ndarray] = { x : 0 for x in self.X}
        c : int = 0
        for x in self.X:
            if labels[x] != 0:
                neighbors_index : list[int] = self.get_closest_points(x, epsilon)
                if len(neighbors_index) < min_points:
                    labels[x] = -1
                    continue


# project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))

# clustering_df : pd.DataFrame = pd.read_csv(f"{project_root}/TP04/data/raw/clustering.csv").drop(columns=['index'])
# GMM(clustering_df, 10)