import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.spatial import distance_matrix
from sklearn.mixture import GaussianMixture

def removeKNearest(coordinates,K):
    N = coordinates.shape[0]
    updated_coordinates = np.copy(coordinates)

    for k in range(K):
        dist_matrix = distance_matrix(updated_coordinates,updated_coordinates)
        np.fill_diagonal(dist_matrix, np.inf)
        row_idx,col_idx = np.where(dist_matrix == np.min(dist_matrix))[0]
        a = updated_coordinates[row_idx]
        b = updated_coordinates[col_idx]
        c = np.expand_dims(0.5 * (a + b), 0)
        updated_coordinates[row_idx] = 0
        updated_coordinates[col_idx] = 0
        updated_coordinates = np.append(updated_coordinates, c, axis=0)
        updated_coordinates = updated_coordinates[~np.all(updated_coordinates == 0, axis=1)]


    return updated_coordinates


class FixedCovMixture:
    """The model to estimate gaussian mixture with fixed covariance matrix
        in order to reduce duplication in SMLM experiments"""
        
    def __init__(self, n_components, var, max_iter=100, random_state=None, tol=1e-10):
        self.n_components = n_components
        self.var = var
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol=tol

    def fit(self,X):
        np.random.seed(self.random_state)
        n_obs, n_features = X.shape
        self.mean_ = removeKNearest(X,n_obs-self.n_components)
        self.init_means = np.copy(self.mean_)
        gmixture = GaussianMixture(n_components=self.n_components, covariance_type='tied')
        gmixture.means_ = self.init_means
        gmixture.covariances_ = self.var*np.eye(2)
        precision_matrix_cholesky = np.sqrt(1 / self.var) * np.eye(2)
        gmixture.precisions_cholesky_ = precision_matrix_cholesky
        gmixture.weights_ = np.ones((self.n_components,))/self.n_components
        log_like = gmixture.score_samples(X)
        return np.sum(log_like), gmixture

        
