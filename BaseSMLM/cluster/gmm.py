import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from scipy.spatial import distance
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

class VBGMM:
    """Fit a Bayesian GMM with for a fixed number of clusters with variable concentration parameter"""
    def __init__(self,X):
        self.X = X

    def plot_ellipses(self, ax, weights, means, covars):
        for n in range(means.shape[0]):
            eig_vals, eig_vecs = np.linalg.eigh(covars[n])
            unit_eig_vec = eig_vecs[0] / np.linalg.norm(eig_vecs[0])
            angle = np.arctan2(unit_eig_vec[1], unit_eig_vec[0])
            angle = 180 * angle / np.pi
            eig_vals = 2 * np.sqrt(2) * np.sqrt(eig_vals)
            ell = mpl.patches.Ellipse(
                means[n], eig_vals[0], eig_vals[1], angle=180 + angle, edgecolor="black"
            )
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(weights[n])
            ell.set_facecolor("#56B4E9")
            ax.add_artist(ell)


    def plot_results(self,X,estimator):
        fig,(ax1,ax2) = plt.subplots(1,2,figsize=(8,3))
        ax1.scatter(X[:, 0], X[:, 1], s=2, marker="x", color='cornflowerblue', alpha=0.8)
        ax1.set_xticks(())
        ax1.set_yticks(())
        self.plot_ellipses(ax1, estimator.weights_, estimator.means_, estimator.covariances_)

        ax2.get_xaxis().set_tick_params(direction="out")
        ax2.yaxis.grid(True, alpha=0.7)
        for k, w in enumerate(estimator.weights_):
            ax2.bar(
                k,
                w,
                width=0.9,
                color="#56B4E9",
                zorder=3,
                align="center",
                edgecolor="black",
            )
        ax2.set_ylim(0.0, 1.1)
        ax2.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
        ax2.tick_params(axis="x", which="both", top=False)

    def fit(self,concentration=1.0,plot=False):
        random_state, n_components, n_features = 2, 12, 2
        estimator = BayesianGaussianMixture(
                    weight_concentration_prior_type="dirichlet_process",
                    n_components=2 * n_components,
                    reg_covar=0,
                    init_params="random",
                    max_iter=1500,
                    mean_precision_prior=0.8,
                    random_state=random_state,
                )
        estimator.weight_concentration_prior = concentration
        estimator.fit(self.X)
        if plot:
            self.plot_results(self.X,estimator)
            plt.show()

class VBGMM_Test:
    """Fit a Bayesian GMM with for a fixed number of clusters 
       with variable concentration parameter"""
    def __init__(self,X):
        self.X = X

    def plot_ellipses(self, ax, weights, means, covars):
        for n in range(means.shape[0]):
            eig_vals, eig_vecs = np.linalg.eigh(covars[n])
            unit_eig_vec = eig_vecs[0] / np.linalg.norm(eig_vecs[0])
            angle = np.arctan2(unit_eig_vec[1], unit_eig_vec[0])
            angle = 180 * angle / np.pi
            eig_vals = 2 * np.sqrt(2) * np.sqrt(eig_vals)
            ell = mpl.patches.Ellipse(
                means[n], eig_vals[0], eig_vals[1], angle=180 + angle, edgecolor="black"
            )
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(weights[n])
            ell.set_facecolor("#56B4E9")
            ax.add_artist(ell)


    def plot_results(self, ax1, ax2, estimator, X, title, plot_title=False):
        ax1.set_title(title)
        ax1.scatter(X[:, 0], X[:, 1], s=2, marker="x", color='cornflowerblue', alpha=0.8)
        ax1.set_xticks(())
        ax1.set_yticks(())
        self.plot_ellipses(ax1, estimator.weights_, estimator.means_, estimator.covariances_)

        ax2.get_xaxis().set_tick_params(direction="out")
        ax2.yaxis.grid(True, alpha=0.7)
        for k, w in enumerate(estimator.weights_):
            ax2.bar(
                k,
                w,
                width=0.9,
                color="#56B4E9",
                zorder=3,
                align="center",
                edgecolor="black",
            )
        ax2.set_ylim(0.0, 1.1)
        ax2.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
        ax2.tick_params(axis="x", which="both", top=False)

        if plot_title:
            ax1.set_ylabel("Mixtures")
            ax2.set_ylabel("Weights")

    def fit(self):
        random_state, n_components, n_features = 2, 12, 2
        estimators = [
            (
                r"$\gamma_0=$",
                BayesianGaussianMixture(
                    weight_concentration_prior_type="dirichlet_distribution",
                    n_components=2 * n_components,
                    reg_covar=0,
                    init_params="random",
                    max_iter=1500,
                    mean_precision_prior=0.8,
                    random_state=random_state,
                ),
                [0.001, 1, 1000],
            ),
            (
                r"$\gamma_0=$",
                BayesianGaussianMixture(
                    weight_concentration_prior_type="dirichlet_process",
                    n_components=2 * n_components,
                    reg_covar=0,
                    init_params="random",
                    max_iter=1500,
                    mean_precision_prior=0.8,
                    random_state=random_state,
                ),
                [1, 1000, 100000],
            ),
        ]

        for title, estimator, concentrations_prior in estimators:
            plt.figure(figsize=(4.7 * 3, 8))
            plt.subplots_adjust(
                bottom=0.04, top=0.90, hspace=0.05, wspace=0.05, left=0.03, right=0.99
            )

            gs = gridspec.GridSpec(3, len(concentrations_prior))
            for k, concentration in enumerate(concentrations_prior):
                estimator.weight_concentration_prior = concentration
                estimator.fit(self.X)
                self.plot_results(
                    plt.subplot(gs[0:2, k]),
                    plt.subplot(gs[2, k]),
                    estimator,
                    self.X,
                    r"%s$%.1e$" % (title, concentration),
                    plot_title=k == 0,
                )
            plt.show()

class GMM_Ensemble:
    """Fit regular GMM with EM using r-T grid search (each r-T pair will fix the number of clusters)"""
    def __init__(self,X):
        self.X = X

    def plot_ellipses(self, ax, weights, means, covars):
        for n in range(means.shape[0]):
            eig_vals, eig_vecs = np.linalg.eigh(covars[n])
            unit_eig_vec = eig_vecs[0] / np.linalg.norm(eig_vecs[0])
            angle = np.arctan2(unit_eig_vec[1], unit_eig_vec[0])
            angle = 180 * angle / np.pi
            eig_vals = 2 * np.sqrt(2) * np.sqrt(eig_vals)
            ell = mpl.patches.Ellipse(
                means[n], eig_vals[0], eig_vals[1], angle=180 + angle, edgecolor="black"
            )
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(weights[n])
            ell.set_facecolor("#56B4E9")
            ax.add_artist(ell)


    def plot_results(self,X,ax1,ax2,estimator):
        ax1.scatter(X[:, 0], X[:, 1], s=2, marker="x", color='cornflowerblue', alpha=0.8)
        ax1.set_xticks(())
        ax1.set_yticks(())
        self.plot_ellipses(ax1, estimator.weights_, estimator.means_, estimator.covariances_)

        ax2.get_xaxis().set_tick_params(direction="out")
        ax2.yaxis.grid(True, alpha=0.7)
        for k, w in enumerate(estimator.weights_):
            ax2.bar(
                k,
                w,
                width=0.9,
                color="#56B4E9",
                zorder=3,
                align="center",
                edgecolor="black",
            )
        ax2.set_ylim(0.0, 1.1)
        ax2.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
        ax2.tick_params(axis="x", which="both", top=False)


    def show_components(self,ax,idx,idxx,means,labels):
        ax.scatter(self.X[idxx,0], self.X[idxx,1], c=labels, cmap='rainbow', s=1,marker='x')
        ax.scatter(self.X[idx,0], self.X[idx,1], color='black', s=1,marker='x')
        ax.scatter(means[:,0], means[:,1], color='red', s=10,marker='o')
        plt.show()
        
    def show_fit(self,X,estimator,idx,idxx,means,labels):
        fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(9,3))
        self.plot_results(X,ax1,ax2,estimator)
        self.show_components(ax3,idx,idxx,means,labels)
        plt.tight_layout()

    def fit(self, xlim, ylim, rseq, thseq, plot=False):
        score_matrix = np.zeros((len(rseq),len(thseq)))
        N = self.X.shape[0]
        D = distance.cdist(self.X,self.X)
        D = D[:N, :N]
        for i,r in enumerate(rseq):
            K = np.sum(D <= r, axis=1) - 1
            L = np.sqrt((np.diff(xlim) * np.diff(ylim)) * K / (np.pi * (N - 1)))
            for j,th in enumerate(thseq):
                print(f'Fitting GMM: r={r}, T={th}')
                idx = np.argwhere(L < th)
                idxx = np.argwhere(L >= th)
                if len(idx) > 0:
                    A = D < 2*r
                    A = np.delete(A,idx,axis=0); A = np.delete(A,idx,axis=1)
                    np.fill_diagonal(A,0); csr = csr_matrix(A)
                    n_components, labels = connected_components(csr,directed=False)
                    means = np.empty((n_components, 2))
                    coordinates = self.X[idxx]
                    for component in range(n_components):
                        component_indices = np.where(labels == component)[0]
                        component_coordinates = coordinates[component_indices]
                        means[component] = np.mean(component_coordinates, axis=0)
                    estimator = GaussianMixture(
                                n_components=n_components,
                                means_init=means)
                    Xfilt = np.squeeze(self.X[idxx])
                    estimator.fit(Xfilt)
                    if plot: plt.close(); self.show_fit(Xfilt,estimator,idx,idxx,means,labels)
                    score_matrix[i,j] = estimator.score(self.X)
                    
        return score_matrix
                    
        
