import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from scipy.spatial import distance
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

class RTCluster:
    def __init__(self,X,r,T):
        self.X = X
        self.r = r
        self.T = T
        
    def show_components(self,idx,idxx,means,labels):
        fig,ax=plt.subplots()
        ax.scatter(self.X[idxx,0], self.X[idxx,1], c=labels, cmap='rainbow', s=1,marker='x')
        ax.scatter(self.X[idx,0], self.X[idx,1], color='black', s=1,marker='x')
        ax.scatter(means[:,0], means[:,1], color='red', s=10,marker='o')
        ax.set_aspect(1.0)
        plt.show()
        
    def radius_gyration(self,X):
        X = np.squeeze(X)
        mean = np.mean(X, axis=0)
        grad = np.mean(np.linalg.norm(X-mean,axis=1)**2)
        return grad

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
            ell.set_alpha(0.3)
            ell.set_facecolor("#56B4E9")
            ax.add_artist(ell)
            
    def plot_fit(self,X,model):
        fig,(ax1,ax2) = plt.subplots(1,2,figsize=(7,3))
        ax1.scatter(X[:,0],X[:,1], s=2, marker="x", color='red', alpha=0.8)
        ax1.set_xticks(()); ax1.set_yticks(())
        self.plot_ellipses(ax1, model.weights_, model.means_, model.covariances_)
        ax2.get_xaxis().set_tick_params(direction="out")
        ax2.yaxis.grid(True, alpha=0.7)
        for k, w in enumerate(model.weights_):
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
        plt.show()
        
    def cluster(self,showK=False,show_clusters=False,fit_model=False,show_fit=False,min_size=6):
        N = self.X.shape[0]
        D = distance.cdist(self.X,self.X)
        D = D[:N, :N]
        K = np.sum(D <= self.r, axis=1) - 1
        if showK:
            plt.hist(K,bins=50)
            plt.show()
        idx = np.argwhere(K < self.T)
        idxx = np.argwhere(K >= self.T) #points that have T neighbors in radius r
        if len(idx) > 0:
            A = D < 2*self.r
            A = np.delete(A,idx,axis=0); A = np.delete(A,idx,axis=1)
            np.fill_diagonal(A,0); csr = csr_matrix(A)
            n_components, labels = connected_components(csr,directed=False)
            means = np.empty((n_components, 2))
            coordinates = np.squeeze(self.X[idxx])
            for component in range(n_components):
                component_indices = np.where(labels == component)[0]
                component_coordinates = coordinates[component_indices]
                num_points,_ = component_coordinates.shape
                means[component] = np.mean(component_coordinates, axis=0)
                
            if show_clusters:
                self.show_components(idx,idxx,means,labels)
                
            if fit_model:
                component_means = []
                component_covariances = []
                component_weights = []
                loglikes = []
                for component in range(n_components):
                    print(f'Fitting VBGMM component {component}')
                    component_indices = np.where(labels == component)[0]
                    component_coordinates = coordinates[component_indices]
                    num_points, _ = component_coordinates.shape
                    
                    if num_points > min_size:
                        model = BayesianGaussianMixture(
                            weight_concentration_prior_type="dirichlet_process",
                            n_components=5,
                            reg_covar=0,
                            init_params="random",
                            max_iter=1500,
                            mean_precision_prior=0.8,
                            random_state=None,
                            weight_concentration_prior=1.0
                        )
                        model.fit(component_coordinates)
                        component_means.append(model.means_)
                        component_covariances.append(model.covariances_)
                        component_weights.append(model.weights_)
                        this_loglike = np.sum(model.score_samples(component_coordinates))
                        loglikes.append(this_loglike)
   
                loglikes = np.array(loglikes)
                combined_means = np.concatenate(component_means, axis=0)
                combined_covariances = np.concatenate(component_covariances, axis=0)
                combined_weights = np.array(component_weights).flatten()
                norm = np.sum(combined_weights)
                combined_weights /= norm
                combined_model = GaussianMixture() #dummy model for storing things
                combined_model.means_ = combined_means
                combined_model.covariances_ = combined_covariances
                combined_model.weights_ = combined_weights
                loglike = np.sum(loglikes)-np.log(norm) #for renormalize
                print(f'Num params: {2*len(combined_weights)}, Loglike: {loglike}')
                
                if show_fit:
                    self.plot_fit(coordinates,combined_model)    
                                      
                score = loglike
                
            else:
                score = None
        return score
                    
class RTGridSearch:
    def __init__(self,X):
        self.X = X
    def search(self, rseq, thseq, showK=False, show_clusters=False, fit_model=False,show_fit=False):
        score_matrix = np.zeros((len(rseq),len(thseq)))
        N = self.X.shape[0]
        D = distance.cdist(self.X,self.X)
        D = D[:N, :N]
        for i,r in enumerate(rseq):
            for j,th in enumerate(thseq):
                print(f'Grid Search: r={r}, T={th}')
                clust = RTCluster(self.X,r,th)
                score = clust.cluster(show_clusters=show_clusters,fit_model=fit_model,
                                      show_fit=show_fit,showK=showK)
                print(score)
                score_matrix[i,j] = score
        return score_matrix

        
