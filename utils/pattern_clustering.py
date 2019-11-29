
from sklearn.neighbors import KNeighborsClassifier
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import silhouette_score
from scipy import sparse
import matplotlib.pyplot as plt
import numpy as np

class PatternClustering():
    def __init__(self, affinity='complete', metric='jaccard'):
        self.affinity = affinity
        self.metric = metric
        self.threshold = None
        self.criterion = None
        
    def fit(self, X=None, distance_matrix=None, threshold=10, criterion='maxclust', n_jobs=-1):
        assert X is not None or distance_matrix is not None
        
        if distance_matrix is None:
            print('==> Computing Distance Matrix...')
            distance_matrix = pairwise_distances(X, metric=self.metric, n_jobs=n_jobs)
            #distance_matrix = squareform(pdist(X, self.metric))
            del X
            print('Done.')
        
        print('==> Computing Linakage Matrix...')
        self.linkage_matrix = linkage(distance_matrix, self.affinity)
        print('Done.')
        
        self.threshold = threshold
        self.criterion = criterion
        self.silhouette_avgs = self.compute_elbow(distance_matrix)
        del distance_matrix
    
    def get_clusters(self, threshold=None, criterion=None):
        if criterion is None:
            criterion = self.criterion
        if threshold is None:
            threshold = self.threshold
        return fcluster(self.linkage_matrix, threshold, criterion)
    
    def predict(self, model_samples, X, n_jobs=-1):
        assert self.affinity in ['average', 'single', 'complete']
        features, labels = model_samples
        distance_matrix = pairwise_distances(X, features, metric=self.metric, n_jobs=n_jobs)
        
        distance_to_clusters = []
        for cluster_id in range(1, max(labels) + 1):
            idx_list = [i for i, val in enumerate(labels) if val == cluster_id]
            if self.affinity == 'average':
                _ = np.mean(distance_matrix[:, idx_list], axis=1)
            elif self.affinity == 'single':
                _ = np.min(distance_matrix[:, idx_list], axis=1)
            elif self.affinity == 'complete':
                _ = np.max(distance_matrix[:, idx_list], axis=1)
            distance_to_clusters.append(_)
        distance_to_clusters = np.vstack(distance_to_clusters)
        return distance_to_clusters
        
        # closest_cluster = np.argmin(distance_to_clusters, axis=0)
        # return [idx+1 for idx in closest_cluster]

        # closest_sample = distance_matrix.argmin(axis=1)
        # return [labels[idx] for idx in closest_sample]
        
        # knn = KNeighborsClassifier(metric=self.metric, weights='distance', n_neighbors=1)            
        # knn.fit(model_samples.astype(bool), self.get_clusters())  
        # return knn.predict(X)
        
    def fancy_dendrogram(self, *args, **kwargs):
        max_d = kwargs.pop('max_d', None)
        if max_d and 'color_threshold' not in kwargs:
            kwargs['color_threshold'] = max_d
        annotate_above = kwargs.pop('annotate_above', 0)

        ddata = dendrogram(*args, **kwargs)

        if not kwargs.get('no_plot', False):
            plt.title('Hierarchical Clustering Dendrogram (truncated)')
            plt.xlabel('sample index or (cluster size)')
            plt.ylabel('distance')
            for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
                x = 0.5 * sum(i[1:3])
                y = d[1]
                if y > annotate_above:
                    plt.plot(x, y, 'o', c=c)
                    plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                                 textcoords='offset points',
                                 va='top', ha='center')
            if max_d:
                plt.axhline(y=max_d, c='k')
        return ddata
    
    '''
    @Param p: (int) only show the last <?> merges
    @Param d: (float) only annotates distance above <?>
    '''
    def show_dendrogram(self, p=30, d=150):
        plt.figure(figsize=(25, 8))
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('sample index')
        plt.ylabel('distance')
        self.fancy_dendrogram(
            self.linkage_matrix,
            leaf_rotation=90.,  # rotates the x axis labels
            leaf_font_size=15.,  # font size for the x axis labels
            show_contracted=True,
            truncate_mode='lastp',  # show only the last p merged clusters
            p=p,  # show only the last p merged clusters
            annotate_above=d,  # useful in small plots so annotations don't overlap
        )
        plt.show()
        
    def compute_elbow(self, distance_matrix, krange=[2, 50]):        
        silhouette_avgs = []
        ks = range(*krange)
        for k in ks:
            cluster_labels = self.get_clusters(threshold=k, criterion = 'maxclust')
            silhouette_avg = silhouette_score(distance_matrix, cluster_labels, metric="precomputed")
            silhouette_avgs.append(silhouette_avg)
            
        return silhouette_avgs
    
    def show_elbow(self, krange=[2, 50], silhouette_avgs=None, title='Cluster Number Selection'):
        if silhouette_avgs is None:
            silhouette_avgs = self.silhouette_avgs
        ks = range(*krange)
        plt.figure(figsize=(25, 8))
        plt.title(title)
        plt.xlabel('Number of clusters (K)')
        plt.ylabel('Silhouette Coefficient')
        plt.plot(ks, silhouette_avgs)
        plt.scatter(ks, silhouette_avgs)
        plt.show()