import matplotlib.pyplot as plt
import numpy as np 
import sklearn.cluster

def clustering(points, method='dbscan'):
    ''' Perform clustering to the points using the method given
    
    Parameters
    ----------
    
    `points` (`numpy.ndarray`): point cloud to be clustered

    `method` (`string`): method used. Options: `'dbscan'`, `'kmeans'`, `'optics'`, `'meanshift'`, `'AgglomerativeClustering'`, `'birch'`

    Returns
    -------
    `labels` (`numpy.ndarray`): labels of each point
    '''
    if method == 'dbscan':
        db = sklearn.cluster.DBSCAN(eps=2,min_samples=3).fit(points)#eps=1.8, min_samples=20
        labels_db = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels_db)) - (1 if -1 in labels_db else 0)
        n_noise_ = list(labels_db).count(-1)
        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)
        return labels_db

    if method == 'kmeans':
        kmeans = sklearn.cluster.KMeans(n_clusters=60, random_state=0, n_init="auto").fit(points)
        labels_km = kmeans.labels_
        labels_unique = np.unique(labels_km)
        n_clusters_ = len(labels_unique)
        print("number of estimated clusters : %d" % n_clusters_)
        return labels_km

    if method == 'meanshift':
        # The following bandwidth can be automatically detected using
        bandwidth = sklearn.cluster.estimate_bandwidth(points, quantile=0.1, n_samples=20)

        ms = sklearn.cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(points)
        labels_ms = ms.labels_
        cluster_centers = ms.cluster_centers_

        labels_unique = np.unique(labels_ms)
        n_clusters_ = len(labels_unique)

        print("number of estimated clusters : %d" % n_clusters_)
        return labels_ms

    if method == 'optics':
        clust = sklearn.cluster.OPTICS(min_samples=3)

        # Run the fit
        clust.fit(points)
        labels_op = clust.labels_[clust.ordering_]
        labels_unique = np.unique(labels_op)
        n_clusters_ = len(labels_unique)
        print("number of estimated clusters : %d" % n_clusters_)
        return labels_op

    if method == 'Agglomerative':
        clustering = sklearn.cluster.AgglomerativeClustering(70).fit(points)
        labels_unique = np.unique(clustering.labels_)
        n_clusters_ = len(labels_unique)
        print("number of estimated clusters : %d" % n_clusters_)
        return clustering.labels_

    if method == 'birch':
        brc = sklearn.cluster.Birch(n_clusters=70).fit(points)
        labels_brc = brc.predict(points)
        labels_unique = np.unique(labels_brc)
        n_clusters_ = len(labels_unique)
        print("number of estimated clusters : %d" % n_clusters_)
        return labels_brc