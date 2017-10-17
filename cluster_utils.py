# cluster_utils.py

"""
Cluster related functions and classes

Available classes:
- TSCluster

Available functions:
- LB_Keogh(s1, s2, r): Calculate LB_Keogh lower bound to dynamic time warping.
- smooth(x, window_len, window): Smooth array x by convolving with filter (e.g. hanning function)
"""

from collections import defaultdict
import numpy as np

def LB_Keogh(s1, s2, r):
    """
    Calculate LB_Keogh lower bound to dynamic time warping.
    """
    LB_sum = 0
    for ind, i in enumerate(s1):

        lower_bound = min(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])
        upper_bound = max(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])

        if i > upper_bound:
            LB_sum = LB_sum + (i - upper_bound) ** 2
        elif i < lower_bound:
            LB_sum = LB_sum + (i - lower_bound) ** 2

    return np.sqrt(LB_sum)


def smooth(x, window_len=48, window='hanning'):
    """
    Smooth array x by convolving with filter (e.g. hanning function)
    """
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."
    if window_len < 3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s = np.r_[2 * x[0] - x[window_len - 1::-1], x, 2 * x[-1] - x[-1:-window_len:-1]]

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='same')
    smoothed = y[window_len:-window_len + 1]

    return smoothed


class TSCluster(object):
    def __init__(self, num_clust):
        """
        Class to cluster time series
        
        Notes
        -----
        - num_clust is the number of clusters for the k-means algorithm
        - assignments holds the assignments of data points (indices) to clusters
        - centroids/medoids holds the centroids/medoids of the clusters
        - ts_dists holds the distance between a time series and its cluster's __oid 
        """
        self.num_clust = num_clust

    def k_medoids_clust(self, data, dist_matrix, num_iter):
        """
        Clustering using kmedoids on data ([num_series, length] matrix) using
        dist_matrix ([num_series, num_series]) matrix
        """

        # Turn dist_matrix from upper-triangular to full
        for i in range(len(data)):
            for j in range(0, i):
                dist_matrix[i][j] = dist_matrix[j][i]

        # Actual clustering
        M, C = self.kMedoids(dist_matrix, self.num_clust, num_iter)

        # Wrap up
        self.medoids = []
        for c_ts_idx in M:
            self.medoids.append(data[c_ts_idx])

        self.ts_dists = defaultdict(dict)
        self.assignments = defaultdict(list)
        for c in C:  # just 0, 1, 2, .... k-1
            c_ts_idx = M[c]
            for ts_idx in C[c]:
                self.assignments[c].append(data[ts_idx])
                self.ts_dists[c][ts_idx] = max(dist_matrix[c_ts_idx][ts_idx],
                                               dist_matrix[ts_idx][c_ts_idx])

        # Even though whole point of medoids is to avoid Euclidean mean-based centroids, I think it is still
        # nice to show the 'mean' of the curves of one cluster for kmedoids to produce smoother representations
        # of each curve. Plus, the pathological example doesn't apply if the cluster does indeed contain similar shapes.
        self.centroids = []
        for c in C:
            cur_centroid = np.zeros(data.shape[1])
            for ts_idx in C[c]:
                cur_centroid += data[ts_idx]
            cur_centroid /= len(C[c])
            self.centroids.append(cur_centroid)

    def kMedoids(self, D, k, tmax=100):
        """
        Perform k-medoids clustering on pair-wise distance matrix D.

        Returns
        -------
        M: array of medoid indices
        C: dictionary of cluster assignments, key is cluster index, value is array of data indices
        """
        m, n = D.shape

        if k > n:
            raise Exception('too many medoids')

        # Randomly initialize an array of k medoid indices
        M = np.arange(n)
        np.random.shuffle(M)
        M = np.sort(M[:k])

        # Create a copy of the array of medoid indices
        Mnew = np.copy(M)

        C = {}  # cluster assignments
        for t in range(tmax):
            # Determine clusters, i.e. arrays of data indices
            J = np.argmin(D[:, M], axis=1)
            for kappa in range(k):
                C[kappa] = np.where(J == kappa)[0]

            # Update cluster medoids
            for kappa in range(k):
                J = np.mean(D[np.ix_(C[kappa], C[kappa])], axis=1)
                j = np.argmin(J)
                Mnew[kappa] = C[kappa][j]
            np.sort(Mnew)

            # Check for convergence
            if np.array_equal(M, Mnew):
                break
            M = np.copy(Mnew)
        else:
            # Final update of cluster memberships
            J = np.argmin(D[:, M], axis=1)
            for kappa in range(k):
                C[kappa] = np.where(J == kappa)[0]

        return M, C