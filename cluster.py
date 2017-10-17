# cluster.py

"""
Cluster
Four main steps:
1) Preprocess / prepare data
2) Compute DTW distance matrix on preprocessed data
3) Use distance matrix to cluster
4) Create elbow plot

Available classes:
- ClusterAnalsyis
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, wait
import cPickle as pickle
from functools import partial
import math
import matplotlib
import matplotlib.pylab as plt
plt.style.use('ggplot')
import numpy as np
import os
import sys

from utils import setup_logging
from cluster_utils import LB_Keogh, smooth, TSCluster

# Strings used for files
OUTPUTS_PATH = 'outputs/'
TS_STR = 'n{}-w{}-ds{}'
DIST_MATRIX_STR = 'n{}-w{}-ds{}-r{}'
KCLUST_STR = 'n{}-w{}-ds{}-r{}-k{}-it{}'

class ClusterAnalysis(object):
    def __init__(self):
        _, self.logger = setup_logging(save_path='cluster.log')

        # If on linux (i.e. Shannon), use this matplotlib mode (no screen mode)
        if sys.platform =='linux' or sys.platform == 'linux2':
            matplotlib.use('Agg')     # use this if on linux

    ####################################################################################################################
    # Utils
    ####################################################################################################################
    def make_fake_data(self, n=10, l=100):
        """
        Test by on synthetic data composed of sin and cos waves. n time series, each of length l
        """
        def gen_fake_data(n, l, fn='sin'):
            # Create fake time series data of dimension [n,l]
            # Noisy sin waves and cos waves
            ts_data = np.zeros([n, l])
            for i in range(n):
                y_vals = []
                for x in np.linspace(0, 2 * math.pi, l):
                    noise = np.random.normal(0,0.25)
                    y_val = eval('np.' + fn + '(x + noise)')
                    y_vals.append(y_val)
                ts_data[i, :] = np.array(y_vals)
            return ts_data

        self.logger.info('Testing with n={}, l={}'.format(n, l))
        # Make fake data
        ts_data_sin = gen_fake_data(n/2, l, fn='sin')
        ts_data_cos = gen_fake_data(n/2, l, fn='cos')
        ts_data = np.vstack([ts_data_sin, ts_data_cos])
        pickle.dump(ts_data, open('test_ts_data.pkl', 'wb'), protocol=2)

    ####################################################################################################################
    # 1) Preprocess data
    ####################################################################################################################
    def prepare_ts(self, ts_data_path, w=None, ds=None):
        """
        Create and save np array of [num_timeseries, len]

        Parameters
        ----------
        ts_data_path: str, path to pkl file with data matrix [num_timeseries, len]
        w: float or int, window size to use for smoothing
            - if float, w is ratio of window size to len
            - if int, w is window size
        ds: int, ratio at which to downsample time series
            - used to speed up clustering
            - e.g. 3 = sample every third point
        """

        # Get all series from videos with predictions
        self.logger.info('PREPARING TIME SERIES')

        self.logger.info('Loading data from: {}'.format(ts_data_path))
        ts_data = pickle.load(open(ts_data_path, 'rb'))

        # Smooth
        if w:
            self.logger.info('Smoothing each time series with w=={}'.format(w))
            if type(w) == float:
                window_len = int(w * ts_data.shape[1])
            elif type(w) == int:
                window_len = w
            for i in range(ts_data.shape[0]):
                ts_data[i,:] = smooth(ts_data[i,:], window_len=window_len)

        # Downsample
        if ds:
            ts_downsampled = np.zeros([ts_data.shape[0], ts_data.shape[1] / ds])
            self.logger.info('Downsampling at a ratio of {}'.format(ds))
            for i in range(ts_data.shape[0]):
                ts_downsampled[i,:] = ts_data[i,:][::ds]
            ts_data = ts_downsampled

        # Save stuff
        self.logger.info('Processed time series data has shape: {}'.format(ts_data.shape))
        # self._save_ts(ts_data, ts_data.shape[0], w, ds)

        # Later will divide by std of ts to normalize. Skip if it's 0 for some reason
        # if np.std(downsampled) == 0:
        #     continue

        return ts_data

    # Save and load
    def save_ts(self, ts, n, w, ds):
        self.logger.info('Saving processed time series data')
        params_str = TS_STR.format(n, w, ds)
        path = self._get_ts_path(params_str)
        with open(path, 'wb') as f:
            pickle.dump(ts, f, protocol=2)

    def load_ts(self, n, w, ds):
        self.logger.info('Trying to load time series data')
        params_str = TS_STR.format(n, w, ds)
        path = self._get_ts_path(params_str)
        ts = pickle.load(open(path, 'rb'))
        return ts

    def _get_ts_path(self, params_str):
        path = os.path.join(OUTPUTS_PATH, 'data', 'ts_{}.pkl'.format(params_str))
        return path

    ####################################################################################################################
    # 2) Create DTW distance matrix used for clustering
    ####################################################################################################################
    def compute_dtw_dist_matrix(self, ts_data, r):
        """
        Compute pair-wise distance matrix on time series data 
        
        Params
        ------
        data: np array of dimension (num_timeseries, max_leN)
        r: int, window size for LB_Keogh
        """
        self.logger.info('COMPUTING DTW DISTANCE MATRIX')

        # Create DTW distance matrix in parallel
        dist_matrix = np.zeros([len(ts_data), len(ts_data)])

        def callback(i, j, future):
            dist = future.result()
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist
            # print 'Distance matrix entries not zero: {}'.format((dist_matrix != 0).sum())

        indices = [(i,j) for i in range(len(ts_data)) for j in range(i+1, len(ts_data))] # only calculate upper triangle
        with ProcessPoolExecutor(max_workers=4) as executer:
            fs = []
            for i, j in indices:
                future = executer.submit(LB_Keogh, ts_data[i], ts_data[j], r)
                future.add_done_callback(partial(callback, i, j))
                fs.append(future)
            wait(fs)

        return dist_matrix

    # Save and load
    def save_dtw_dist_matrix(self, dtw_dist_matrix, n, w, ds, r):
        self.logger.info('Saving DTW-based distance matrix')
        params_str = DIST_MATRIX_STR.format(n, w, ds, r)
        path = self._get_dtw_dist_matrix_path(params_str)
        with open(path, 'wb') as f:
            pickle.dump(dtw_dist_matrix, f, protocol=2)

    def load_dtw_dist_matrix(self, n, w, ds, r):
        self.logger.info('Trying to load DTW-based distance matrix')
        params_str = DIST_MATRIX_STR.format(n, w, ds, r)
        path = self._get_dtw_dist_matrix_path(params_str)
        dist_matrix = pickle.load(open(path, 'rb'))
        return dist_matrix

    def _get_dtw_dist_matrix_path(self, params_str):
        path = os.path.join(OUTPUTS_PATH, 'data', 'dtw-dist-matrix_{}.pkl'.format(params_str))
        return path

    ####################################################################################################################
    # 3) Clustering
    ####################################################################################################################
    def cluster_ts(self, data, dist_matrix, method, k=None, it=None):
        """
        Cluster data and save outputs

        Parameters
        ----------
        data: np of dimension [num_timeseries, len]
        dist_matrix: np of dimension [num_timeseries, len]
        method: str, clustering method to use
        k: int (for parameteric clustering methods)
        it: number of iterations
        """
        self.logger.info('CLUSTERING TIME SERIES')
        if method == 'kmedoids':
            clusterer = self._cluster_ts_kmedoids(data, dist_matrix, k, it)
            return clusterer
        else:
            self.logger.info('Method unknown: {}'.format(method))

        self.logger.info('Done clustering')

    def _cluster_ts_kmedoids(self, data, dist_matrix, k, it):
        """
        Cluster using kmedoids
        """
        self.logger.info('K-medoids clustering: k={}, it={}'.format(k, it))
        clusterer = TSCluster(num_clust=k)
        clusterer.k_medoids_clust(data, dist_matrix, it)

        return clusterer

    # Save and load
    def save_kclust(self, clusterer, alg, n, w, ds, r, k, it):
        """
        Save centroids, assignments, plots, and ts-dists
        """
        self.logger.info('Saving centroids, assignments, figure')
        params_str = KCLUST_STR.format(n, w, ds, r, k, it)

        # Save data
        centroids_path = self._get_centroids_path(alg, params_str)
        medoids_path = self._get_medoids_path(alg, params_str)
        assignments_path = self._get_assignments_path(alg, params_str)
        ts_dists_path = self._get_ts_dists_path(alg, params_str)
        with open(centroids_path, 'wb') as f:
            pickle.dump(clusterer.centroids, f, protocol=2)
        with open(medoids_path, 'wb') as f:
            pickle.dump(clusterer.medoids, f, protocol=2)
        with open(assignments_path, 'wb') as f:
            pickle.dump(clusterer.assignments, f, protocol=2)
        with open(ts_dists_path, 'wb') as f:
            pickle.dump(clusterer.ts_dists, f, protocol=2)

        # Save figures
        for i, c in enumerate(clusterer.centroids):
            plt.plot(c, label=i)
        plt.legend()
        plt.savefig(os.path.join(OUTPUTS_PATH, 'imgs', '{}-centroids_{}.png'.format(alg, params_str)))
        plt.gcf().clear()               # clear figure so for next k

        # Plot medoids if kmedoids
        if alg == 'kmedoids':
            for i, m in enumerate(clusterer.medoids):
                plt.plot(m, label=i)
            plt.legend()
            plt.savefig(os.path.join(OUTPUTS_PATH, 'imgs', '{}-medoids_{}.png'.format(alg, params_str)))
            plt.gcf().clear()

        # Some extra logging
        for centroid_idx, assignments in clusterer.assignments.items():
            self.logger.info('Centroid {}: {} series'.format(centroid_idx, len(assignments)))

    def _load_ts_dists(self, alg, n, w, ds, r, k, it):
        """
        Used for computing elbow plot
        """
        params_str = KCLUST_STR.format(n, w, ds, r, k, it)
        path = self._get_ts_dists_path(alg, params_str)
        ts_dists = pickle.load(open(path, 'rb'))
        return ts_dists

    def _get_centroids_path(self, alg, params_str):
        path = os.path.join(OUTPUTS_PATH, 'data', '{}-centroids_{}.pkl'.format(alg, params_str))
        return path

    def _get_medoids_path(self, alg, params_str):
        path = os.path.join(OUTPUTS_PATH, 'data', '{}-medoids_{}.pkl'.format(alg, params_str))
        return path

    def _get_assignments_path(self, alg, params_str):
        path = os.path.join(OUTPUTS_PATH, 'data', '{}-assignments_{}.pkl'.format(alg, params_str))
        return path

    def _get_ts_dists_path(self, alg, params_str):
        path = os.path.join(OUTPUTS_PATH, 'data', '{}-ts-dists_{}.pkl'.format(alg, params_str))
        return path

    ####################################################################################################################
    # 4) Create elbow plot for kmedoids
    ####################################################################################################################
    def compute_kclust_error(self, alg, n, w, ds, r, k, it):
        """
        Calculate coherence of each cluster for each k. Return dictionary from k to error.
        
        Params
        ------
        - Compute sum_over_series{DTWDistance(series, centroid} for each k., basically WCSS
        - alg: kmeans or kmedoids
        - w: window used to smoooth
        - n: int, number of time series, will be part of filename in previously saved files
        - ds: int, downsample rate used to preprocess
        - r: window size for LB_Keogh
        - k is a comma-separated list of ints
        - it: int, number of iterations used for clustering
        """
        self.logger.info('CREATING ELBOW PLOT')

        k2error = {}
        for cur_k in k.split(','):
            cur_k = int(cur_k)
            error = 0.0
            try:
                ts_dists = self._load_ts_dists(alg, n, w, ds, r, cur_k, it)
                for c_idx, c_members in ts_dists.items():
                    for m_idx, dist in c_members.items():
                        error += dist
                error /= float(n)
                k2error[cur_k] = error
                self.logger.info('k: {}, error: {}'.format(cur_k, error))
            except Exception as e:
                print e

        return k2error

    def save_kclust_error(self, k2error, alg, n, w, ds, r, k, it):
        """
        Use k2error to save elbow plot
        """
        params_str = KCLUST_STR.format(n, w, ds, r, k, it)
        data_out_path = os.path.join(OUTPUTS_PATH, 'data', '{}-error_{}.pkl'.format(alg, params_str))
        with open(data_out_path, 'wb') as f:
            pickle.dump(k2error, f, protocol=2)

        # Plot
        plt.plot(k2error.keys(), k2error.values())
        plt.xlabel('k')
        plt.ylabel('Within cluster distance')
        img_out_path = os.path.join(OUTPUTS_PATH, 'imgs', '{}-error_{}.pdf'.format(alg, params_str))
        plt.savefig(img_out_path)
        plt.gcf().clear()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cluster time series')

    # Action to take
    parser.add_argument('--make_fake_data', dest='make_fake_data', action='store_true')
    parser.add_argument('--prepare_ts', dest='prepare_ts', action='store_true')
    parser.add_argument('--compute_dtw_dist_matrix', dest='compute_dtw_dist_matrix', action='store_true')
    parser.add_argument('--cluster_ts', dest='cluster_ts', action='store_true')
    parser.add_argument('-m', '--method', dest='method', default='kmedoids', help='kmeans,kmedoids,hierarchical,hdbscan')
    parser.add_argument('--compute_kclust_error', dest='compute_kclust_error', action='store_true')

    # Time series data parameters
    parser.add_argument('--data_path', dest='data_path', default=None, help='path to pkl data')
    parser.add_argument('-n', dest='n', default=None, help='n - get from filename')
    parser.add_argument('-w', dest='w', type=float, default=None,
                        help='window size for smoothing predictions.'
                             'If w is a decimal, then it is used as the ratio of the length of a video')
    parser.add_argument('-ds', dest='ds', type=int, default=None, help='downsample rate')

    # Distance matrix parameters
    parser.add_argument('-r', dest='r', type=int, default=None, help='LB_Keogh window size')

    # Clustering parameters
    parser.add_argument('-k', dest='k', default=None, help='k-medoids: list of comma-separated k to evaluate')
    parser.add_argument('-it', dest='it', type=int, default=None, help='k-medoids: number of iterations')

    cmdline = parser.parse_args()

    if cmdline.w >= 1:
        cmdline.w = int(cmdline.w)

    analysis = ClusterAnalysis()

    # Take action
    if cmdline.make_fake_data:
        analysis.make_fake_data(n=10, l=100)

    elif cmdline.prepare_ts:
        ts_data = analysis.prepare_ts(cmdline.data_path, cmdline.w, cmdline.ds)
        analysis.save_ts(ts_data, ts_data.shape[0], cmdline.w, cmdline.ds)

    elif cmdline.compute_dtw_dist_matrix:
        ts_data = analysis.load_ts(cmdline.n, cmdline.w, cmdline.ds)
        dist_matrix = analysis.compute_dtw_dist_matrix(ts_data, cmdline.r)
        analysis.save_dtw_dist_matrix(dist_matrix, cmdline.n, cmdline.w, cmdline.ds, cmdline.r)

    elif cmdline.cluster_ts:
        ts_data = analysis.load_ts(cmdline.n, cmdline.w, cmdline.ds)
        dist_matrix = analysis.load_dtw_dist_matrix(cmdline.n, cmdline.w, cmdline.ds, cmdline.r)

        for k in cmdline.k.split(','):
            print '=' * 100
            k = int(k)
            clusterer = analysis.cluster_ts(ts_data, dist_matrix, cmdline.method, k, cmdline.it)
            analysis.save_kclust(clusterer, cmdline.method, cmdline.n, cmdline.w, cmdline.ds, cmdline.r, k, cmdline.it)

    elif cmdline.compute_kclust_error:
        k2error = analysis.compute_kclust_error(cmdline.method,
                                      cmdline.n, cmdline.w, cmdline.ds,
                                      cmdline.r,
                                      cmdline.k, cmdline.it)
        analysis.save_kclust_error(k2error, cmdline.method, cmdline.n, cmdline.w, cmdline.ds,
                                   cmdline.r, cmdline.k, cmdline.it)

# Make outputs/data, outputs/img