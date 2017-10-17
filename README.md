# Shape-based clustering of time series using dynamic time warping

## Setup
1. Create `outputs/data` and `outputs/imgs` folder


### Test on example data, where data is a numpy matrix (num_timeseries, timeseries_len)
```
python cluster.py --make_fake_data_same_lengths
python cluster.py --prepare_ts --data_path test_ts_data_matrix.pkl -w 10 -ds 1
python cluster.py --compute_dtw_dist_matrix -n 50 -w 10 -ds 1 -r 10
python cluster.py --cluster_ts -n 50 -w 10 -ds 1 -r 10 -k 2,3,4,5 -it 100
python cluster.py --compute_kclust_error -n 50 -w 10 -ds 1 -r 10 -k 2,3,4,5 -it 100
```

Note: the following example will overwrite outputs for the previous example.

### Test on example data, where data is a list of numpy vectors (i.e. time series of different lengths)
```
python cluster.py --make_fake_data_diff_lengths
python cluster.py --prepare_ts --data_path test_ts_data_list.pkl -w 10 -ds 1
python cluster.py --compute_dtw_dist_matrix -n 50 -w 10 -ds 1 -r 10
python cluster.py --cluster_ts -n 50 -w 10 -ds 1 -r 10 -k 2,3,4,5 -it 100
python cluster.py --compute_kclust_error -n 50 -w 10 -ds 1 -r 10 -k 2,3,4,5 -it 100
```

#### Notes on parameters:
- w: window size for smoothing (int, e.g. int(0.1 * n); or float, e.g 0.1)
- ds: downsample rate
- n: number of samples in dataset
- r: window size for LB_Keogh (int, e.g. int(0.03 * n))
- k: comma-separated ints for values of k in k-medoids
- it: number of iterations to run k-medoids