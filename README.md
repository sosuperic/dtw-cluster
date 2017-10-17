# Shape-based clustering of time series using dynamic time warping

## Setup
1. Create `outputs/data` and `outputs/imgs` folder


## Test on example data
```
python cluster.py --make_fake_data
python cluster.py --prepare_ts --data_path test_ts_data.pkl -w 10 -ds 1
python cluster.py --compute_dtw_dist_matrix -n 10 -w 10 -ds 1 -r 10
python cluster.py --cluster_ts -n 10 -w 10 -ds 1 -r 10 -k 2,3,4,5,6,7,8,9,10 -it 100
python cluster.py --compute_kclust_error -n 10 -w 10 -ds 1 -r 10 -k 2,3,4,5,6,7,8,9,10 -it 100
```
#### Notes on parameters:
- w: window size for smoothing (int, e.g. int(0.1 * n))
- ds: downsample rate
- n: number of samples in dataset
- r: window size for LB_Keogh (int, e.g. int(0.03 * n))
- k: comma-separated ints for values of k in k-medoids
- it: number of iterations to run k-medoids