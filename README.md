# Project S (Final) - Team SKEP
 The late deadline assignment for Project S CS 189 at Berkeley

### Group Members:
Kaahan Radia
Stephen Lin
Eric Li
Parker Nelson

### How to Generate Datasets:

The two necessary datasets are included (`final_data.csv` for eclipse and moon phase models; `final_motion_data.csv` and `fourier_features_final_5000.npy` for planetary motion)

If you would like to generate the dataset from scratch you can follow the following instructions

From the root directory, run:

```python planetary_motion/generate_raw_data.py final_data.csv --start 1850-01-01-00-00-00 --end 2000-01-01-00-00-00 --freq 5D```

and 

```python planetary_motion/generate_raw_data.py motion_data.csv --start 1995-01-01-00-00-00 --end 2000-01-01-00-00-00 --freq 1D```

as well as 

```python planetary_motion/fourier_data.py ./raw_features/motion_data.csv ./fourier_features/fourier_motion```



### How to use Planetary Motion models

We've included trained models (for Ridge, DFT, and OLS) trained on aforementioned datasets.

You can run inference for some time `t` (format `%Y-%m-%d %H:%M:%S`) by executing the following command (arbitrary date and model chosen):

```python planetary_motion/planetary_motion.py "2000-01-05 00:00:00" ./raw_features/final_motion_data.csv ./fourier_features/fourier_motion.npy ridge --load_models True```

`ridge` can be replaced by `ols` or `dft` to use the respective models.

To train the models from scratch, simply exclude the ``--load_models True`` option i.e. 

```python planetary_motion/planetary_motion.py "2000-01-05 00:00:00" ./raw_features/final_5_year_data.csv ./fourier_features/test_code_5.npy ridge```


### How to use Eclipse and Moon Phase models




### Required Packages
SciPy
AstroPy
Pandas
Numpy
Astroplan
tqdm
sklearn


