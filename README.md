# Project S (Final) - Team SKEP
 The late deadline assignment for Project S CS 189 at Berkeley

### Group Members:
Kaahan Radia
Stephen Lin
Eric Li
Parker Nelson

### How to Generate Datasets:

The three of the four necessary datasets are included (`moon_phase_data.csv` moon phase models; `solar_eclipse_data.csv`; `eclipfinal_motion_data.csv` for planetary motion). Unfortunately our fourier features dataset is too large for GitHub (we're looking into alternate hosting solutions), so please generate the fourier features using the command below.

If you would like to generate the datasets from scratch you can follow the following instructions.

From the root directory, run:

```python planetary_motion/generate_raw_data.py final_data.csv --start 1850-01-01-00-00-00 --end 2000-01-01-00-00-00 --freq 5D```

and 

```python planetary_motion/generate_raw_data.py motion_data.csv --start 1995-01-01-00-00-00 --end 2000-01-01-00-00-00 --freq 1D```

as well as the fourier features generation:

```python planetary_motion/fourier_data.py ./raw_features/motion_data.csv ./fourier_features/fourier_motion```.

Finally, execute the notebooks `eclipse/predict_eclipse.ipynb` and `moon_phase/predict_moon_phase.ipynb` with local variable `GENERATE_DATA=True`. See the notebooks for more details.


### How to use Planetary Motion Models

We've included trained models (for Ridge, DFT, and OLS) trained on aforementioned datasets.

You can run inference for some time `t` (format `%Y-%m-%d %H:%M:%S`) by executing the following command (arbitrary date and model chosen):

```python planetary_motion/planetary_motion.py "2000-01-05 00:00:00" ./raw_features/final_motion_data.csv ./fourier_features/fourier_motion.npy ridge --load_models True```

`ridge` can be replaced by `ols` or `dft` to use the respective models.

To train the models from scratch, simply exclude the ``--load_models True`` option i.e. 

```python planetary_motion/planetary_motion.py "2000-01-05 00:00:00" ./raw_features/final_5_year_data.csv ./fourier_features/test_code_5.npy ridge```

### How to use Eclipse Models

We have included a command-line script for eclipse models. First, navigate to the `eclipse` directory. Then execute the `predict_eclipse.py` script:

```python predict_eclipse.py DATE TIME```, where DATE is in the format YYYY-MM-DD and time is in the format HH:MM:SS.
Example: `python predict_eclipse.py 2144-05-03 01:02:06`

### How to use Moon Phase Models
To use the moon phase models, simply execute the notebook `moon_phase/predict_moon_phase.ipynb`.

### Required Packages
Numpy
SciPy
Pandas
sklearn
tqdm
AstroPy
Astroplan
jdcal
fbprophet


