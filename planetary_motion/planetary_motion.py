from datetime import datetime, date
import tqdm
import argparse

import numpy as np
import pandas as pd

from models import RidgeSVD, OLS, DFT

BODY_NAMES = ['mercury', 'venus', 'mars', 'jupiter', 'saturn', 'moon', 'sun']
models = {
    'ridge': RidgeSVD,
    'ols': OLS,
    'dft': DFT
}


def main(t, raw_data_path, fourier_data_path, model_type='ridge', lam=None, load_models=False):
    """Train (if necessary) planetary motion models and predict location at time t"""
    # load data
    X = ff = np.load(fourier_data_path)
    raw_data = pd.read_csv(raw_data_path)

    predictions = {}

    if not load_models:
        # train model from scratch (save for future use)
        print("Training models and saving predictions")
        for body in tqdm.tqdm(BODY_NAMES):
            # create rectangular gt
            ecliptical = raw_data[[body + '_lambda', body + '_delta']].values
            ecliptical[:, 0] = ecliptical[:, 0] * (np.pi / 180)
            y = ecliptical[:, 1] * (
                    np.cos(ecliptical[:, 0]) + 1j * np.sin(ecliptical[:, 0]))

            # train model
            model = models[model_type]()
            if lam:
                model.train(X, y, lam)
            else:
                model.train(X, y)

            # save model
            model.save(f"{model_type}_{body}")
            predictions[body] = model.predict(t)
    else:
        # load models
        for body in tqdm.tqdm(BODY_NAMES):
            model = models[model_type]()
            model.load(f"{model_type}_{body}")
            predictions[body] = model.predict(t)

    # report predictions
    print(f"Earth is centered at (0, 0)")
    print(f"Coordinate Predictions (in AU units): ")
    for body in BODY_NAMES:
        print(f"{body.title()}: {predictions[body][1]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create fourier feature matrix')

    parser.add_argument('time', type=str,
                        help='Date in %Y-%m-%d %H:%M:%S')
    parser.add_argument('raw_data_path', type=str,
                        help='Location of raw data (.csv)')
    parser.add_argument('fourier_data_path', type=str,
                        help='Location of fourier feature matrix (.npy)')
    parser.add_argument('model_type', type=str,
                        help='Model to train / load; one of [ridge, ols, dft]')
    parser.add_argument('--load_models', type=bool, default=False,
                        help='Load models (True or False); requires them to be already generated!')

    args = parser.parse_args()

    t = datetime.strptime(args.time.split(' ')[0], '%Y-%m-%d')
    t = (t.date() - date(t.year, 1, 1)).days
    main(t, args.raw_data_path, args.fourier_data_path, args.model_type, load_models=args.load_models)





