import astropy as ast
from astropy.coordinates import solar_system_ephemeris, EarthLocation, GeocentricTrueEcliptic, get_body, SkyCoord, Distance
from astroplan.moon import moon_phase_angle
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

# All of the celestial bodies to generate synthetic data on
BODY_NAMES = ['mercury', 'venus', 'mars', 'jupiter', 'saturn', 'moon', 'sun']


def get_coordinates(body):
    # Takes a Skycoord object, returns (theta, phi, r, x, y, z) in (deg, deg, AU, AU, AU, AU)
    angles = [float(i) for i in body.to_string().split(' ')]
    body_dist_string = body.distance.to_string()
    r = float(body_dist_string[:-3])
    units = body_dist_string[-2:]
    if units == 'km':
        r /= 1.496e+8

    phi = angles[0]
    theta = angles[1]

    # Extract the Cartesian coordinates from the SkyCoord object
    c = body.cartesian
    x = c.x.value
    y = c.y.value
    z = c.z.value

    body_dist_string = body.distance.to_string()
    units = body_dist_string[-2:]

    # Convert from km to AU if necessary
    if units == 'km':
        x /= 1.496e+8
        y /= 1.496e+8
        z /= 1.496e+8
    return (phi, theta, r, x, y, z)


def random_dates(start, end, n=10):

    start_u = start.value//10**9
    end_u = end.value//10**9

    return pd.to_datetime(np.random.randint(start_u, end_u, n), unit='s')


def add_noise(coords):
    # Takes a tuple of (theta, phi, r, x, y, z) in (deg, deg, AU, AU, AU, AU)
    # and returns a noisified version of the data
    # Currently doesn't add noise to r
    theta, phi, r, x, y, z = coords
    return (theta + np.random.normal(0,1), phi + np.random.normal(0,1), r + np.random.normal(0, 0.1),
            x + np.random.normal(0, 0.1), y + np.random.normal(0, 0.1), z + np.random.normal(0, 0.1))


def main(save_name, start="1850-01-01-00-00-00", end="2000-01-01-00-00-00", freq='1D'):
    times = pd.date_range(start=start, end=end, freq=freq)

    # Location is the Medicina Radio Observatory, located in Italy. Chosen for proximity to Greece
    loc = EarthLocation.of_site('medicina')
    rows = defaultdict(list)

    # Generate coordinate data in terms of spherical Geocentric Celestial Reference System (GCRS), default for astropy
    for time in tqdm(times):
        time = ast.time.Time(time.to_pydatetime())
        bodies = []

        with solar_system_ephemeris.set('builtin'):
            for body_name in BODY_NAMES:
                bodies.append(get_body(body_name, time, loc))

        rows['time'].append(time)
        rows['location'].append(str(loc))
        rows['moon_phase'].append(moon_phase_angle(time).value)

        for body_name, body in zip(BODY_NAMES, bodies):
            coordinates = add_noise(get_coordinates(body))
            coord_strings = ['theta', 'phi', 'r', 'x', 'y', 'z']
            for i in range(len(coord_strings)):
                c = coordinates[i]
                rows[body_name + '_' + coord_strings[i]].append(c)

    celestial_bodies = pd.DataFrame(rows)
    celestial_bodies.to_csv('./raw_features/' + save_name, index=False)

    # Load the generated data (which needs to be converted to the geocentric ecliptic coordinate system)
    data = pd.read_csv('./raw_features/' + save_name)

    # Convert coordinates to the standard geocentric true ecliptic coordinate system
    # See https://docs.astropy.org/en/stable/api/astropy.coordinates.GeocentricTrueEcliptic.html for more documentation

    rows = defaultdict(list)
    for name in tqdm(BODY_NAMES):
        phi_col = data[name + '_phi']
        theta_col = data[name + '_theta']
        r_col = data[name + '_r']
        for phi, theta, r in zip(phi_col, theta_col, r_col):
            ecliptic = SkyCoord(theta, phi, abs(r), frame='gcrs', unit=('deg', 'deg', 'AU')).transform_to(
                GeocentricTrueEcliptic())
            coordinates = get_coordinates(ecliptic)
            coord_strings = ['lambda', 'beta', 'delta', 'x', 'y', 'z']
            for i in range(len(coord_strings)):
                c = coordinates[i]
                rows[name + '_' + coord_strings[i]].append(c)

    # Prepare the final dataset as final_data.csv

    final_data = pd.DataFrame(rows)
    final_data['time'] = data['time']  # Time is given in yyyy-mm-dd hh:mm:ss format
    final_data['location'] = data['location']  # Location is given as (longitude, latitude, height) in m
    final_data['moon_phase'] = data['moon_phase']  # Moon phase
    final_data.set_index('time', inplace=True)
    # All other columns are the spherical and Cartesian geocentric true ecliptic coordinate system values
    # as described in the writeup
    final_data.to_csv('./raw_features/final_' + save_name)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create fourier feature matrix')

    parser.add_argument('save_name', type=str, help='Name of file e.g. data_5k_greece')
    parser.add_argument('--start', type=str, default="1999-01-01-00-00-00",
                        help='Date in %Y-%m-%d-%H-%M-%S')
    parser.add_argument('--end', type=str, default="2000-01-01-00-00-00",
                        help='Date in %Y-%m-%d-%H-%M-%S')
    parser.add_argument('--frequency', type=str, default='1D',
                        help='Frequency of sample e.g. 1D is 1 day, 1M is 1 month, ...')

    args = parser.parse_args()

    main(args.save_name, args.start, args.end, args.frequency)
