from collections import defaultdict
import numpy as np
import pandas as pd
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, EarthLocation, GeocentricTrueEcliptic, get_body, SkyCoord
import jdcal
import os

SAROS_LENGTH = 6585.32
JD_OFFSET = 2010506.5

# Celestial bodies to generate synthetic data for
BODY_NAMES = ['moon', 'sun']

# Location is the Medicina Radio Observatory, located in Italy. Chosen for proximity to Greece
LOCATION = EarthLocation.of_site('medicina')

# Utility functions for processing dates and collecting prediction data (i.e. fetching sun/moon coordinates given a time)
def to_jd(calendar_date, time=None):
	month_labels = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
	months = dict(zip(month_labels, range(1, 13)))
	try:
		jd = sum(jdcal.gcal2jd(calendar_date[0], months[calendar_date[1]], calendar_date[2]))
	except KeyError: # if the month is already an integer 
		jd = sum(jdcal.gcal2jd(calendar_date[0], int(calendar_date[1]), calendar_date[2]))
	if time:
		hour, minute, second = time.split(':')
		jd += (int(hour) + int(minute)/60 + int(second)/3600)/24
	return jd

def to_isoformat(date):
	date = date.split()
	months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
	months = dict(zip(months, range(1, 13)))
	return '%s-%d-%s' % (date[0], months[date[1]], date[2])

def convert_latitude(latitude):
	if latitude[-1] == 'N':
		return float(latitude[:-1])
	elif latitude[-1] == 'S':
		return -float(latitude[:-1])
	else:
		raise ValueError
		
def convert_longitude(longitude):
	if longitude[-1] == 'E':
		return float(longitude[:-1])
	elif longitude[-1] == 'W':
		return -float(longitude[:-1])
	else:
		raise ValueError


def get_X_test(date, time):
	def get_coordinate_rows(times):
		def get_coordinates(body):
			# Takes a Skycoord object, returns (theta, phi) in (deg, deg)
			angles = [float(i) for i in body.to_string().split(' ')]
			
			phi = angles[0]
			theta = angles[1]
			return phi, theta

		rows = defaultdict(list)
		for time in times:
			bodies = []

			with solar_system_ephemeris.set('builtin'):
				for body_name in BODY_NAMES:
					bodies.append(get_body(body_name, time, LOCATION))

			rows['time'].append(time)
			rows['location'].append(str(LOCATION))

			for body_name, body in zip(BODY_NAMES, bodies):
				theta, phi = get_coordinates(body)
				ecliptic = SkyCoord(theta, phi, frame='gcrs', unit=('deg', 'deg')).transform_to(GeocentricTrueEcliptic())
				coordinates = get_coordinates(ecliptic)
				coord_strings = ['theta', 'phi']

				for i in range(len(coord_strings)):
					c = coordinates[i]
					rows[body_name + '_' + coord_strings[i]].append(c)
		return rows

	times = [Time(date + 'T' + time + '.000')]
	rows = get_coordinate_rows(times)
	data = pd.DataFrame(rows)
	data.rename(columns={'time': 'ISO Date'}, inplace=True)
	data = data.sort_values('ISO Date')
	data['ISO Date'] = data['ISO Date'].apply(lambda x: Time(x))

	X_test = np.array(data[['moon_theta', 'moon_phi', 'sun_theta', 'sun_phi']])
	iso_dates = data['ISO Date']
	return X_test, iso_dates


def predict_saros(calendar_date, offset_to_saros_number):
	julian_day = to_jd(calendar_date.split())
	offset = (julian_day - JD_OFFSET) % SAROS_LENGTH
	min_diff = float('inf')
	saros = None
	for curr_offset in offset_to_saros_number:
		diff = abs(offset - curr_offset)
		if diff < min_diff:
			min_diff = diff
			saros = offset_to_saros_number[curr_offset]
	return saros

def get_offset_to_saros_number(data):
	active_saros = list(range(117, 157)) # Saros 117 through 157 are ongoing
	saros_offsets = (data['Julian Day'] - JD_OFFSET) % SAROS_LENGTH
	saros_table = pd.DataFrame({'Saros Number': data['Saros Number'], 'Saros Offset': saros_offsets})
	offset_to_saros_number = {}
	for _, row in saros_table.iterrows():
		saros = row['Saros Number']
		if saros in active_saros:
			active_saros.remove(saros)
			offset_to_saros_number[int(row['Saros Offset'])] = int(saros)
	return offset_to_saros_number


def predict_saros(calendar_date, offset_to_saros_number):
	julian_day = to_jd(calendar_date.split())
	offset = (julian_day - JD_OFFSET) % SAROS_LENGTH
	min_diff = float('inf')
	saros = None
	for curr_offset in offset_to_saros_number:
		diff = abs(offset - curr_offset)
		if diff < min_diff:
			min_diff = diff
			saros = offset_to_saros_number[curr_offset]
	return saros

def get_offset_in_saros(calendar_date, time):
	julian_day = to_jd(calendar_date.split(), time)
	return ((julian_day - JD_OFFSET) // SAROS_LENGTH) % 3

def get_eclipse_number(data, n, calendar_date, time):
	data = data[data['Saros Number'] == n]
	start = data.iloc[0]['Julian Day']
	jd = to_jd(calendar_date.split(), time)
	return (jd - start) // (SAROS_LENGTH - 0.1) # a small constant is needed for measurement inaccuracy, found empirically

def mod_yhat(yhat):
	while True:
		if yhat > 180:
			yhat -= 360
		elif yhat < -180:
			yhat += 360
		else:
			break
	return yhat

# See https://github.com/facebook/prophet/issues/223. For surpressing fbprophet output.
# from https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
class suppress_stdout_stderr(object):
	'''
	A context manager for doing a "deep suppression" of stdout and stderr in
	Python, i.e. will suppress all print, even if the print originates in a
	compiled C/Fortran sub-function.
	   This will not suppress raised exceptions, since exceptions are printed
	to stderr just before a script exits, and after the context manager has
	exited (at least, I think that is why it lets exceptions through).

	'''
	def __init__(self):
		# Open a pair of null files
		self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
		# Save the actual stdout (1) and stderr (2) file descriptors.
		self.save_fds = (os.dup(1), os.dup(2))

	def __enter__(self):
		# Assign the null pointers to stdout and stderr.
		os.dup2(self.null_fds[0], 1)
		os.dup2(self.null_fds[1], 2)

	def __exit__(self, *_):
		# Re-assign the real stdout/stderr back to (1) and (2)
		os.dup2(self.save_fds[0], 1)
		os.dup2(self.save_fds[1], 2)
		# Close the null files
		os.close(self.null_fds[0])
		os.close(self.null_fds[1])
