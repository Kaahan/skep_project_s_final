from utils import *
import numpy as np
import pandas as pd
from astropy.time import Time

# Classification
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import fbprophet


# Command-line
import sys
import warnings
warnings.filterwarnings("ignore") # ignore ErfaWarning, e.g. WARNING: ErfaWarning: ERFA function "utctai" yielded 1 of "dubious year (Note 3)" [astropy._erfa.core]
import contextlib # ignore fbprophet's Prophet model output



class PredictIncidence():
	def __init__(self, date, time):
		self.date = date
		self.time = time

	def load_data(self):
		data = pd.read_csv('solar_eclipse_final_data.csv')
		data = data.sort_values('ISO Date')
		data['ISO Date'] = data['ISO Date'].apply(lambda x: Time(x))
		return data


	def train_model(self, data):
		X_train = np.array(data[['moon_theta', 'moon_phi', 'sun_theta', 'sun_phi']])
		X_train = np.append(X_train, [[d.jd] for d in data['ISO Date']], axis=1)
		y_train = np.array(data['eclipse'])

		clf = QuadraticDiscriminantAnalysis()
		clf.fit(X_train, y_train)
		return clf

	def predict(self):
		data = self.load_data()
		clf = self.train_model(data)

		X_test, iso_dates = get_X_test(self.date, self.time)
		X_test = np.append(X_test, [[d.jd] for d in iso_dates], axis=1)
		if clf.predict(X_test)[0] == 1:
			return True
		else:
			return False

class PredictLatitude():
	def __init__(self, date, time):
		self.date = date
		self.time = time
		self.poly = PolynomialFeatures(degree=7)

	def load_data(self):
		data = pd.read_csv('solar_eclipse_pos_data.csv')
		return data

	def train_model(self, data):
		X_train = data[['moon_theta', 'moon_phi', 'sun_theta', 'sun_phi']]
		X_train = pd.concat([X_train, data['ISO Date'].str.split('T').apply(lambda x: to_jd(x[0].split('-')))], axis=1)
		y_train = data[['latitude']]

		X_train = self.poly.fit_transform(X_train)
		reg = LinearRegression()
		reg.fit(X_train, y_train)
		return reg
	
	def predict(self):
		data = self.load_data()
		reg = self.train_model(data)

		X_test, iso_dates = get_X_test(self.date, self.time)
		iso_dates = iso_dates.astype(str)
		X_test = pd.DataFrame(X_test)
		X_test = pd.concat([X_test, iso_dates.str.split('T').apply(lambda x: to_jd(x[0].split('-')))], axis=1)
		X_test = self.poly.fit_transform(X_test)
		return reg.predict(X_test)

class PredictLongitude:
	def __init__(self, date, time):
		self.date = date
		self.time = time

	def load_data(self):
		# data = pd.read_csv('solar_eclipse_data.csv')
		# data['ISO Date'] = data['Calendar Date'] + ' ' + data['Eclipse Time']
		# data['ISO Date'] = data['ISO Date'].apply(lambda x: to_isoformat(x))


		data = pd.read_csv('solar_eclipse_data.csv')
		data = data[(117 <= data['Saros Number']) & (data['Saros Number'] < 157)] # Saros 117 through 157 are ongoing
		prior_to_2020 = data['Calendar Date'].str.split().apply(lambda x: int(x[0]) <= 2020) # Select for years prior to 2020
		data = data[prior_to_2020]

		jds = []
		for _, row in data.iterrows():
			jds.append(to_jd(row['Calendar Date'].split(), row['Eclipse Time']))

		data['Julian Day'] = jds

		return data

	def train_model(self, data, n, k):
		"""
		Train a model to predict longitude for Saros n and offset k.
		"""
		data = data[data['Saros Number'] == n]
		if len(data) == 0:
			raise ValueError
		pd.options.mode.chained_assignment = None # handle false-positive warning. See https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
		
		ks = []
		for _, row in data.iterrows():
			ks.append(get_offset_in_saros(row['Calendar Date'], row['Eclipse Time']))
		data['k'] = ks
		pd.options.mode.chained_assignment = 'warn' # reset to default value
		
		data = data[data['k'] == k]
		data = data[['Longitude']]
		data['Longitude'] = data['Longitude'].apply(lambda x: convert_longitude(x))
		data['ISO Date'] = pd.date_range(start='1/1/1900', periods=len(data))

		data.rename(columns={'ISO Date': 'ds', 'Longitude': 'y'}, inplace=True)
		data['y'] = data['y']
		num_eclipses = len(data)
		if num_eclipses >= 16: # future work - tune the hyperparameters for Prophet
			prophet = fbprophet.Prophet(
					n_changepoints=4, changepoint_prior_scale=3, daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False
				).add_seasonality('daily', period=15, fourier_order=2)
		else:
			prophet = fbprophet.Prophet(changepoint_prior_scale=1, daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False
				).add_seasonality('daily', period=12, fourier_order=2)
		prophet.fit(data)
		# Each "day" corresponds to a Saros cycle (e.g. period of approx. 18 years).
		# We use days since pandas' datetime is limited to a smaller timespan than what we need.
		return prophet, num_eclipses


	def predict(self):
		data = self.load_data()
		month_num = self.date.split('-')[1]
		month_lookup = {'01': 'January', '02': 'February', '03': 'March', '04': 'April', '05': 'May', '06': 'June',
						'07': 'July', '08': 'August', '09': 'September', '10': 'October', '11': 'November', '12': 'December'}
		month = month_lookup[month_num]
		day = self.date.split('-')[2]
		year = self.date.split('-')[0]
		date = '%s %s %s' % (day, month, year)
		time = self.time

		offset_to_saros_number = get_offset_to_saros_number(data)
		n = predict_saros(date, offset_to_saros_number)
		k = get_offset_in_saros(date, time)
		eclipse_num = get_eclipse_number(data, n, date, time)

		with suppress_stdout_stderr():
			prophet, num_eclipses = self.train_model(data, n, k)

		num_periods = num_eclipses - eclipse_num // 3 - 1 # how many more eclipses to predict. Divide by 3 to account for the fact we trained on every 3 eclipses

		forecast = prophet.make_future_dataframe(periods=int(num_periods), freq='D')
		forecast = prophet.predict(forecast)
		pred_row = forecast.iloc[int(eclipse_num // 3)] # index into the predicted row
		longitude = mod_yhat(pred_row['yhat'])
		return longitude


if __name__ == '__main__':
	if len(sys.argv) != 3:
		help_msg = "Usage: front_end.py DATE TIME, where DATE is in the format YYYY-MM-DD and time is in the format HH:MM:SS.\nExample: python predict_eclipse.py 2144-05-03 01:02:06\nList of sample eclipses: https://eclipse.gsfc.nasa.gov/SEsaros/SEsaros150.html"
		print(help_msg)
		exit(1)

	date = sys.argv[1]
	time = sys.argv[2]
	if PredictIncidence(date, time).predict():
		print('A solar eclipse is happening on %s at %s.' % (date, time))
		print('Starting location prediction.')
		latitude = PredictLatitude(date, time).predict()[0][0]
		longitude = PredictLongitude(date, time).predict()
		print('Predicted latitude: %.2f' % latitude)
		print('Predicted longitude: %.2f' % longitude)

	else:
		print('No eclipse is happening on %s at %s.' % (date, time))



