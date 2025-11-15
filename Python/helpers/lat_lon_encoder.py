import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class LatLonEncoder(BaseEstimator, TransformerMixin):
	"""
	A transformer that encodes latitude and longitude coordinates using trigonometric functions.

	This transformer converts latitude and longitude coordinates into sine and cosine
	components to handle the cyclical nature of geographic coordinates, which improves
	model performance for location-based features.

	Notes
	-----
	Statistical Theory Principles:
	- Trigonometric encoding addresses the discontinuity at ±180° longitude and ±90° latitude
	- Sine and cosine transformations preserve spatial relationships while making coordinates continuous
	- This approach prevents models from treating nearby coordinates as distant (e.g., 179° and -179° longitude)
	- The encoding maintains the Euclidean distance properties important for spatial analysis

	Examples
	--------
	>>> import pandas as pd
	>>> from lat_lon_encoder import LatLonEncoder
	>>>
	>>> # Sample data with geographic coordinates
	>>> data = pd.DataFrame({
	...     'latitude': [40.7128, 34.0522, 51.5074],
	...     'longitude': [-74.0060, -118.2437, -0.1278]
	... })
	>>>
	>>> # Initialize and transform
	>>> encoder = LatLonEncoder()
	>>> transformed = encoder.transform(data)
	>>> print(transformed.columns)
	Index(['latitude_sin', 'latitude_cos', 'longitude_sin', 'longitude_cos'], dtype='object')
	>>>
	>>> # With original columns preserved
	>>> encoder_preserve = LatLonEncoder(drop_original=False)
	>>> transformed_preserve = encoder_preserve.transform(data)
	>>> print(transformed_preserve.columns)
	Index(['latitude', 'longitude', 'latitude_sin', 'latitude_cos',
		   'longitude_sin', 'longitude_cos'], dtype='object')

	Parameters
	----------
	lat_col : str, default="latitude"
		Name of the latitude column in the DataFrame
	lon_col : str, default="longitude"
		Name of the longitude column in the DataFrame
	drop_original : bool, default=True
		Whether to drop the original latitude and longitude columns after transformation

	Attributes
	----------
	lat_col : str
		Name of the latitude column
	lon_col : str
		Name of the longitude column
	drop_original : bool
		Whether to drop original columns after transformation
	"""

	def __init__(
		self,
		lat_col: str = "latitude",
		lon_col: str = "longitude",
		drop_original: bool = True,
	):
		self.lat_col = lat_col
		self.lon_col = lon_col
		self.drop_original = drop_original

	def fit(self, X: pd.DataFrame, y=None) -> "LatLonEncoder":
		"""
		Fit the transformer (no operation needed as this is a stateless transformer).

		Parameters
		----------
		X : pd.DataFrame
			Input DataFrame containing latitude and longitude columns
		y : array-like, default=None
			Target values (ignored)

		Returns
		-------
		self : LatLonEncoder
			Returns the transformer instance
		"""
		return self

	def transform(self, X: pd.DataFrame) -> pd.DataFrame:
		"""
		Transform latitude and longitude coordinates into trigonometric features.

		Parameters
		----------
		X : pd.DataFrame
			Input DataFrame containing latitude and longitude columns

		Returns
		-------
		X_transformed : pd.DataFrame
			DataFrame with original features plus trigonometric latitude/longitude features

		Raises
		------
		ValueError
			If the specified latitude or longitude columns are not found in the DataFrame
		"""
		if not {self.lat_col, self.lon_col}.issubset(X.columns):
			raise ValueError(
				f"Columns '{self.lat_col}' and '{self.lon_col}' must exist in input DataFrame."
			)

		# Convert to radians
		lat_rad = np.radians(X[self.lat_col])
		lon_rad = np.radians(X[self.lon_col])

		# Create transformed features
		X_transformed = X.copy()
		X_transformed[f"{self.lat_col}_sin"] = np.sin(lat_rad)
		X_transformed[f"{self.lat_col}_cos"] = np.cos(lat_rad)
		X_transformed[f"{self.lon_col}_sin"] = np.sin(lon_rad)
		X_transformed[f"{self.lon_col}_cos"] = np.cos(lon_rad)

		if self.drop_original:
			X_transformed = X_transformed.drop(columns=[self.lat_col, self.lon_col])

		return X_transformed
