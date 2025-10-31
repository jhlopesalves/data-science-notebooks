"""
OlistAnalyzer: A class for analyzing Olist E-commerce data.

This module provides a unified interface for loading, processing, and analyzing
the Olist Brazilian E-commerce dataset from multiple CSV files.
"""

from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests


class OlistAnalyzer:
	"""
	A comprehensive analyzer for the Olist E-commerce dataset.

	This class encapsulates the entire data analysis pipeline, from loading
	raw CSV files to generating customer summaries and sales pivot tables.

	Attributes
	----------
	data_path : str
	    Path to the directory containing Olist CSV files or GitHub folder URL.
	data_dict : dict
	    Dictionary mapping table names to their respective DataFrames.
	master_df : pd.DataFrame or None
	    The merged master table containing all relevant order information.
	customer_summary_df : pd.DataFrame or None
	    Customer-level summary statistics.

	Examples
	--------
	>>> analyzer = OlistAnalyzer(data_path='data/')
	>>> customer_summary = analyzer.build_customer_summary()
	>>> sales_pivot = analyzer.get_sales_pivot()
	"""

	def __init__(self, data_path: str, use_github: bool = False):
		"""
		Initialize the OlistAnalyzer with data source.

		Parameters
		----------
		data_path : str
		    Path to local directory containing CSV files or GitHub folder URL.
		use_github : bool, optional
		    If True, treat data_path as a GitHub URL, by default False.
		"""
		self.data_path = data_path
		self.data_dict: dict[str, pd.DataFrame] = {}
		self.master_df = None
		self.customer_summary_df = None
		self.use_github = use_github
		self._load_data()

	def _load_data(self) -> None:
		"""
		Load CSV files into a dictionary of DataFrames.

		This private method is called during initialization and loads all
		CSV files from either a local directory or GitHub repository.
		"""
		if self.use_github:
			self.data_dict = self._load_data_github(self.data_path)
		else:
			self.data_dict = self._load_data_local(self.data_path)

	def _load_data_local(self, data_path: str) -> dict[str, pd.DataFrame]:
		"""
		Load CSV files from a local directory into a dictionary of DataFrames.

		Returns
		-------
		dict
		    A dictionary where keys are cleaned file names and values are
		    DataFrames.

		Raises
		------
		ValueError
		    If the path does not exist or is not a directory.
		"""

		path = Path(data_path)

		# Check if the path exists and is a directory
		if not path.exists() or not path.is_dir():
			raise ValueError(
				f"The path '{data_path}' does not exist or is not a directory."
			)

		# Build file_map: cleaned keys to original filenames
		file_map = {
			file.name.replace("olist_", "")
			.replace("_dataset.csv", "")
			.replace(".csv", ""): file.name
			for file in path.iterdir()
			if file.is_file() and file.suffix == ".csv"
		}

		# Build data_map: cleaned keys to loaded DataFrames
		data_map = {
			key: pd.read_csv(path / filename) for key, filename in file_map.items()
		}
		return data_map

	def _load_data_github(self, github_folder_url: str) -> dict[str, pd.DataFrame]:
		"""
		Load CSV files from a GitHub folder URL into a dictionary of DataFrames.

		Returns
		-------
		dict
		    A dictionary where keys are cleaned file names and values are
		    DataFrames.

		Raises
		------
		ValueError
		    If the URL is invalid, the folder does not exist, or no CSV
		    files are found.
		"""

		# Parse the GitHub URL to extract owner, repo, branch, folder path
		parsed = urlparse(github_folder_url)
		if parsed.hostname != "github.com":
			raise ValueError("URL must be a valid GitHub repository URL.")

		path_parts = parsed.path.strip("/").split("/")
		if len(path_parts) < 4 or path_parts[2] != "tree":
			raise ValueError(
				"URL must point to a GitHub folder (e.g., /tree/branch/path/to/folder)."
			)
		owner = path_parts[0]
		repo = path_parts[1]
		branch = path_parts[3]
		folder_path = "/".join(path_parts[4:])

		# Construct GitHub API URL to list folder contents
		base_url = "https://api.github.com/repos"
		api_url = f"{base_url}/{owner}/{repo}/contents/{folder_path}?ref={branch}"

		# Fetch the contents
		response = requests.get(api_url, timeout=10)
		if response.status_code != 200:
			raise ValueError(
				f"Failed to fetch GitHub contents: "
				f"{response.status_code} - {response.text}"
			)
		contents = response.json()
		if not isinstance(contents, list):
			raise ValueError(
				"GitHub API did not return a list of files (folder may not exist)."
			)

		# Build file_map: cleaned keys to raw download URLs
		file_map = {}
		for item in contents:
			if item["type"] == "file" and item["name"].endswith(".csv"):
				cleaned_key = (
					item["name"]
					.replace("olist_", "")
					.replace("_dataset.csv", "")
					.replace(".csv", "")
				)
				raw_url = item["download_url"]
				file_map[cleaned_key] = raw_url
		if not file_map:
			raise ValueError("No CSV files found in the GitHub folder.")

		data_map = {key: pd.read_csv(url) for key, url in file_map.items()}

		return data_map

	def build_master_table(self) -> pd.DataFrame:
		"""
		Create a master table by merging multiple DataFrames.

		Performs a series of left joins to combine order, item, product,
		customer, payment, and category translation data. Filters for
		orders with status 'delivered'.

		Returns
		-------
		pd.DataFrame
		    The merged master DataFrame.

		Raises
		------
		KeyError
		    If required DataFrames are missing.
		ValueError
		    If required DataFrames are empty.
		"""

		# Required keys for the merge operation
		required_keys = [
			"orders",
			"order_items",
			"order_payments",
			"products",
			"customers",
			"product_category_name_translation",
		]
		missing_keys = [key for key in required_keys if key not in self.data_dict]
		if missing_keys:
			raise KeyError(f"Missing required DataFrames in data_dict: {missing_keys}")

		orders = self.data_dict["orders"].copy()
		items = self.data_dict["order_items"].copy()
		payments = self.data_dict["order_payments"].copy()
		products = self.data_dict["products"].copy()
		customers = self.data_dict["customers"].copy()
		translation = self.data_dict["product_category_name_translation"].copy()

		orders_delivered = orders[orders["order_status"] == "delivered"].copy()

		items_products = items.merge(products, on="product_id", how="left")
		items_products = items_products.merge(
			translation, on="product_category_name", how="left"
		)

		order_items_agg = (
			items_products.groupby("order_id")
			.agg(
				total_price=("price", "sum"),
				total_freight=("freight_value", "sum"),
				total_items=("order_item_id", "count"),
				category_english=("product_category_name_english", "first"),
			)
			.reset_index()
		)

		order_payments_agg = (
			payments.groupby("order_id")
			.agg(
				total_payment=("payment_value", "sum"),
				payment_installments=("payment_installments", "max"),
			)
			.reset_index()
		)

		merged_df = orders_delivered.merge(order_items_agg, on="order_id", how="left")
		merged_df = merged_df.merge(order_payments_agg, on="order_id", how="left")
		merged_df = merged_df.merge(customers, on="customer_id", how="left")

		self.master_df = merged_df
		return merged_df

	def build_customer_summary(self) -> pd.DataFrame:
		"""
		Build customer summary with aggregated metrics.

		Returns
		-------
		pd.DataFrame
		    Customer summary DataFrame with engineered features.

		Raises
		------
		ValueError
		    If master_df has not been built yet.
		"""
		if self.master_df is None:
			raise ValueError(
				"Master table has not been built yet. Call build_master_table() first."
			)

		# Ensure order_purchase_timestamp is datetime for proper min/max
		self.master_df["order_purchase_timestamp"] = pd.to_datetime(
			self.master_df["order_purchase_timestamp"]
		)

		# Group by customer_unique_id and aggregate
		customer_summary = self.master_df.groupby("customer_unique_id").agg({
			"total_price": "sum",
			"total_freight": "sum",
			"order_id": "nunique",
			"order_purchase_timestamp": ["min", "max"],
		})

		# Flatten the MultiIndex columns
		customer_summary.columns = [
			"total_spending",
			"total_freight",
			"total_orders",
			"first_order_date",
			"last_order_date",
		]

		# Store the customer_summary_df in the instance
		self.customer_summary_df = customer_summary
		return customer_summary

	def get_sales_pivot(self, time_period: str = "year") -> pd.DataFrame:
		"""
		Create a pivot table showing total sales by product category over
		time.

		Parameters
		----------
		time_period : str, optional
		    Time aggregation level: 'year' or 'month', by default 'year'.

		Returns
		-------
		pd.DataFrame
		    Pivot table with product categories as rows and time periods as
		    columns.

		Raises
		------
		ValueError
		    If master_df has not been built yet or invalid time_period
		    specified.
		"""
		if self.master_df is None:
			raise ValueError("Master table has not been built yet.")

		if time_period not in ["year", "month"]:
			raise ValueError("time_period must be 'year' or 'month'")

		df = self.master_df.copy()
		df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])

		if time_period == "year":
			df["period"] = df["order_purchase_timestamp"].dt.year
		else:
			df["period"] = df["order_purchase_timestamp"].dt.to_period("M")

		pivot = df.pivot_table(
			values="total_price",
			index="category_english",
			columns="period",
			aggfunc="sum",
			fill_value=0,
		)

		return pivot

	def get_top_categories(self, n: int = 10) -> pd.DataFrame:
		"""
		Get the top N product categories by total revenue.

		Parameters
		----------
		n : int, optional
		    Number of top categories to return, by default 10.

		Returns
		-------
		pd.DataFrame
		    DataFrame with top categories and their total revenue.

		Raises
		------
		ValueError
		    If master_df has not been built yet.
		"""
		if self.master_df is None:
			raise ValueError("Master table has not been built yet.")

		category_revenue = (
			self.master_df.groupby("category_english")["total_price"]
			.sum()
			.sort_values(ascending=False)
			.head(n)
			.reset_index()
		)
		category_revenue.columns = ["category", "total_revenue"]

		return category_revenue

	def get_data_info(self) -> dict[str, dict[str, int | str]]:
		"""
		Get summary information about loaded datasets.

		Returns
		-------
		dict
		    Dictionary containing dataset names, shapes, and status.
		"""
		info = {}
		for name, df in self.data_dict.items():
			info[name] = {
				"rows": df.shape[0],
				"columns": df.shape[1],
				"status": "loaded",
			}
		return info
