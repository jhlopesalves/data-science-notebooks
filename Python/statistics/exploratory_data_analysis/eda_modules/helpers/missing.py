import pandas as pd


def missing_data_summary(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Return a comprehensive summary of missing data patterns for columns with missing values.

	This function provides a detailed analysis of missing values across columns
	that contain missing data in the input DataFrame. It calculates various statistics
	including counts, percentages, and data types to help identify patterns and
	prioritize data cleaning efforts.

	Parameters
	----------
	df : pd.DataFrame
		Input DataFrame to analyze for missing values.

	Returns
	-------
	pd.DataFrame
		A DataFrame containing the following columns for each column with missing values:
		- 'Data Type': The data type of each column (dtype)
		- 'Missing Count': Number of missing/NaN values in each column
		- 'Missing %': Percentage of missing values (rounded to 2 decimal places)
		- 'Non-Missing Count': Number of non-missing values in each column

		The summary is sorted in descending order by 'Missing %' to highlight
		columns with the most missing data first. Columns without missing values
		are excluded from the summary.

	Examples
	--------
	>>> import pandas as pd
	>>> import numpy as np
	>>>
	>>> # Create sample DataFrame with missing values
	>>> data = {
	...     'A': [1, 2, np.nan, 4, 5],
	...     'B': [np.nan, np.nan, 3, 4, 5],
	...     'C': [1, 2, 3, 4, 5],
	...     'D': [1, np.nan, np.nan, np.nan, np.nan]
	... }
	>>> df = pd.DataFrame(data)
	>>>
	>>> # Generate missing data summary
	>>> summary = missing_data_summary(df)
	>>> print(summary)
			Data Type  Missing Count  Missing %  Non-Missing Count
	D         int64              4       80.0                  1
	B       float64            2       40.0                  3
	A       float64            1       20.0                  4

	Notes
	-----
	- Missing values are identified using pandas' `isnull()` method
	- Percentage calculation uses total row count as denominator
	- Results are sorted to help prioritize data cleaning efforts
	- Only columns with missing values are included in the summary
	- Useful for initial data quality assessment and EDA workflows
	"""
	# Total number of rows
	total_rows = len(df)

	# Calculate missing count per column
	missing_count = df.isnull().sum()

	# Filter columns that have missing values
	columns_with_missing = missing_count[missing_count > 0].index

	# If no columns have missing values, return empty DataFrame with correct structure
	if len(columns_with_missing) == 0:
		return pd.DataFrame(
			columns=["Data Type", "Missing Count", "Missing %", "Non-Missing Count"]
		)

	# Calculate percentage of missing values for columns with missing data
	missing_percentage = (missing_count[columns_with_missing] / total_rows) * 100

	# Count non-missing values for columns with missing data
	non_missing_count = total_rows - missing_count[columns_with_missing]

	# Data types for columns with missing data
	column_dtypes = df[columns_with_missing].dtypes

	# Combine everything into a single summary table
	summary = pd.DataFrame({
		"Data Type": column_dtypes,
		"Missing Count": missing_count[columns_with_missing],
		"Missing %": missing_percentage.round(2),
		"Non-Missing Count": non_missing_count,
	})

	# Sort descending by missing percentage
	summary = summary.sort_values(by="Missing %", ascending=False)

	return summary
