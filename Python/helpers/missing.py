import pandas as pd


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
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
	- Missing values are identified using pandas' `isna()` method
	- Percentage calculation uses total row count as denominator
	- Results are sorted to help prioritize data cleaning efforts
	- Only columns with missing values are included in the summary
	- Useful for initial data quality assessment and EDA workflows
	"""
	# Total number of rows
	total_rows = len(df)

	# Vectorized missing counts for every column
	missing_count = df.isna().sum()

	# Keep only columns that actually contain missing values
	missing_count = missing_count[missing_count.gt(0)]

	# If no columns have missing values, return empty DataFrame with correct structure
	if missing_count.empty:
		return pd.DataFrame(
			columns=["Data Type", "Missing Count", "Missing %", "Non-Missing Count"]
		)

	# Build the summary using aligned Series to avoid copying column data
	summary = pd.DataFrame({
		"Data Type": df.dtypes.reindex(missing_count.index),
		"Missing Count": missing_count,
	})
	summary["Missing %"] = (summary["Missing Count"] / total_rows * 100).round(2)
	summary["Non-Missing Count"] = total_rows - summary["Missing Count"]

	# Sort descending by missing percentage
	return summary.sort_values(by="Missing %", ascending=False)
