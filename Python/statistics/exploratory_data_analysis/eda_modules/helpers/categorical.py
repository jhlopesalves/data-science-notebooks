from typing import Dict, List, Optional

import pandas as pd


def get_categorical_columns(df: pd.DataFrame) -> List[str]:
	"""
	Identify categorical columns in a DataFrame based on data types.

	Parameters
	----------
	df : pd.DataFrame
		Input DataFrame to analyze

	Returns
	-------
	list
		List of column names that are likely categorical (object, category, or bool types)

	Examples
	--------
	>>> cat_cols = get_categorical_columns(df)
	>>> print(f"Found {len(cat_cols)} categorical columns")
	Found 43 categorical columns

	>>> cat_cols[:5]  # First 5 categorical columns
	['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour']
	"""
	mask = df.dtypes.astype(str).isin(["object", "category", "bool"])

	return df.columns[mask].tolist()


def make_overview(
	df: pd.DataFrame, columns: Optional[List[str]] = None
) -> pd.DataFrame:
	"""
	Build a comprehensive overview summary for categorical features.

	Provides key statistics including number of unique categories, most frequent
	category, missing data percentage, and distribution metrics.

	Parameters
	----------
	df : pd.DataFrame
		Input DataFrame containing categorical data
	columns : list, optional
		Specific columns to analyze. If None, analyzes all categorical columns

	Returns
	-------
	pd.DataFrame
		Overview table with columns:
		- n_unique: Number of distinct non-NaN categories
		- top_category: Most frequent label (NaN shown as 'NaN')
		- top_count: Count of the most frequent label
		- top_pct: Percentage of rows that the top label represents
		- missing_pct: Percentage of missing values

	Examples
	--------
	>>> overview = make_overview(df)
	>>> overview.head()

	>>> # Analyze specific columns only
	>>> neighborhood_overview = make_overview(ames_df, columns=['Neighborhood', 'MSZoning'])
	"""
	if columns is None:
		columns = get_categorical_columns(df)

	records = []
	for col in columns:
		s = df[col]
		total = len(s)

		counts = s.value_counts(dropna=False)
		top_label_raw = counts.index[0]
		top_label = "NaN" if pd.isna(top_label_raw) else str(top_label_raw)
		top_count = int(counts.iloc[0])
		top_pct = round(top_count / total * 100.0, 2)
		n_unique = int(s.nunique(dropna=True))
		missing_pct = round(s.isna().mean() * 100.0, 2)

		records.append({
			"feature": col,
			"n_unique": n_unique,
			"top_category": top_label,
			"top_count": top_count,
			"top_pct": top_pct,
			"missing_pct": missing_pct,
		})

	if not records:
		# Return empty DataFrame with proper columns if no categorical columns found
		return pd.DataFrame(
			columns=["n_unique", "top_category", "top_count", "top_pct", "missing_pct"]
		)

	overview = pd.DataFrame(records).set_index("feature")
	overview.sort_values("n_unique", ascending=False, inplace=True)

	return overview


def per_column_tables(
	df: pd.DataFrame, columns: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
	"""
	Generate detailed frequency tables for each categorical column.

	Creates a dictionary where keys are column names and values are DataFrames
	containing complete frequency distributions with cumulative percentages.

	Parameters
	----------
	df : pd.DataFrame
		Input DataFrame containing categorical data
	columns : list, optional
		Specific columns to analyze. If None, analyzes all categorical columns

	Returns
	-------
	dict
		Dictionary mapping column names to frequency tables with columns:
		- category: Category label
		- count: Frequency count
		- percent: Percentage of total
		- cum_percent: Cumulative percentage

	Examples
	--------
	>>> freq_tables = per_column_tables(df)
	>>> # Access frequency table for Neighborhood
	>>> neighborhood_table = freq_tables['Neighborhood']
	>>> print(neighborhood_table.head())

	>>> # Get tables for specific columns
	>>> selected_tables = per_column_tables(ames_df, ['SaleCondition', 'MSZoning'])
	"""
	if columns is None:
		columns = get_categorical_columns(df)

	result = {}

	for col in columns:
		s = df[col]
		counts = s.value_counts(dropna=False)
		total = counts.sum()

		freq_df = (
			counts.rename_axis("category")
			.reset_index(name="count")
			.assign(
				percent=lambda d: (d["count"] / total * 100).round(2),
			)
		)
		freq_df["cum_percent"] = freq_df["percent"].cumsum().round(2)
		freq_df["category"] = freq_df["category"].astype(str).replace("nan", "NaN")

		result[col] = freq_df

	return result
