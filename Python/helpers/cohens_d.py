import pandas as pd


def cohens_d(data: pd.DataFrame, feature: str, group_col: str) -> float:
	"""
	Calculate Cohen's d effect size for a numeric feature between two groups.

	Cohen's d measures the standardized difference between means of two groups,
	using the pooled standard deviation as the denominator.

	Parameters
	----------
	data : pd.DataFrame
		DataFrame containing the feature and grouping column.
	feature : str
		Name of the numeric feature to analyze.
	group_col : str
		Name of the binary grouping column (0 and 1).

	Returns
	-------
	float
		Cohen's d effect size value.

	Notes
	-----
	Cohen's d is calculated using the formula:
	d = (mean_group0 - mean_group1) / pooled_standard_deviation

	Where the pooled standard deviation is:
	sp = sqrt(((n0-1)*std0² + (n1-1)*std1²) / (n0 + n1 - 2))

	Interpretation guidelines:
	- d ≈ 0.2: Small effect size
	- d ≈ 0.5: Medium effect size
	- d ≈ 0.8: Large effect size

	Examples
	--------
	>>> cohen_d = calculate_cohens_d(X, "OverallQual")
	>>> print(f"Cohen's d: {cohen_d:.3f}")
	Cohen's d: 1.185
	"""
	group_stats = data.groupby(group_col)[feature].agg(["mean", "std", "count"])

	mean_0, std_0, n_0 = group_stats.loc[0]
	mean_1, std_1, n_1 = group_stats.loc[1]

	# Calculate pooled standard deviation
	sp = (((n_0 - 1) * std_0**2 + (n_1 - 1) * std_1**2) / (n_0 + n_1 - 2)) ** 0.5

	# Calculate Cohen's d
	cohen_d = (mean_0 - mean_1) / sp

	return cohen_d
