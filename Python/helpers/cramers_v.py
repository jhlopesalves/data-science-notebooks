from scipy.stats import chi2_contingency


def cramers_v(series_feature, series_garage_missing):
	"""
	Calculate Cramér's V statistic for association between two categorical variables.

	Parameters
	----------
	series_feature : pd.Series
		First categorical variable (feature to test).
	series_garage_missing : pd.Series
		Second categorical variable (GarageMissing indicator).

	Returns
	-------
	tuple
		(v, pvalue) where:
		- v: Cramér's V value, ranging from 0 (no association) to 1 (perfect association)
		- pvalue: p-value from chi-square test of independence

	Notes
	-----
	Cramér's V is derived from the chi-square statistic and normalized to measure the strength
	of association between two nominal variables. The formula is:

	.. math:: V = \\sqrt{\\frac{\\chi^2}{n \\times \\min(r-1, c-1)}}

	where:
	- :math:`\\chi^2` is the chi-square statistic from the contingency table,
	- :math:`n` is the total number of observations,
	- :math:`r` is the number of rows in the contingency table,
	- :math:`c` is the number of columns in the contingency table.

	Interpretation:
	- V < 0.1: Weak association
	- 0.1 ≤ V < 0.3: Moderate association
	- V ≥ 0.3: Strong association

	This normalization ensures V is bounded between 0 and 1, making it comparable across tables
	of different sizes.
	"""
	contingency = pd.crosstab(series_feature, series_garage_missing)
	chi2, pvalue, dof, expected = chi2_contingency(contingency)
	n = contingency.to_numpy().sum()
	phi2 = chi2 / n
	r, k = contingency.shape
	v = np.sqrt(phi2 / min(r - 1, k - 1))
	return v, pvalue
