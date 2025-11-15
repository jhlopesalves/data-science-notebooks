from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def cramers_v_summary(
	data: pd.DataFrame,
	row: str,
	cols,
	*,
	alpha: float = 0.05,
	correction: bool | None = True,
	dropna: bool = True,
	strength_thresholds: tuple[float, float, float] = (0.1, 0.3, 0.5),
) -> pd.DataFrame:
	"""Compute Cramér's V association between a categorical row variable and one or more columns.

	This is a convenience wrapper around :func:`scipy.stats.chi2_contingency` that returns a
	compact, analysis-ready summary for each target column, similar in spirit to
	:func:`pointbiserial_summary`.

	Parameters
	----------
	data : pd.DataFrame
		Data containing the row and column variables.
	row : str
		Name of the row variable (typically the "feature" or predictor).
	cols : list[str] or tuple[str] or np.ndarray
		Names of categorical target columns to test against ``row``.
	alpha : float, optional
		Significance level for p-values. Default is 0.05.
	correction : bool or None, optional
		Whether to apply Yates' continuity correction for 2x2 tables.
		Forwarded to :func:`scipy.stats.chi2_contingency`. If ``None``, SciPy's default is used.
	dropna : bool, optional
		If True (default), drop rows with NaN in either the row or column variable
		before building the contingency table. If False, keep rows where both are non-missing.
	strength_thresholds : tuple[float, float, float], optional
		Effect size cut-offs for (weak, moderate, strong) association, in |V|.
		Default is (0.1, 0.3, 0.5).

	Returns
	-------
	pd.DataFrame
		One row per column in ``cols`` with:

		- target: Name of the target column
		- n: Total sample size used in the contingency table
		- chi2: Chi-square statistic
		- dof: Degrees of freedom
		- p_value: p-value from chi-square test of independence
		- v: Cramér's V association measure (0 to 1)
		- strength: Qualitative strength description based on thresholds
		- is_significant: Boolean indicating if ``p_value < alpha``
		- alpha: Significance level used
		- note: Any warnings or notes about the calculation (e.g. sparse table, single level)

	Notes
	-----
	Theory
	^^^^^^
	Cramér's V is a normalized measure of association between two categorical variables,
	derived from the chi-square statistic of the contingency table. For a contingency table
	with ``r`` rows and ``c`` columns, total sample size ``n`` and chi-square statistic
	:math:`\\chi^2`, Cramér's V is defined as:

	.. math::

		V = \\sqrt{\\frac{\\chi^2}{n \\times \\min(r-1, c-1)}}

	Key properties:
	- ``0`` means no association (independence)
	- ``1`` means perfect association
	- The normalization by ``min(r-1, c-1)`` ensures comparability across tables
	  of different sizes.

	Common interpretation (Cohen-style guidelines, very rough):
	- ``V < 0.1``: very weak or negligible association
	- ``0.1 ≤ V < 0.3``: weak
	- ``0.3 ≤ V < 0.5``: moderate
	- ``V ≥ 0.5``: strong

	Assumptions
	^^^^^^^^^^^
	1. Observations are independent
	2. Variables are measured on a nominal (categorical) scale
	3. Expected cell counts are not too small (rule-of-thumb: most expected cells > 5)

	Examples
	--------
	>>> import pandas as pd
	>>> import numpy as np
	>>> from helpers.cramers_v import cramers_v_summary
	>>>
	>>> # Toy example: relationship between garage presence and house style
	>>> np.random.seed(42)
	>>> data = pd.DataFrame({
	...     "GarageMissing": np.random.choice(["Yes", "No"], size=200, p=[0.3, 0.7]),
	...     "HouseStyle": np.random.choice(["1Story", "2Story", "1.5Fin"], size=200),
	...     "Neighborhood": np.random.choice(["CollgCr", "Veenker", "Crawfor"], size=200),
	... })
	>>>
	>>> results = cramers_v_summary(
	...     data=data,
	...     row="GarageMissing",
	...     cols=["HouseStyle", "Neighborhood"],
	... )
	>>>
	>>> print(results[["target", "v", "p_value", "strength", "is_significant"]])

	Typical use cases
	-----------------
	- Feature selection for categorical predictors
	- Exploring dependence between survey items (Likert scales)
	- Association between demographic variables (e.g. gender vs. preference)
	- Assessing association in contingency tables as part of chi-square tests

	See Also
	--------
	scipy.stats.chi2_contingency : Underlying chi-square test of independence
	pointbiserial_summary : Similar summary API for continuous/binary relationships
	"""
	if isinstance(cols, (str, bytes)):
		raise TypeError(
			"`cols` must be an iterable of column names, not a single string."
		)

	cols = list(cols)

	if row not in data.columns:
		raise KeyError(f"Row column '{row}' not found in DataFrame.")

	row_series = data[row]
	row_non_na = row_series.dropna()

	if row_non_na.nunique() < 2:
		raise ValueError(
			f"Row variable '{row}' has fewer than 2 distinct non-missing values; "
			"cannot compute Cramér's V."
		)

	weak, moderate, strong = strength_thresholds
	rows = []

	for col in cols:
		if col not in data.columns:
			rows.append({
				"target": col,
				"n": 0,
				"chi2": np.nan,
				"dof": np.nan,
				"p_value": np.nan,
				"v": np.nan,
				"strength": None,
				"is_significant": None,
				"alpha": alpha,
				"note": f"target column '{col}' not found",
			})
			continue

		temp = data[[row, col]]

		if dropna:
			temp = temp.dropna()
		else:
			temp = temp[temp[row].notna() & temp[col].notna()]

		if temp.empty:
			rows.append({
				"target": col,
				"n": 0,
				"chi2": np.nan,
				"dof": np.nan,
				"p_value": np.nan,
				"v": np.nan,
				"strength": None,
				"is_significant": None,
				"alpha": alpha,
				"note": "no complete cases",
			})
			continue

		contingency = pd.crosstab(temp[row], temp[col])
		n = contingency.to_numpy().sum()

		# If any dimension collapses to a single level, V is undefined
		if contingency.shape[0] < 2 or contingency.shape[1] < 2:
			rows.append({
				"target": col,
				"n": n,
				"chi2": np.nan,
				"dof": np.nan,
				"p_value": np.nan,
				"v": np.nan,
				"strength": None,
				"is_significant": None,
				"alpha": alpha,
				"note": "insufficient levels in row/column variable",
			})
			continue

		# chi2_contingency can operate on numpy arrays directly
		chi2, p_value, dof, expected = chi2_contingency(
			contingency.to_numpy(), correction=correction
		)

		phi2 = chi2 / n if n > 0 else np.nan
		r_dim, c_dim = contingency.shape
		denom = min(r_dim - 1, c_dim - 1)

		if denom <= 0 or np.isnan(phi2):
			v = np.nan
		else:
			v = float(np.sqrt(phi2 / denom))

		# Strength categorization
		if np.isnan(v):
			strength = None
			is_significant = None
			note = "association undefined (NaN)"
		else:
			av = abs(v)
			if av >= strong:
				strength = "strong"
			elif av >= moderate:
				strength = "moderate"
			elif av >= weak:
				strength = "weak"
			else:
				strength = "very weak"

			is_significant = bool(p_value < alpha) if not np.isnan(p_value) else None
			note = None

		# Heuristic note about sparsity
		if expected.size > 0 and np.any(expected < 5):
			heart_note = "some expected cell counts < 5; chi-square approximation may be unreliable"
			note = heart_note if note is None else f"{note}; {heart_note}"

		rows.append({
			"target": col,
			"n": int(n),
			"chi2": float(chi2),
			"dof": int(dof),
			"p_value": float(p_value),
			"v": v,
			"strength": strength,
			"is_significant": is_significant,
			"alpha": alpha,
			"note": note,
		})

	return pd.DataFrame(rows)


def cramers_v(series_feature, series_garage_missing):
	"""Backward-compatible thin wrapper returning ``(v, p_value)``.

	This function mirrors the original helper API used in some notebooks but delegates
	all computation to :func:`cramers_v_summary`. For new code, prefer calling
	:func:`cramers_v_summary` directly for a richer, tabular output.
	"""
	data = pd.DataFrame({
		"feature": series_feature,
		"garage_missing": series_garage_missing,
	})
	res = cramers_v_summary(data=data, row="feature", cols=["garage_missing"])
	row = res.iloc[0]
	return row["v"], row["p_value"]
