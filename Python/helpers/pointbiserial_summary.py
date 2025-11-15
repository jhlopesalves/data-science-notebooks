import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr


def pointbiserial_summary(
	data: pd.DataFrame,
	indicator: str,
	targets,
	*,
	alpha: float = 0.05,
	dropna: bool = True,
	strength_thresholds: tuple[float, float, float] = (0.1, 0.3, 0.5),
) -> pd.DataFrame:
	"""
	Compute point-biserial correlations between a binary indicator and one or more targets.

	The point-biserial correlation is a special case of Pearson correlation that measures
	the relationship between a continuous variable and a binary variable (0/1). It's
	mathematically equivalent to computing Pearson correlation where one variable is
	dichotomous.

	Parameters
	----------
	data : pd.DataFrame
		Data containing the indicator and target columns.
	indicator : str
		Name of the binary indicator column (0/1 or bool). NaNs are allowed.
	targets : list[str] or tuple[str] or np.ndarray
		Names of numeric / ordinal target columns.
	alpha : float, optional
		Significance level for p-values. Default is 0.05.
	dropna : bool, optional
		If True (default), drop rows with NaN in either the indicator or target.
		If False, keep rows where the indicator and target are both non-missing.
	strength_thresholds : tuple[float, float, float], optional
		Effect size cut-offs for (weak, moderate, strong), in |r|.
		Default is (0.1, 0.3, 0.5).

	Returns
	-------
	pd.DataFrame
		One row per target with:
		- target: Name of the target variable
		- n: Sample size used in correlation calculation
		- r_pb: Point-biserial correlation coefficient (-1 to 1)
		- p_value: Two-tailed p-value for significance test
		- strength: Qualitative strength description based on thresholds
		- direction: Direction of relationship (positive/negative/none)
		- relationship: Combined strength and direction description
		- is_significant: Boolean indicating if p < alpha
		- alpha: Significance level used
		- note: Any warnings or notes about the calculation

	Notes
	-----
	Theory:
	- The point-biserial correlation measures the strength and direction of the
	  linear relationship between a binary variable and a continuous variable.
	- Formula: r_pb = (M1 - M0) / s * √(p(1-p)) where:
		* M1, M0 are means of the continuous variable for the two binary groups
		* s is the standard deviation of the continuous variable
		* p is the proportion of cases in the "1" group
	- Interpretation follows Cohen's guidelines for effect sizes:
		* |r| < 0.1: Very weak
		* 0.1 ≤ |r| < 0.3: Weak
		* 0.3 ≤ |r| < 0.5: Moderate
		* |r| ≥ 0.5: Strong

	Assumptions:
	1. The indicator variable is truly binary (only two distinct values)
	2. The target variable is continuous or ordinal
	3. Linear relationship between the variables
	4. Homoscedasticity (equal variances across groups)

	Examples
	--------
	>>> import pandas as pd
	>>> import numpy as np
	>>>
	>>> # Create sample data with binary indicator and continuous targets
	>>> np.random.seed(42)
	>>> data = pd.DataFrame({
	...     'treatment': np.random.choice([0, 1], 100),
	...     'score': np.random.normal(50, 10, 100) + 5 * np.random.choice([0, 1], 100),
	...     'age': np.random.normal(35, 5, 100),
	...     'income': np.random.normal(50000, 10000, 100)
	... })
	>>>
	>>> # Compute point-biserial correlations
	>>> results = pointbiserial_summary(
	...     data=data,
	...     indicator='treatment',
	...     targets=['score', 'age', 'income']
	... )
	>>>
	>>> # Display results
	>>> print(results[['target', 'r_pb', 'p_value', 'strength', 'is_significant']])

	Common Use Cases:
	- A/B testing: Treatment vs. control group differences
	- Medical studies: Disease presence vs. continuous outcomes
	- Education research: Program participation vs. test scores
	- Business analytics: Customer segment vs. spending behavior

	See Also
	--------
	scipy.stats.pointbiserialr : Underlying correlation function
	pandas.DataFrame.corr : General correlation matrix
	scipy.stats.pearsonr : Pearson correlation for continuous variables
	"""
	if isinstance(targets, (str, bytes)):
		raise TypeError(
			"`targets` must be an iterable of column names, not a single string."
		)

	targets = list(targets)

	if indicator not in data.columns:
		raise KeyError(f"Indicator column '{indicator}' not found in DataFrame.")

	ind_series = data[indicator]

	# Basic type / binary checks on the full column (ignoring NaNs)
	non_na_indicator = ind_series.dropna()

	if non_na_indicator.nunique() < 2:
		raise ValueError(
			f"Indicator '{indicator}' has fewer than 2 distinct non-missing values; "
			"cannot compute point-biserial correlation."
		)

	# Allow numeric or boolean. Anything else is suspicious.
	if not (
		np.issubdtype(non_na_indicator.dtype, np.number)
		or non_na_indicator.dtype == "bool"
	):
		raise TypeError(
			f"Indicator '{indicator}' must be numeric or boolean for point-biserial correlation. "
			f"Got dtype {non_na_indicator.dtype!r}."
		)

	weak, moderate, strong = strength_thresholds

	rows = []

	for t in targets:
		if t not in data.columns:
			rows.append({
				"target": t,
				"n": 0,
				"r_pb": np.nan,
				"p_value": np.nan,
				"strength": None,
				"direction": None,
				"relationship": None,
				"is_significant": None,
				"alpha": alpha,
				"note": f"target column '{t}' not found",
			})
			continue

		temp = data[[indicator, t]]

		if dropna:
			temp = temp.dropna()
		else:
			temp = temp[temp[indicator].notna() & temp[t].notna()]

		n = len(temp)

		if n == 0:
			rows.append({
				"target": t,
				"n": 0,
				"r_pb": np.nan,
				"p_value": np.nan,
				"strength": None,
				"direction": None,
				"relationship": None,
				"is_significant": None,
				"alpha": alpha,
				"note": "no complete cases",
			})
			continue

		ind_unique = temp[indicator].nunique()
		t_unique = temp[t].nunique()

		if ind_unique != 2 or t_unique < 2:
			if ind_unique != 2:
				note = "indicator not binary in subset"
			else:
				note = "insufficient variability in target"

			rows.append({
				"target": t,
				"n": n,
				"r_pb": np.nan,
				"p_value": np.nan,
				"strength": None,
				"direction": None,
				"relationship": None,
				"is_significant": None,
				"alpha": alpha,
				"note": note,
			})
			continue

		r, p = pointbiserialr(temp[indicator], temp[t])

		if np.isnan(r):
			strength = None
			direction = None
			relationship = None
			is_significant = None
			note = "correlation undefined (NaN)"
		else:
			ar = abs(r)
			if ar >= strong:
				strength = "strong"
			elif ar >= moderate:
				strength = "moderate"
			elif ar >= weak:
				strength = "weak"
			else:
				strength = "very weak"

			if np.isclose(r, 0):
				direction = "none"
			elif r < 0:
				direction = "negative"
			else:
				direction = "positive"

			relationship = f"{strength} {direction}".strip()
			is_significant = bool(p < alpha) if not np.isnan(p) else None
			note = None

		rows.append({
			"target": t,
			"n": n,
			"r_pb": r,
			"p_value": p,
			"strength": strength,
			"direction": direction,
			"relationship": relationship,
			"is_significant": is_significant,
			"alpha": alpha,
			"note": note,
		})

	return pd.DataFrame(rows)
