import math
from typing import Callable, Dict, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_seaborn(
	df: pd.DataFrame,
	x: Optional[str] = None,
	y: Optional[str] = None,
	plot_type: str = "scatterplot",
	hue: Optional[str] = None,
	palette: Optional[Union[str, list]] = None,
	size: Optional[str] = None,
	title: str = "",
	xlabel: str = "",
	ylabel: str = "",
	figsize: Tuple[int, int] = (10, 8),
	grid: bool = True,
	despine: bool = True,
	rotation: int = 0,
	ylim: Optional[Tuple[float, float]] = None,
	xlim: Optional[Tuple[float, float]] = None,
	# Annotation controls
	annotate: bool = False,
	annotation_format: Union[
		str, Callable[[float, pd.Series], str]
	] = "auto",  # "auto" | "value" | "percent" | "value+percent" | callable
	annotation_position: Union[str, Callable[..., Dict]] = "auto",
	annotation_offset: float = 3.0,  # in points
	annotation_clip: bool = False,
	annotation_font_kwargs: Optional[Dict] = None,
	percent_reference: Literal["group_total", "column_total", "ymax"] = "group_total",
	annotation_scope: Literal["all", "last"] = "all",
	**kwargs,
) -> None:
	"""
	Generate a seaborn plot with professional styling and optional annotations.

	This function serves as a flexible wrapper around various seaborn plotting
	functions, providing a consistent interface for creating common
	visualizations. It includes sensible defaults for aesthetics and adds a
	powerful annotation layer for bar-like charts.

	Parameters
	----------
	df : pd.DataFrame
	    The DataFrame containing the data to be plotted.
	x : str, optional
	    The name of the column to be used for the x-axis. Default is None.
	y : str, optional
	    The name of the column to be used for the y-axis. Default is None.
	plot_type : str, default='scatterplot'
	    The type of seaborn plot to generate. Examples include 'scatterplot',
	    'lineplot', 'barplot', 'countplot', 'boxplot', 'violinplot', etc.
	hue : str, optional
	    Column name for color encoding. Default is None.
	palette : str or list, optional
	    Color palette for the plot. Can be a seaborn palette name or a list
	    of colors. Default is None.
	size : str, optional
	    Column name for size encoding (e.g., in a scatterplot). Default is None.
	title : str, default=''
	    The title of the plot.
	xlabel : str, default=''
	    The label for the x-axis.
	ylabel : str, default=''
	    The label for the y-axis.
	figsize : tuple of (int, int), default=(10, 8)
	    The size of the figure in inches (width, height).
	grid : bool, default=True
	    If True, a grid is added to the plot.
	despine : bool, default=True
	    If True, the top and right spines of the plot are removed.
	rotation : int, default=0
	    The rotation angle for x-axis tick labels.
	ylim : tuple of (float, float), optional
	    The limits for the y-axis. Default is None.
	xlim : tuple of (float, float), optional
	    The limits for the x-axis. Default is None.
	annotate : bool, default=False
	    If True, annotates bars on 'barplot' or 'countplot'.
	annotation_format : str or callable, default='auto'
	    Controls the format of the annotations.
	    - 'auto': Chooses format based on data (percent for [0,1], value+percent for [0,100], else value).
	    - 'value': Displays the numeric value of the bar.
	    - 'percent': Displays the percentage relative to `percent_reference`.
	    - 'value+percent': Shows "value (percent%)".
	    - callable: A function that takes (height, row) and returns a string.
	annotation_position : str or callable, default='auto'
	    Position of the annotation text.
	    - 'auto': Places text inside for small bars, outside for large bars.
	    - 'inside': Forces text inside the bar.
	    - 'outside': Forces text outside the bar.
	    - callable: A function that returns a dict of `ax.text` kwargs.
	annotation_offset : float, default=3.0
	    Offset of the annotation text from the bar, in points.
	annotation_clip : bool, default=False
	    If True, allows annotations to be drawn outside the plot area.
	annotation_font_kwargs : dict, optional
	    Additional keyword arguments for the annotation text (e.g., fontsize, weight).
	percent_reference : {'group_total', 'column_total', 'ymax'}, default='group_total'
	    Defines the denominator for percentage calculations.
	    - 'group_total': Percentage of the total within each x-category (and hue subgroup).
	    - 'column_total': Percentage of the total of the entire column.
	    - 'ymax': Percentage relative to the y-axis maximum.
	annotation_scope : {'all', 'last'}, default='all'
	    Determines which bars to annotate.
	    - 'all': Annotates all bars.
	    - 'last': Annotates only the last bar.
	**kwargs
	    Additional keyword arguments passed directly to the underlying seaborn
	    plotting function.

	Returns
	-------
	None
	    The function displays the plot but does not return any value.

	Examples
	--------
	>>> import pandas as pd
	>>> data = {
	...     'category': ['A', 'A', 'B', 'B', 'C', 'C'],
	...     'value': [10, 15, 7, 12, 5, 10],
	...     'group': ['X', 'Y', 'X', 'Y', 'X', 'Y']
	... }
	>>> df = pd.DataFrame(data)
	>>> plot_seaborn(
	...     df,
	...     x='category',
	...     y='value',
	...     hue='group',
	...     plot_type='barplot',
	...     title='Bar Plot of Values by Category and Group',
	...     xlabel='Category',
	...     ylabel='Value',
	...     annotate=True,
	...     annotation_format='value'
	... )
	"""
	fig, ax = plt.subplots(figsize=figsize)

	# Map plot type to seaborn function (raises AttributeError on typo, which is fine)
	plot_func = getattr(sns, plot_type)

	# Build call parameters lazily to avoid passing None to seaborn
	plot_params = {"data": df, "ax": ax}
	if palette is not None:
		plot_params["palette"] = palette
	if x is not None:
		plot_params["x"] = x
	if y is not None:
		plot_params["y"] = y
	if hue:
		plot_params["hue"] = hue
	if size:
		plot_params["size"] = size
	plot_params.update(kwargs)

	# Draw the plot
	plot_func(**plot_params)

	# Labels and title
	ax.set(title=title, xlabel=xlabel, ylabel=ylabel)

	# User axis limits (if provided)
	if ylim is not None:
		ax.set_ylim(ylim)
	if xlim is not None:
		ax.set_xlim(xlim)

	# Light, readable grid
	if grid:
		ax.grid(True, axis="y", linestyle="--", alpha=0.6)
		ax.set_axisbelow(True)

	if despine:
		sns.despine(ax=ax, top=True, right=True)

	if rotation:
		plt.xticks(rotation=rotation)

	# Annotation pathway (bar-like plots only)
	if annotate and plot_type in {"barplot", "countplot"}:
		_annotate_bars(
			ax=ax,
			df=df,
			x=x,
			y=y,
			hue=hue,
			annotation_format=annotation_format,
			annotation_position=annotation_position,
			annotation_offset=annotation_offset,
			annotation_clip=annotation_clip,
			annotation_font_kwargs=annotation_font_kwargs
			or {
				"fontsize": 11,
				"weight": "bold",
			},
			percent_reference=percent_reference,
			scope=annotation_scope,
		)

	plt.tight_layout()
	plt.show()


def _annotate_bars(
	ax: plt.Axes,
	df: pd.DataFrame,
	x: Optional[str],
	y: Optional[str],
	hue: Optional[str],
	annotation_format: Union[str, Callable[[float, pd.Series], str]],
	annotation_position: Union[str, Callable[..., Dict]],
	annotation_offset: float,
	annotation_clip: bool,
	annotation_font_kwargs: Dict,
	percent_reference: str,
	scope: str,
) -> None:
	"""
	Add text annotations to bars in a bar-like plot.

	This internal helper function iterates through the patches (bars) of a
	matplotlib Axes object, calculates the appropriate annotation text based
	on the specified format, and places it on the plot.

	Parameters
	----------
	ax : plt.Axes
	    The matplotlib Axes object containing the plot.
	df : pd.DataFrame
	    The source DataFrame.
	x : str, optional
	    The column for the x-axis.
	y : str, optional
	    The column for the y-axis.
	hue : str, optional
	    The column for color encoding.
	annotation_format : str or callable
	    The format string or function for the annotation text.
	annotation_position : str or callable
	    The positioning logic for the annotation.
	annotation_offset : float
	    The text offset in points.
	annotation_clip : bool
	    Whether to clip annotations at the axes boundaries.
	annotation_font_kwargs : dict
	    Font properties for the annotation text.
	percent_reference : str
	    The reference for percentage calculations.
	scope : str
	    The scope of bars to annotate ('all' or 'last').

	Returns
	-------
	None
	"""
	# Extract bars; seaborn draws one Rectangle per bar
	patches = [p for p in ax.patches if hasattr(p, "get_height")]
	if not patches:
		return

	# Determine orientation: vertical bars have non-trivial height
	vertical = True
	if patches:
		# If width is large and height is tiny for all, it's horizontal
		sample = patches[0]
		vertical = sample.get_height() >= sample.get_width()

	# Build percent denominators if needed
	denominators = None
	wants_percent = callable(annotation_format) or (
		isinstance(annotation_format, str)
		and ("percent" in annotation_format or annotation_format == "auto")
	)
	if wants_percent and percent_reference in {"group_total", "column_total"}:
		denominators = _compute_denominators(df, x, y, hue, percent_reference)

	# Convert an offset in points to data units via transforms for stable spacing
	def data_offset(d_points: float) -> float:
		if vertical:
			# Move in y-direction by given display points and invert to data units
			dy = (
				ax.transData.inverted().transform((
					0,
					ax.transData.transform((0, 0))[1] + d_points,
				))[1]
				- 0
			)
			# If the axis is log-scaled, a constant offset in data units is not
			# perceptually constant. Points -> data is still better than guessing.
			return dy
		else:
			dx = (
				ax.transData.inverted().transform((
					ax.transData.transform((0, 0))[0] + d_points,
					0,
				))[0]
				- 0
			)
			return dx

	# Decide per-bar label text
	def format_label(height: float, row_stub: Optional[pd.Series]) -> str:
		if callable(annotation_format):
			return annotation_format(height, row_stub)

		fmt = annotation_format  # type: ignore[assignment]
		if fmt == "auto":
			if y is None:
				fmt = "percent"
			else:
				# Heuristic: treat as percent if values plausibly represent rates
				if 0.0 <= height <= 1.0:
					fmt = "percent"
				elif 0.0 <= height <= 100.0:
					fmt = "value+percent"
				else:
					fmt = "value"

		if fmt == "value":
			return f"{height:.0f}" if height == round(height) else f"{height:.1f}"

		if fmt == "percent":
			pct = _compute_percent(height, row_stub, denominators)
			return f"{pct:.1f}%"

		if fmt == "value+percent":
			pct = _compute_percent(height, row_stub, denominators)
			vtxt = f"{height:.0f}" if height == round(height) else f"{height:.1f}"
			return f"{vtxt} ({pct:.1f}%)"

		# Fallback to value
		return f"{height:.1f}"

	# Estimate if we need extra headroom to avoid clipping labels
	pad_needed = False
	max_coord = -math.inf
	for idx, bar in enumerate(patches):
		if scope == "last" and idx != len(patches) - 1:
			continue
		h = bar.get_height() if vertical else bar.get_width()
		base = bar.get_y() if vertical else bar.get_x()
		coord = base + h
		max_coord = max(max_coord, coord)

	# Approximate one label's height (in data units) to add headroom
	pad = data_offset(annotation_offset + 8.0)  # include text height guess
	if vertical:
		top = ax.get_ylim()[1]
		if max_coord + pad > top:
			pad_needed = True
			ax.set_ylim(ax.get_ylim()[0], max_coord + pad)
	else:
		right = ax.get_xlim()[1]
		if max_coord + pad > right:
			pad_needed = True
			ax.set_xlim(ax.get_xlim()[0], max_coord + pad)

	# Place labels
	for idx, bar in enumerate(patches):
		if scope == "last" and idx != len(patches) - 1:
			continue

		# Current bar geometry
		bx = bar.get_x()
		by = bar.get_y()
		bw = bar.get_width()
		bh = bar.get_height()

		# Row stub for custom formatters: best-effort map from categorical coords
		row_stub = None
		if x is not None:
			try:
				# This is approximate: for grouped bars, seaborn positions are
				# dodged; we prefer providing category names rather than exact row.
				row_stub = pd.Series({
					"x": ax.get_xticklabels()[int(round(bx + bw / 2))].get_text()
					if vertical and ax.get_xticklabels()
					else None,
					"hue": None,
				})
			except Exception:
				row_stub = None

		height = bh if vertical else bw
		label = format_label(height, row_stub)

		# Choose position automatically when requested
		if callable(annotation_position):
			pos_kwargs = annotation_position(ax=ax, bar=bar, height=height)
		else:
			pos = annotation_position  # "auto" | "inside" | "outside"
			# Threshold to switch to inside when bars are short
			threshold = 0.08  # 8% of axis span
			axis_span = (
				(ax.get_ylim()[1] - ax.get_ylim()[0])
				if vertical
				else (ax.get_xlim()[1] - ax.get_xlim()[0])
			)
			small_bar = (height / axis_span) < threshold

			if pos == "inside" or (pos == "auto" and small_bar):
				# Inside near the top
				if vertical:
					tx = bx + bw / 2
					ty = by + height - data_offset(annotation_offset)
					ha, va = "center", "top"
				else:
					tx = bx + height - data_offset(annotation_offset)
					ty = by + bh / 2
					ha, va = "right", "center"
			else:
				# Outside above/right of the bar
				if vertical:
					tx = bx + bw / 2
					ty = by + height + data_offset(annotation_offset)
					ha, va = "center", "bottom"
				else:
					tx = bx + height + data_offset(annotation_offset)
					ty = by + bh / 2
					ha, va = "left", "center"

			pos_kwargs = dict(x=tx, y=ty, ha=ha, va=va)

		# Contrast text colour against bar face colour when drawing inside
		color = (
			_contrast_colour(bar)
			if pos_kwargs.get("va") in {"top", "center"}
			and (
				(vertical and pos_kwargs["y"] < by + height)
				or (not vertical and pos_kwargs["x"] < bx + height)
			)
			else None
		)

		ax.text(
			pos_kwargs["x"],
			pos_kwargs["y"],
			label,
			ha=pos_kwargs.get("ha", "center"),
			va=pos_kwargs.get("va", "bottom"),
			clip_on=annotation_clip,
			color=color,
			**annotation_font_kwargs,
		)


def _compute_denominators(
	df: pd.DataFrame,
	x: Optional[str],
	y: Optional[str],
	hue: Optional[str],
	reference: str,
) -> pd.DataFrame:
	"""
	Compute denominators for percentage calculations.

	This function calculates the total values needed to compute percentages
	for annotations, based on the specified grouping and reference.

	Parameters
	----------
	df : pd.DataFrame
	    The source DataFrame.
	x : str, optional
	    The column for the x-axis.
	y : str, optional
	    The column for the y-axis. For 'countplot', this is None.
	hue : str, optional
	    The column for color encoding.
	reference : str
	    The reference for percentage calculation ('group_total' or 'column_total').

	Returns
	-------
	pd.DataFrame
	    A DataFrame containing the denominator values for each group.
	"""
	# Remove duplicates: if hue == x, only use x
	group_cols = []
	if x is not None:
		group_cols.append(x)
	if hue is not None and hue != x:
		group_cols.append(hue)

	if not group_cols:
		# No grouping columns, compute total
		if y is None:
			total = len(df)
		else:
			total = df[y].sum()
		return pd.DataFrame({"denom": [total]})

	if y is None:
		grouped = df.groupby(group_cols).size()
		denom = grouped.rename("denom").reset_index()
	else:
		grouped = df.groupby(group_cols)[y].sum()
		denom = grouped.rename("denom").reset_index()

	if reference == "column_total":
		total = denom["denom"].sum()
		denom["denom"] = total
	# For "group_total", values already grouped by x[/hue].
	return denom


def _compute_percent(
	value: float,
	row_stub: Optional[pd.Series],
	denominators: Optional[pd.DataFrame],
) -> float:
	"""
	Calculate a percentage value for an annotation.

	This helper safely computes a percentage, falling back to the raw value
	if a proper denominator cannot be determined.

	Parameters
	----------
	value : float
	    The numerator value (the height of the bar).
	row_stub : pd.Series, optional
	    A series containing approximate row information (e.g., x-value).
	denominators : pd.DataFrame, optional
	    A DataFrame of denominator values.

	Returns
	-------
	float
	    The calculated percentage, or the original value on failure.
	"""
	if denominators is None or denominators.empty:
		return float(value)

	denom_value = None
	if "denom" in denominators.columns and len(denominators) == 1:
		denom_value = float(denominators["denom"].iloc[0])

	if denom_value is None and row_stub is not None:
		# Attempt to match by x (and later hue if provided)
		try:
			key_cols = [c for c in denominators.columns if c != "denom"]
			# Basic match on first key column, else fall back to total if present
			if key_cols:
				# This is approximate; label text will still be informative.
				matches = denominators
				for col in key_cols:
					if row_stub.get(col) is not None:
						matches = matches[matches[col] == row_stub[col]]
				if not matches.empty:
					denom_value = float(matches["denom"].sum())
		except Exception:
			denom_value = None

	if denom_value is None or denom_value == 0:
		return float(value)

	return (value / denom_value) * 100.0


def _contrast_colour(bar) -> str:
	"""
	Select a contrasting color (black or white) for text on a colored bar.

	This function uses a simple luminance calculation to determine whether
	black or white text will be more readable against the bar's face color.

	Parameters
	----------
	bar : matplotlib.patches.Patch
	    The bar patch from which to get the face color.

	Returns
	-------
	str
	    'black' or 'white', representing the chosen text color.
	"""
	try:
		r, g, b, _ = bar.get_facecolor()
		# Relative luminance approximation
		lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
		return "black" if lum > 0.6 else "white"
	except Exception:
		return "white"
