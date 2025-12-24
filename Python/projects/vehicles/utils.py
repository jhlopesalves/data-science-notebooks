import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


# Linear relationship: each year of age has consistent depreciation impact
# Reference year is dynamically determined during fit to ensure data consistency
class YearToAgeTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        reference_year: int | None = None,
        year_col: str = "year",
        output_col: str = "car_age",
        drop_year: bool = True,
        min_age: int | None = 0,
        max_age: int | None = 50,
    ):
        self.reference_year = reference_year
        self.year_col = year_col
        self.output_col = output_col
        self.drop_year = drop_year
        self.min_age = min_age
        self.max_age = max_age

    # Fit method: validates input structure and determines the temporal baseline
    def fit(self, X, y=None):
        # Ensure input is DataFrame (required for column-based operations)
        if not hasattr(X, "columns"):
            raise TypeError("YearToAgeTransformer expects a pandas DataFrame.")
        # Verify required column exists before transformation
        if self.year_col not in X.columns:
            raise ValueError(f"Column '{self.year_col}' not found in X.")

        # Set reference year: use provided value or infer max from training data
        # This prevents data leakage and ensures ages remain relative to the dataset
        if self.reference_year is None:
            self.ref_year_ = int(X[self.year_col].max())
        else:
            self.ref_year_ = self.reference_year

        # Store original feature names and count for scikit-learn pipeline compatibility
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True  # Flag for check_is_fitted validation
        return self

    # Transform method: performs the actual year-to-age conversion
    def transform(self, X):
        # Verify transformer was fitted before transform (sklearn protocol)
        check_is_fitted(self, attributes="is_fitted_")
        X = X.copy()  # Avoid modifying original DataFrame

        # Convert to numeric, coercing invalid values to NaN (handles mixed types)
        year = pd.to_numeric(X[self.year_col], errors="coerce")
        car_age = self.ref_year_ - year

        # Apply lower bound clipping (prevents negative ages from future years or data errors)
        if self.min_age is not None:
            car_age = car_age.clip(lower=self.min_age)
        # Apply upper bound clipping (prevents extreme ages from vintage/antique cars)
        if self.max_age is not None:
            car_age = car_age.clip(upper=self.max_age)

        X[self.output_col] = car_age

        # Optionally remove original year column to reduce feature redundancy
        if self.drop_year:
            X = X.drop(columns=[self.year_col])

        return X


# Custom transformer that extracts brand/manufacturer from full car name
class BrandExtractor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        name_col: str = "name",
        output_col: str = "brand",
        drop_name: bool = True,
        min_count: int | None = 100,
        other_value: str = "OTHER",
    ):
        self.name_col = name_col
        self.output_col = output_col
        self.drop_name = drop_name
        self.min_count = min_count
        self.other_value = other_value

    # Internal helper to perform tokenisation and case normalisation
    def _extract_brand_series(self, X: pd.DataFrame) -> pd.Series:
        series = (
            X[self.name_col]
            .astype("string")
            .str.strip()
            .str.upper()  # Ensures "Maruti" and "maruti" map to the same category
            .str.replace(r"\s+", " ", regex=True)
        )

        # Extract first word; preserves NaNs to allow standard imputation downstream
        return series.str.split(" ").str[0]

    # Fit method: identifies high-frequency brands to keep, strictly from training data
    def fit(self, X, y=None):
        if not hasattr(X, "columns"):
            raise TypeError("BrandExtractor expects a pandas DataFrame.")
        if self.name_col not in X.columns:
            raise ValueError(f"Column '{self.name_col}' not found in X.")

        brand_series = self._extract_brand_series(X)

        # Build whitelist of brands based on frequency threshold
        if self.min_count is None or self.min_count <= 1:
            self.kept_brands_ = None
        else:
            counts = brand_series.value_counts()
            kept = counts[counts >= self.min_count].index
            self.kept_brands_ = set(kept.tolist())

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True
        return self

    # Transform method: extracts brand and maps rare/unseen values to a catch-all category
    def transform(self, X):
        check_is_fitted(self, attributes="is_fitted_")
        X_df = X.copy()

        brand = self._extract_brand_series(X_df)

        # Group rare or unseen brands -> OTHER (preserves NaN values for imputer)
        if self.kept_brands_ is not None:
            mask = ~brand.isin(self.kept_brands_) & brand.notna()
            brand = brand.mask(mask, self.other_value)

        X_df[self.output_col] = brand

        if self.drop_name:
            X_df = X_df.drop(columns=[self.name_col])

        return X_df


def squeeze_dataframe(X):
    """Helper to flatten 2D DataFrame/array to 1D Series."""
    return X.squeeze()


def densify_matrix(X):
    """Helper to convert Sparse Matrix to Dense Array (safely)."""
    if hasattr(X, "toarray"):
        return X.toarray()
    return X
