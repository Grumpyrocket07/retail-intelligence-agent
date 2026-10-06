"""Turns churn clues into model inputs. Fitted on the training rows only, then reused unchanged for validation,
test and the live API. Lives in its own module so saved copies can be loaded from anywhere."""
import numpy as np
import pandas as pd

from ml.churn_dataset import FEATURES

GAP_COLS = ["avg_gap_days", "recency_over_avg_gap"]  # empty for customers with a single order day


class Prep:
    """Fitted on the TRAIN rows only, then applied unchanged to validation and test."""

    def fit(self, df: pd.DataFrame) -> "Prep":
        self.medians = df[GAP_COLS].median()
        x = self._log(df)
        self.mean, self.std = x.mean(), x.std().replace(0, 1)
        return self

    def tree_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Raw clues + a 'no_repeat' flag; gradient boosting copes with the empty values itself."""
        out = df[FEATURES].copy()
        out["no_repeat"] = out["avg_gap_days"].isna().astype(int)
        return out

    def _log(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df[FEATURES].copy()
        x[GAP_COLS] = x[GAP_COLS].fillna(self.medians)
        x["no_repeat"] = df["avg_gap_days"].isna().astype(int)
        skewed = [c for c in x.columns if c not in ("is_uk", "no_repeat")]
        x[skewed] = np.log1p(x[skewed].clip(lower=0))  # squash huge values (one customer spent £450k)
        return x

    def scaled(self, df: pd.DataFrame) -> np.ndarray:
        x = self._log(df)
        return ((x - self.mean) / self.std).to_numpy(dtype=np.float32)


# --------------------------------------------------------------------------- models
