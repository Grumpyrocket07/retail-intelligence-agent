"""The two services behind the API: churn scoring and sentiment scoring."""
import json
import logging
from pathlib import Path
from typing import Callable, Optional

import joblib
import numpy as np
import pandas as pd

from api import settings
from ml.churn_dataset import build_features

log = logging.getLogger("api.services")

CHURN_FILES = {
    "logistic_regression": "churn_logreg.joblib",
    "gradient_boosting": "churn_gb.joblib",
    "pytorch_mlp": "churn_mlp.pt",
}


# --------------------------------------------------------------------------- sentiment
def sentiment_label(p: float, low: float = settings.UNCERTAIN_LOW, high: float = settings.UNCERTAIN_HIGH) -> str:
    if p >= high:
        return "positive"
    if p <= low:
        return "negative"
    return "uncertain"


class SentimentService:
    """Loads the fine-tuned DistilBERT folder and scores texts. CPU is fine for a few reviews at a time."""

    def __init__(self, path: str, max_len: int = 128):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not Path(path).exists() and "/" not in str(path):
            raise FileNotFoundError(f"Sentiment model folder not found: {path}")
        self._torch = torch
        self.path, self.max_len = str(path), max_len
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).eval()

    def positive_probabilities(self, texts: list[str]) -> list[float]:
        # DistilBERT does not accept token_type_ids, and some tokenizer versions add them, so pass only what it needs.
        enc = self.tokenizer(texts, truncation=True, max_length=self.max_len, padding=True,
                             return_tensors="pt", return_token_type_ids=False)
        with self._torch.no_grad():
            logits = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
        return self._torch.softmax(logits.float(), dim=-1)[:, 1].tolist()


# --------------------------------------------------------------------------- churn
def _load_prep(path: Path):
    """Load the fitted preprocessor. Copies saved before Prep moved out of train_churn.py point at __main__.Prep."""
    import __main__

    from ml.preprocess import Prep

    if not hasattr(__main__, "Prep"):
        __main__.Prep = Prep
    return joblib.load(path)["prep"]


def load_churn_model(models_dir: Path = settings.MODELS_DIR, override: Optional[str] = None):
    """Returns (predict_fn, model_name, threshold). predict_fn maps a table of clues to churn scores."""
    metrics = json.loads((models_dir / "churn_metrics.json").read_text())
    name = override or metrics["selected_model"]
    if name not in CHURN_FILES:
        raise ValueError(f"Unknown churn model '{name}'. Choose one of {sorted(CHURN_FILES)}")
    threshold = float(metrics["results"][name]["thresholds_from_validation"]["recall_80"])
    prep = _load_prep(models_dir / "churn_prep.joblib")
    path = models_dir / CHURN_FILES[name]

    if name == "logistic_regression":
        lr = joblib.load(path)
        return (lambda df: lr.predict_proba(prep.scaled(df))[:, 1]), name, threshold
    if name == "gradient_boosting":
        gb = joblib.load(path)
        return (lambda df: gb.predict_proba(prep.tree_frame(df))[:, 1]), name, threshold

    import torch

    from ml import mlp as mlp_mod

    blob = torch.load(path, map_location="cpu", weights_only=True)
    nets = []
    for state in blob["states"]:
        net = mlp_mod.build_mlp(blob["n_in"])
        net.load_state_dict(state)
        nets.append(net.eval())
    return (lambda df: mlp_mod.mlp_proba(nets, prep.scaled(df))), name, threshold


class ChurnService:
    """Scores every recently active customer once at start-up, then answers look-ups instantly."""

    def __init__(self, orders, returns, customers, predict_fn: Callable, threshold: float, model_name: str,
                 as_of: Optional[str] = None, active_within_days: int = settings.ACTIVE_WITHIN_DAYS):
        self.model_name, self.threshold, self.active_within_days = model_name, float(threshold), active_within_days
        self.as_of = pd.Timestamp(as_of) if as_of else orders["order_date"].max().normalize() + pd.Timedelta(days=1)

        # identical clue-building code to training: no train/serve mismatch
        self.features = build_features(orders, returns, customers, self.as_of).set_index("customer_id")
        active = self.features[self.features["recency_days"] <= active_within_days]
        self.scores = pd.Series(np.asarray(predict_fn(active), dtype=float), index=active.index)
        log.info("Churn service ready: model=%s as_of=%s scored=%d of %d known customers",
                 model_name, self.as_of.date(), len(self.scores), len(self.features))

    @property
    def customers_scored(self) -> int:
        return len(self.scores)

    def _signals(self, row) -> dict:
        gap = row["avg_gap_days"]
        return {
            "days_since_last_order": int(row["recency_days"]),
            "orders_last_365d": int(row["n_orders_365d"]),
            "spend_last_365d": round(float(row["spend_365d"]), 2),
            "avg_days_between_orders": None if pd.isna(gap) else round(float(gap), 1),
        }

    def lookup(self, customer_id: int) -> Optional[dict]:
        if customer_id not in self.features.index:
            return None
        base = {"customer_id": int(customer_id), "as_of": str(self.as_of.date()), "model": self.model_name,
                "threshold": round(self.threshold, 4), "signals": self._signals(self.features.loc[customer_id])}
        if customer_id not in self.scores.index:
            return {**base, "scored": False, "churn_score": None, "flagged_at_risk": None,
                    "reason": f"No order in the last {self.active_within_days} days. The model only scores recently "
                              f"active customers; a customer this quiet has almost certainly already churned."}
        score = float(self.scores.loc[customer_id])
        return {**base, "scored": True, "churn_score": round(score, 4),
                "flagged_at_risk": score >= self.threshold, "reason": None}

    def top(self, limit: int) -> list[dict]:
        ids = self.scores.sort_values(ascending=False).head(limit).index
        return [self.lookup(int(i)) for i in ids]


def build_churn_service() -> ChurnService:
    """Reads the order tables from MySQL and loads the saved model."""
    from etl.db import get_engine
    from ml.churn_dataset import load_tables

    predict_fn, name, threshold = load_churn_model(override=settings.CHURN_MODEL)
    orders, returns, customers = load_tables(get_engine())
    return ChurnService(orders, returns, customers, predict_fn, threshold, name, as_of=settings.CHURN_AS_OF)
