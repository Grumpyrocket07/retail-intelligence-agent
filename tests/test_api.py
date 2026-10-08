"""API tests with stand-in models, so they run fast and need no database or GPU. Run: python -m pytest -q"""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.services import ChurnService, sentiment_label


def _orders():
    rows = [
        ("A1", 1, "2011-11-01 10:00", 100.0, 4, 4),
        ("A2", 1, "2011-12-01 10:00", 60.0, 2, 2),      # customer 1: active, orders about monthly
        ("B1", 2, "2010-01-05 10:00", 80.0, 3, 3),      # customer 2: last order ~700 days ago (dormant)
        ("C1", 3, "2011-12-09 10:00", 20.0, 1, 1),      # customer 3: one order, yesterday
        ("D1", 4, "2011-08-01 10:00", 300.0, 9, 7),     # customer 4: quiet for 4+ months, active within a year
    ]
    df = pd.DataFrame(rows, columns=["invoice_no", "customer_id", "order_date", "order_total", "n_lines", "n_products"])
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df


RETURNS = pd.DataFrame({"customer_id": [5], "return_date": pd.to_datetime(["2011-06-01"]), "line_total": [10.0]})
CUSTOMERS = pd.DataFrame({"customer_id": [1, 2, 3, 4, 5], "country_name": ["United Kingdom"] * 5})


def _score(df):
    """Stand-in model: the longer since the last order, the higher the score."""
    return (df["recency_days"] / (df["recency_days"] + 30)).to_numpy()


class FakeSentiment:
    def positive_probabilities(self, texts):
        out = []
        for t in texts:
            t = t.lower()
            out.append(0.97 if "great" in t else 0.02 if "broken" in t else 0.5)
        return out


@pytest.fixture()
def client():
    churn = ChurnService(_orders(), RETURNS, CUSTOMERS, _score, threshold=0.2, model_name="stand_in")
    with TestClient(create_app(churn=churn, sentiment=FakeSentiment(), load_real=False)) as c:
        yield c


def test_health_ok(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["churn"]["loaded"] and body["sentiment"]["loaded"]


def test_sentiment_labels(client):
    pos = client.post("/sentiment", json={"text": "Great gift!"}).json()
    neg = client.post("/sentiment", json={"text": "Arrived broken"}).json()
    mid = client.post("/sentiment", json={"text": "The parcel came on Tuesday."}).json()
    assert (pos["label"], neg["label"], mid["label"]) == ("positive", "negative", "uncertain")
    assert 0 <= pos["positive_probability"] <= 1


def test_sentiment_label_boundaries():
    assert sentiment_label(0.65) == "positive" and sentiment_label(0.35) == "negative"
    assert sentiment_label(0.5) == "uncertain" and sentiment_label(0.64) == "uncertain"


def test_sentiment_batch(client):
    res = client.post("/sentiment/batch", json={"texts": ["great", "broken"]}).json()["results"]
    assert [r["label"] for r in res] == ["positive", "negative"]


@pytest.mark.parametrize("payload", [{"text": ""}, {"text": "   "}, {"text": "x" * 2001}, {}, {"text": 5}])
def test_sentiment_rejects_bad_input(client, payload):
    assert client.post("/sentiment", json=payload).status_code == 422


def test_batch_limits(client):
    assert client.post("/sentiment/batch", json={"texts": []}).status_code == 422
    assert client.post("/sentiment/batch", json={"texts": ["ok"] * 65}).status_code == 422


def test_churn_scored_customer(client):
    body = client.get("/churn/1").json()
    assert body["scored"] and body["as_of"] == "2011-12-10"
    assert body["signals"]["days_since_last_order"] == 8       # 1 Dec 10:00 to 10 Dec 00:00 = 8 whole days
    assert body["signals"]["avg_days_between_orders"] == 30.0
    assert body["flagged_at_risk"] is True                     # 8 / (8 + 30) = 0.21 >= threshold 0.2


def test_churn_single_order_customer_has_no_gap(client):
    body = client.get("/churn/3").json()
    assert body["scored"] and body["signals"]["avg_days_between_orders"] is None
    assert body["flagged_at_risk"] is False


def test_churn_dormant_customer_is_not_scored(client):
    body = client.get("/churn/2").json()
    assert body["scored"] is False and body["churn_score"] is None and "No order" in body["reason"]


def test_churn_unknown_customer_is_404(client):
    assert client.get("/churn/999").status_code == 404
    assert client.get("/churn/5").status_code == 404           # only has a return, no orders
    assert client.get("/churn/abc").status_code == 422


def test_churn_top_is_sorted_and_limited(client):
    body = client.get("/churn/top?limit=2").json()
    ids = [c["customer_id"] for c in body["customers"]]
    assert ids == [4, 1] and body["customers_scored"] == 3     # dormant customer 2 is not scored
    scores = [c["churn_score"] for c in body["customers"]]
    assert scores == sorted(scores, reverse=True)
    assert client.get("/churn/top?limit=0").status_code == 422
    assert client.get("/churn/top?limit=101").status_code == 422


def test_missing_models_give_503_not_a_crash():
    with TestClient(create_app(load_real=False)) as c:
        assert c.get("/health").json()["status"] == "degraded"
        assert c.post("/sentiment", json={"text": "hello"}).status_code == 503
        assert c.get("/churn/1").status_code == 503


def test_sentiment_failure_explains_itself():
    class Broken:
        def positive_probabilities(self, texts):
            raise TypeError("unexpected keyword argument 'token_type_ids'")

    with TestClient(create_app(sentiment=Broken(), load_real=False)) as c:
        r = c.post("/sentiment", json={"text": "hello"})
        assert r.status_code == 500 and "token_type_ids" in r.json()["detail"]
