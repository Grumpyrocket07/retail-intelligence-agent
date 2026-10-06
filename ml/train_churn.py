"""Milestone 2b: train and compare churn models.

Models compared (all trained on the train snapshots, tuned on the validation snapshot, judged once on the test snapshot):
  1. one-clue rule    : "the longer since the last order, the likelier to churn"  (the bar to beat)
  2. logistic regression
  3. gradient boosting (scikit-learn)
  4. PyTorch neural network (average of 5 random starts)

Run:  python -m ml.train_churn
Writes models/*, models/churn_metrics.json and docs/churn_pr_curve.png
"""
import json
import logging
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from ml.churn_dataset import FEATURES
from ml.preprocess import Prep

try:
    from ml import mlp as mlp_mod
    HAVE_TORCH = True
except ImportError:  # PyTorch not installed: train the other models only
    mlp_mod, HAVE_TORCH = None, False

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ml" / "churn_dataset.csv"
MODELS = ROOT / "models"
DOCS = ROOT / "docs"
SEEDS = [0, 1, 2, 3, 4]

log = logging.getLogger("ml.train_churn")


# --------------------------------------------------------------------------- evaluation
def pick_thresholds(y, p) -> dict:
    """Both thresholds are chosen on VALIDATION data only."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    t_f1 = float(thr[int(np.argmax(f1))])
    ok = np.where(rec[:-1] >= 0.80)[0]
    t_r80 = float(thr[ok[-1]]) if len(ok) else float(thr[0])
    return {"best_f1": t_f1, "recall_80": t_r80}


def at_threshold(y, p, t) -> dict:
    pred = p >= t
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    fn, tn = int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"threshold": round(t, 4), "precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(2 * precision * recall / max(precision + recall, 1e-9), 3),
            "accuracy": round((tp + tn) / len(y), 3), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def summarise(name, yva, pva, yte, pte) -> dict:
    th = pick_thresholds(yva, pva)
    return {
        "val_pr_auc": round(average_precision_score(yva, pva), 4),
        "test_pr_auc": round(average_precision_score(yte, pte), 4),
        "test_roc_auc": round(roc_auc_score(yte, pte), 4),
        "thresholds_from_validation": {k: round(v, 4) for k, v in th.items()},
        "test_at_best_f1": at_threshold(yte, pte, th["best_f1"]),
        "test_at_recall_80": at_threshold(yte, pte, th["recall_80"]),
    }


def bootstrap_gain(y, p_model, p_rule, n=300, seed=0) -> dict:
    """Resample the test customers many times: how much PR-AUC does the model gain over the simple rule, and how sure are we?"""
    rng = np.random.default_rng(seed)
    gains = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        gains.append(average_precision_score(y[idx], p_model[idx]) - average_precision_score(y[idx], p_rule[idx]))
    lo, hi = np.percentile(gains, [2.5, 97.5])
    return {"mean": round(float(np.mean(gains)), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}


def permutation_importance(predict_fn, df, y, n_repeats=5, seed=0) -> dict:
    """How much does PR-AUC drop when one clue is shuffled? Bigger drop = more important."""
    rng = np.random.default_rng(seed)
    base = average_precision_score(y, predict_fn(df))
    out = {}
    for col in FEATURES:
        drops = []
        for _ in range(n_repeats):
            shuffled = df.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            drops.append(base - average_precision_score(y, predict_fn(shuffled)))
        out[col] = round(float(np.mean(drops)), 4)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


# --------------------------------------------------------------------------- main
def main():
    MODELS.mkdir(exist_ok=True)
    DOCS.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    train, val, test = (df[df["split"] == s].reset_index(drop=True) for s in ("train", "validation", "test"))
    ytr, yva, yte = (d["churned"].to_numpy() for d in (train, val, test))
    log.info("train %d rows (%.0f%% churn) | validation %d (%.0f%%) | test %d (%.0f%%)",
             len(train), 100 * ytr.mean(), len(val), 100 * yva.mean(), len(test), 100 * yte.mean())

    prep = Prep().fit(train)
    xtr, xva, xte = prep.scaled(train), prep.scaled(val), prep.scaled(test)
    predictors, results = {}, {}

    # 1. one-clue rule: score = days since last order
    results["one_clue_rule"] = summarise("rule", yva, val["recency_days"].to_numpy(float),
                                         yte, test["recency_days"].to_numpy(float))
    predictors["one_clue_rule"] = lambda d: d["recency_days"].to_numpy(float)

    # 2. logistic regression
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(xtr, ytr)
    predictors["logistic_regression"] = lambda d: lr.predict_proba(prep.scaled(d))[:, 1]
    # 3. gradient boosting, small search judged on the validation snapshot
    best = None
    for depth in (3, 5):
        for lr_rate in (0.03, 0.08):
            for iters in (150, 300):
                gb = HistGradientBoostingClassifier(max_depth=depth, learning_rate=lr_rate, max_iter=iters,
                                                    l2_regularization=1.0, random_state=0)
                gb.fit(prep.tree_frame(train), ytr)
                ap = average_precision_score(yva, gb.predict_proba(prep.tree_frame(val))[:, 1])
                if best is None or ap > best[0]:
                    best = (ap, gb, {"max_depth": depth, "learning_rate": lr_rate, "max_iter": iters})
    gb, gb_params = best[1], best[2]
    log.info("gradient boosting settings chosen on validation: %s", gb_params)
    predictors["gradient_boosting"] = lambda d: gb.predict_proba(prep.tree_frame(d))[:, 1]

    # 4. PyTorch network, 5 random starts averaged
    nets = None
    if HAVE_TORCH:
        nets = [mlp_mod.train_mlp(xtr, ytr, xva, yva, seed=s) for s in SEEDS]
        predictors["pytorch_mlp"] = lambda d: mlp_mod.mlp_proba(nets, prep.scaled(d))
    else:
        log.warning("PyTorch is not installed, skipping the neural network (pip install torch)")

    probs = {}
    for name, fn in predictors.items():
        pva, pte = fn(val), fn(test)
        probs[name] = pte
        if name not in results:
            results[name] = summarise(name, yva, pva, yte, pte)

    for name in probs:
        if name != "one_clue_rule":
            results[name]["test_pr_auc_gain_over_rule"] = bootstrap_gain(yte, probs[name], probs["one_clue_rule"])

    # pick the model to serve: best validation PR-AUC among the real models
    contenders = [n for n in ("logistic_regression", "gradient_boosting", "pytorch_mlp") if n in results]
    selected = max(contenders, key=lambda n: results[n]["val_pr_auc"])
    log.info("selected for serving (best validation PR-AUC): %s", selected)

    importance = permutation_importance(predictors[selected], val, yva)

    # save artefacts
    joblib.dump({"prep": prep, "features": FEATURES}, MODELS / "churn_prep.joblib")
    joblib.dump(lr, MODELS / "churn_logreg.joblib")
    joblib.dump(gb, MODELS / "churn_gb.joblib")
    if nets is not None:
        import torch
        torch.save({"seeds": SEEDS, "n_in": xtr.shape[1], "states": [m.state_dict() for m in nets]}, MODELS / "churn_mlp.pt")
    metrics = {
        "rows": {"train": len(train), "validation": len(val), "test": len(test)},
        "churn_rate": {"train": round(float(ytr.mean()), 4), "validation": round(float(yva.mean()), 4), "test": round(float(yte.mean()), 4)},
        "gradient_boosting_settings": gb_params, "selected_model": selected,
        "results": results, "permutation_importance_selected_on_validation": importance,
    }
    (MODELS / "churn_metrics.json").write_text(json.dumps(metrics, indent=2))

    # precision-recall curves on the test snapshot
    plt.figure(figsize=(6.5, 4.5))
    for name, p in probs.items():
        prec, rec, _ = precision_recall_curve(yte, p)
        plt.plot(rec, prec, label=f"{name} (PR-AUC {results[name]['test_pr_auc']:.2f})")
    plt.axhline(yte.mean(), color="grey", ls="--", label=f"always predict churn ({yte.mean():.0%})")
    plt.xlabel("recall"); plt.ylabel("precision"); plt.title("Churn models on the test snapshot (9 Sep 2011)")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(DOCS / "churn_pr_curve.png", dpi=130)

    # readable summary
    print(f"\nChurn rate  train {ytr.mean():.1%} | validation {yva.mean():.1%} | test {yte.mean():.1%}\n")
    rows = []
    for n, r in results.items():
        f, r80 = r["test_at_best_f1"], r["test_at_recall_80"]
        rows.append({"model": n, "val PR-AUC": r["val_pr_auc"], "test PR-AUC": r["test_pr_auc"], "test ROC-AUC": r["test_roc_auc"],
                     "precision": f["precision"], "recall": f["recall"], "F1": f["f1"],
                     "precision@80%recall": r80["precision"], "recall there": r80["recall"]})
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nGain in test PR-AUC over the one-clue rule (95% range from resampling):")
    for n, r in results.items():
        if "test_pr_auc_gain_over_rule" in r:
            g = r["test_pr_auc_gain_over_rule"]
            print(f"  {n:20s} {g['mean']:+.3f}  ({g['ci95'][0]:+.3f} to {g['ci95'][1]:+.3f})")
    print(f"\nSelected: {selected}\nMost important clues (PR-AUC lost when shuffled):")
    for k, v in list(importance.items())[:8]:
        print(f"  {k:24s} {v:.4f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
