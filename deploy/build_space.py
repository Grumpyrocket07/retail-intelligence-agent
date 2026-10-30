"""Assembles the files for a Hugging Face Space into build/space.   python deploy/build_space.py

Only code, the documents and the small churn model files are copied. Never copied: .env, the virtual environment, the dataset,
the 250 MB sentiment model (it is loaded from the Hugging Face Hub instead), tests and evaluation progress files.
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "space"
CODE_DIRS = ["app", "agent", "api", "ml", "observability", "rag", "sql_agent"]
ETL_FILES = ["__init__.py", "db.py"]
CHURN_FILES = ["churn_prep.joblib", "churn_logreg.joblib", "churn_gb.joblib", "churn_mlp.pt", "churn_metrics.json"]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*progress*.json")
FORBIDDEN = (".env", ".venv", "online_retail", "sentiment_model", ".xlsx", ".safetensors")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for name in CODE_DIRS:
        shutil.copytree(ROOT / name, OUT / name, ignore=IGNORE)
    (OUT / "etl").mkdir()
    for name in ETL_FILES:
        shutil.copy(ROOT / "etl" / name, OUT / "etl" / name)
    (OUT / "models").mkdir()
    missing = [f for f in CHURN_FILES if not (ROOT / "models" / f).exists()]
    for name in CHURN_FILES:
        if (ROOT / "models" / name).exists():
            shutil.copy(ROOT / "models" / name, OUT / "models" / name)
    shutil.copy(ROOT / "deploy" / "space_README.md", OUT / "README.md")
    shutil.copy(ROOT / "deploy" / "space_requirements.txt", OUT / "requirements.txt")

    files = sorted(p for p in OUT.rglob("*") if p.is_file())
    bad = [str(p.relative_to(OUT)) for p in files if any(f in str(p.relative_to(OUT)).lower() for f in FORBIDDEN)]
    total = sum(p.stat().st_size for p in files)
    print(f"Built {OUT} : {len(files)} files, {total / 1024:.0f} KB")
    if missing:
        print("WARNING: churn model files not found (run python -m ml.train_churn first):", ", ".join(missing))
    if bad:
        sys.exit(f"STOP: files that must not be uploaded were found: {bad}")
    print("Nothing secret or oversized was copied. Next: see deploy/DEPLOY.md, step C.")


if __name__ == "__main__":
    main()
