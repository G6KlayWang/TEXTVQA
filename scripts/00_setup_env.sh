#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python - <<'PY'
try:
    import nltk
    nltk.download("wordnet")
    nltk.download("omw-1.4")
except Exception as exc:
    print(f"Skipping NLTK resource download: {exc}")
PY

