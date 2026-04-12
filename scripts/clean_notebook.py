#!/usr/bin/env python3
"""Strip outputs and execution metadata from Jupyter notebooks.

Usage as git filter:
    git config filter.strip-notebook.clean 'python scripts/clean_notebook.py'

Or manually:
    python scripts/clean_notebook.py < notebook.ipynb > cleaned.ipynb
"""
import json
import sys


def clean(nb: dict) -> dict:
    for cell in nb.get("cells", []):
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    # Remove kernel / language info that contains local paths
    md = nb.get("metadata", {})
    ks = md.get("kernelspec", {})
    ks.pop("display_name", None)
    ks.pop("name", None)
    md.pop("language_info", None)
    return nb


if __name__ == "__main__":
    nb = json.load(sys.stdin)
    cleaned = clean(nb)
    json.dump(cleaned, sys.stdout, indent=1, ensure_ascii=False)
    sys.stdout.write("\n")
