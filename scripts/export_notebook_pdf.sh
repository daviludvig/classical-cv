#!/usr/bin/env bash
set -euo pipefail

# Export a notebook to HTML, then try to render PDF via headless Chrome/Chromium.
# This avoids the TeX/xelatex dependency used by nbconvert --to pdf.

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <notebook.ipynb> [output_dir]"
  exit 1
fi

NOTEBOOK_PATH="$1"
OUTPUT_DIR="${2:-outputs/exports}"

if [[ ! -f "$NOTEBOOK_PATH" ]]; then
  echo "Notebook not found: $NOTEBOOK_PATH"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

PYTHON_BIN=".venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

echo "[1/2] Exporting HTML..."
"$PYTHON_BIN" -m jupyter nbconvert --to html "$NOTEBOOK_PATH" --output-dir "$OUTPUT_DIR"

BASE_NAME="$(basename "$NOTEBOOK_PATH" .ipynb)"
HTML_PATH="$OUTPUT_DIR/$BASE_NAME.html"
PDF_PATH="$OUTPUT_DIR/$BASE_NAME.pdf"

if [[ ! -f "$HTML_PATH" ]]; then
  echo "Expected HTML not found: $HTML_PATH"
  exit 1
fi

HTML_ABS_PATH="$(cd "$(dirname "$HTML_PATH")" && pwd)/$(basename "$HTML_PATH")"

# Detect Chrome/Chromium command on macOS/Linux
CHROME_CMD=""
if command -v google-chrome >/dev/null 2>&1; then
  CHROME_CMD="google-chrome"
elif command -v chromium >/dev/null 2>&1; then
  CHROME_CMD="chromium"
elif command -v chromium-browser >/dev/null 2>&1; then
  CHROME_CMD="chromium-browser"
elif [[ -x "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" ]]; then
  CHROME_CMD="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
fi

if [[ -n "$CHROME_CMD" ]]; then
  echo "[2/2] Rendering PDF with headless Chrome..."
  "$CHROME_CMD" \
    --headless=new \
    --disable-gpu \
    --allow-file-access-from-files \
    --print-to-pdf="$PDF_PATH" \
    "file://$HTML_ABS_PATH"

  echo "Done: $PDF_PATH"
else
  echo "[2/2] Chrome/Chromium not found. HTML export completed: $HTML_PATH"
  echo "Open this HTML in your browser and use Print -> Save as PDF."
fi
