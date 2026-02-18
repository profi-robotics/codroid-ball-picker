#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_MD="${REPO_ROOT}/docs/manuals/user_manual_uk.md"
METADATA_YAML="${REPO_ROOT}/docs/manuals/user_manual_uk.yaml"
DEFAULT_OUTPUT="${REPO_ROOT}/docs/manuals/user_manual_uk.pdf"

ENGINE="pdflatex"
OUTPUT="${DEFAULT_OUTPUT}"
PANDOC_BIN=""

usage() {
  cat <<USAGE
Usage: bash scripts/build_user_manual_pdf.sh [--engine ENGINE] [--output PATH]

Options:
  --engine ENGINE   PDF engine for pandoc: pdflatex, xelatex, lualatex, or weasyprint (default: pdflatex)
  --output PATH     Output PDF path (default: docs/manuals/user_manual_uk.pdf)
  -h, --help        Show this help message
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --engine)
      if [[ $# -lt 2 ]]; then
        echo "Error: --engine requires a value." >&2
        usage
        exit 1
      fi
      ENGINE="$2"
      shift 2
      ;;
    --output)
      if [[ $# -lt 2 ]]; then
        echo "Error: --output requires a value." >&2
        usage
        exit 1
      fi
      OUTPUT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown argument '$1'." >&2
      usage
      exit 1
      ;;
  esac
done

case "${ENGINE}" in
  pdflatex|xelatex|lualatex|weasyprint)
    ;;
  *)
    echo "Error: unsupported engine '${ENGINE}'. Use pdflatex, xelatex, lualatex, or weasyprint." >&2
    exit 1
    ;;
esac

if command -v pandoc >/dev/null 2>&1; then
  PANDOC_BIN="$(command -v pandoc)"
else
  PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="$(command -v python3 || command -v python || true)"
  fi

  if [[ -z "${PYTHON_BIN}" ]]; then
    echo "Error: Python not found; cannot resolve bundled pandoc path." >&2
    exit 1
  fi

  PANDOC_BIN="$(${PYTHON_BIN} - <<'PY'
import sys

try:
    import pypandoc
except Exception:
    print("")
    raise SystemExit(0)

try:
    print(pypandoc.get_pandoc_path())
except Exception:
    print("")
PY
)"

  if [[ -z "${PANDOC_BIN}" || ! -x "${PANDOC_BIN}" ]]; then
    echo "Error: pandoc is not installed and bundled pandoc is unavailable." >&2
    echo "Install system pandoc or run 'uv sync' to install pypandoc-binary." >&2
    exit 1
  fi
fi

if [[ ! -f "${INPUT_MD}" ]]; then
  echo "Error: input markdown not found: ${INPUT_MD}" >&2
  exit 1
fi

if [[ ! -f "${METADATA_YAML}" ]]; then
  echo "Error: metadata YAML not found: ${METADATA_YAML}" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT}")"

ENGINE_BIN="${ENGINE}"
if ! command -v "${ENGINE}" >/dev/null 2>&1; then
  if [[ -x "${REPO_ROOT}/.venv/bin/${ENGINE}" ]]; then
    ENGINE_BIN="${REPO_ROOT}/.venv/bin/${ENGINE}"
  elif [[ "${ENGINE}" != "weasyprint" ]] && command -v weasyprint >/dev/null 2>&1; then
    echo "Info: ${ENGINE} not found; falling back to weasyprint." >&2
    ENGINE_BIN="$(command -v weasyprint)"
  elif [[ "${ENGINE}" != "weasyprint" ]] && [[ -x "${REPO_ROOT}/.venv/bin/weasyprint" ]]; then
    echo "Info: ${ENGINE} not found; falling back to weasyprint." >&2
    ENGINE_BIN="${REPO_ROOT}/.venv/bin/weasyprint"
  else
    echo "Error: requested PDF engine '${ENGINE}' is not available." >&2
    exit 1
  fi
fi

"${PANDOC_BIN}" \
  "${INPUT_MD}" \
  --metadata-file "${METADATA_YAML}" \
  --pdf-engine "${ENGINE_BIN}" \
  -o "${OUTPUT}"

echo "PDF generated: ${OUTPUT}"
