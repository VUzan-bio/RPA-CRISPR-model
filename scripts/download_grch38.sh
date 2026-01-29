#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-data/genomes/GRCh38}"
ZIP_PATH="${OUT_DIR}/GRCh38.zip"

URL="https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/GCF_000001405.40/download?include_annotation_type=GENOME_FASTA&include_annotation_type=GENOME_GFF&include_annotation_type=RNA_FASTA&include_annotation_type=CDS_FASTA&include_annotation_type=PROT_FASTA&include_annotation_type=SEQUENCE_REPORT&hydrated=FULLY_HYDRATED"

mkdir -p "${OUT_DIR}"

if [[ -n "${NCBI_API_KEY:-}" ]]; then
  curl -L -H "Accept: application/zip" -H "api-key: ${NCBI_API_KEY}" "${URL}" -o "${ZIP_PATH}"
else
  curl -L -H "Accept: application/zip" "${URL}" -o "${ZIP_PATH}"
fi

if command -v unzip >/dev/null 2>&1; then
  unzip -q "${ZIP_PATH}" -d "${OUT_DIR}"
elif command -v python >/dev/null 2>&1; then
  python - "${ZIP_PATH}" "${OUT_DIR}" <<'PY'
import sys
import zipfile
from pathlib import Path

zip_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(out_dir)
PY
elif command -v python3 >/dev/null 2>&1; then
  python3 - "${ZIP_PATH}" "${OUT_DIR}" <<'PY'
import sys
import zipfile
from pathlib import Path

zip_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(out_dir)
PY
elif command -v powershell.exe >/dev/null 2>&1; then
  powershell.exe -NoProfile -Command \
    "Expand-Archive -Path '${ZIP_PATH}' -DestinationPath '${OUT_DIR}' -Force"
elif command -v pwsh >/dev/null 2>&1; then
  pwsh -NoProfile -Command \
    "Expand-Archive -Path '${ZIP_PATH}' -DestinationPath '${OUT_DIR}' -Force"
else
  echo "ERROR: 'unzip', 'python', and PowerShell not available for extraction." >&2
  echo "Install unzip or run: python -m zipfile -e ${ZIP_PATH} ${OUT_DIR}" >&2
  exit 1
fi

echo "Downloaded to ${OUT_DIR}"
