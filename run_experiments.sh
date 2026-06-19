#!/usr/bin/env bash
# Run OPE experiments for the three public datasets.
# Usage:
#   pip install -r requirements.txt
#   bash run_experiments.sh
set -e
cd "$(dirname "$0")"

PYTHON="python3"

echo "========================================"
echo " attempts  (truncate @ 30 rounds)"
echo "========================================"
$PYTHON -u run_ope.py \
  --input-csv data/attempts/ktm_dataframe.csv \
  --input-is-ktm \
  --dataset-name attempts \
  --output-dir results/attempts \
  --truncate-at-round 30 \
  --max-rounds 50 \
  --rmse-threshold 5.0 \
  --objectives ips snips dr mis \
  --epochs 20 --batch-size 32768 --lr 0.005 --max-weight 20.0

echo ""
echo "========================================"
echo " assistments8000  (truncate @ 50 rounds)"
echo "========================================"
$PYTHON -u run_ope.py \
  --input-csv data/assistments8000/ktm_dataframe.csv \
  --input-is-ktm \
  --dataset-name assistments8000 \
  --output-dir results/assistments8000 \
  --truncate-at-round 50 \
  --max-rounds 50 \
  --rmse-threshold 5.0 \
  --objectives ips snips dr mis \
  --epochs 20 --batch-size 32768 --lr 0.005 --max-weight 20.0

echo ""
echo "========================================"
echo " skill_builder  (truncate @ 50 rounds)"
echo "========================================"
$PYTHON -u run_ope.py \
  --input-csv data/skill_builder/ktm_dataframe.csv \
  --input-is-ktm \
  --dataset-name skill_builder \
  --output-dir results/skill_builder \
  --truncate-at-round 50 \
  --max-rounds 50 \
  --rmse-threshold 5.0 \
  --objectives ips snips dr mis \
  --epochs 20 --batch-size 32768 --lr 0.005 --max-weight 20.0

echo ""
echo "All done. Open results/*/report.html to view results."
echo ""
echo "To plot learned policies:"
echo "  python3 plot_policies.py --results-dir results/attempts"
echo "  python3 plot_policies.py --results-dir results/assistments8000"
echo "  python3 plot_policies.py --results-dir results/skill_builder"
