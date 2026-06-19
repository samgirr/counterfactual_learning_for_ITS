# Off-Policy KTM Pipeline

This folder provides a clean end-to-end structure for:

1. PIX dataframe preprocessing and KTM feature construction
2. Gaussian regression by order
3. Propensity computation and off-policy tuple creation
4. Off-policy estimators + policy training
5. End-to-end notebook with visualizations

## Structure

- `src/dataframe_processing.py`
  - `process_pix_dataset(...)`
  - `build_ktm_dataframe(...)`
- `src/gaussian_regression.py`
  - `fit_gaussian_regression(...)`
  - `fit_gaussian_by_order(...)`
- `src/distribution_nll_comparison.py`
  - compare Normal/Laplace/Logistic/Student-t via NLL
  - `compare_distributions(...)`
- `src/distribution_nll_by_order.py`
  - same comparison per `order_sequence`
  - outputs winner table + NLL curves by order
- `src/propensity.py`
  - `attach_behavior_params(...)`
  - `compute_propensity_and_reward(...)`
  - `build_offpolicy_dataset(...)`
- `src/estimators_and_training.py`
  - `train_global_policy(...)`
  - estimators: IPS, SNIPS, CIPS, CSNIPS, DR, SNDR, CDR, CSNDR
  - `compute_log_mixture_propensity(...)`
  - `mc_policy_value(...)`, `optimal_irt_value(...)`
- `src/utils.py`
  - `set_global_seed(...)`
  - `get_default_device(...)`
  - `ensure_parent_dir(...)`
- `src/visualization.py`
  - training curves + policy curve plotting helpers
- `notebooks/end_to_end_offpolicy_pipeline.ipynb`
  - runnable notebook from raw/processed data to plots

## Dependencies

Install from:

- `offpolicy_ktm_pipeline/requirements.txt`

Main libraries used:

- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `torch`
- `tqdm`
- `Pillow`
- `jupyter`
- `ipykernel`

## Run

Open:

- `offpolicy_ktm_pipeline/notebooks/end_to_end_offpolicy_pipeline.ipynb`

The notebook imports from `offpolicy_ktm_pipeline/src` and runs the full workflow.

## One-Shot Script

You can run the whole process with:

`python offpolicy_ktm_pipeline/run_full_pipeline.py`

If your dataset has no explicit order column, add:

`--derive-answer-number-per-user`

This creates `answer_number = 1..len(user)` for each user (row order).

If you want to cap the number of rounds (merge long sequences/outliers), add:

- `--max-rounds <K>`
- optional `--round-cap-strategy tail|quantile`

`tail` keeps early rounds separate and merges the long tail into the last round bucket.

Main outputs in `--output-dir`:

- `gaussian_fit_by_order.csv` (fit quality per order: NLL, RMSE, MAE)
- `offpolicy_tuples.csv`
- `policy_round_results.csv` (behavior/learned/optimal values per round)
- `policy_value_table.csv` (compact table with policy values + IS/DR estimators)
- `policy_round_results_with_gaussian_fit.csv` (round table merged with Gaussian fit quality)
- `gaussian_fit_quality_by_order.png`
- `policy_values_by_round.png`

Optional animation:

- add `--make-animation` (plus `--animation-fps`, `--animation-max-scatter`)
- produces `policy_round_animation.gif` with policy curves and estimator values per round.
