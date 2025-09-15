# `results/` — Experiment results

This directory is **auto-generated** on each run and is **cleaned at startup** to avoid stale files.

## Structure
- **`showcase/`** — Train **and** evaluate on the full dataset (demo/presentation metrics).  
- **`validation/`** — Train/test split (default 80/20); metrics reflect **generalization**.  
- **`train_val_test/`** — Three-way split (70/15/15); validation used for tuning, test kept for final evaluation.  

## Files (per subfolder)
- `train.csv`, `validation.csv`, `test.csv` — Subsets used in training/validation/testing (depending on mode).  
- `metadata.json` — Information about split ratios, seed, class distribution, and sizes.  
- `tree.txt` — Human-readable decision tree.  
- `tree.png` — Render of the decision tree (skipped if too large).  
- `classification_report.txt` — Precision, recall, F1, and support per class.  
- `confusion_matrix.txt` / `confusion_matrix.png` — Confusion matrix (counts + heatmap).  
- `per_class_metrics.png` — Bar chart of per-class Precision/Recall/F1.  
- `accuracy.png` — Single-bar accuracy visualization.  
- `learning_curve.png` — Training vs. validation accuracy across sample sizes (**only in `train_val_test/`**).  
- `model_classification.txt` — Bias, variance, and overall fit interpretation.  

## Notes
- Ratios and random seeds for splits are configured in `main.py`.  
- GridSearchCV and pruning (ccp_alpha) can be optionally enabled and affect results saved here.  
- All contents are re-generated on each run; previous files are removed automatically.  