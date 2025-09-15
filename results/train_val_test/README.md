# `results/train_val_test/` — Train/Validation/Test results

This directory contains the outputs from experiments using a **3-way split**:  
- **70%** for training  
- **15%** for validation (model selection / hyperparameter tuning)  
- **15%** for final testing (unseen evaluation)  

## Files
- `train.csv` — Training subset.  
- `validation.csv` — Validation subset.  
- `test.csv` — Testing subset.  
- `metadata.json` — Split ratios, seed used, class distributions, and dataset sizes.  
- `tree.txt` — Human-readable decision tree structure.  
- `tree.png` — Render of the decision tree (skipped if too large).  
- `classification_report.txt` — Precision, recall, F1, support per class (on test set).  
- `confusion_matrix.txt` / `confusion_matrix.png` — Confusion matrix (counts + heatmap, test set).  
- `per_class_metrics.png` — Bar chart of per-class Precision/Recall/F1 (test set).  
- `accuracy.png` — Single-bar accuracy visualization (test set).  
- `learning_curve.png` — Training vs. validation accuracy as training size increases (diagnoses bias/variance).  
- `model_classification.txt` — Qualitative summary of bias, variance, and fit level.  

## Notes
- This mode provides a **more realistic estimate of generalization**, since validation is separated from the final test.  
- Hyperparameter tuning (via GridSearchCV) or pruning can use the validation set, avoiding information leakage into the test set.  
- All files are re-generated on each run; previous contents are removed automatically.  