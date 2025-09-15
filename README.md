# Hybrid ID3 — Scikit-learn

An improved implementation of a **hybrid ID3 decision tree** built on top of **scikit-learn**, with robust preprocessing and support for both categorical and numerical data.  
Developed as part of the implementation portfolio for **TC3006C – Advanced AI for Data Science I (G101)** at Tecnológico de Monterrey, Campus Estado de México.

## Features

- **Data normalization**: Cleans tokens consistently (removes quotes, trims whitespace, converts empty strings to `None`).  
- **Robust missing-value handling**:  
  - Numeric → imputed with the column **median**.  
  - Categorical → imputed with the constant `"__MISSING__"`.  
- **Categorical encoding**: Uses **OneHotEncoder** with `handle_unknown="ignore"`.  
- **ID3 alignment**: Configured with `criterion="entropy"` in `DecisionTreeClassifier` to replicate **information gain**.  
- **Optional optimization**:  
  - **GridSearchCV** (disabled by default) for hyperparameter tuning (`max_depth`, `min_samples_leaf`, `min_samples_split`, `ccp_alpha`, `max_features`).  
  - **Cost-Complexity Pruning (ccp_alpha)** option available at runtime to reduce overfitting.  
- **Learning diagnostics**:  
  - **Learning curves** (train vs validation accuracy) generated automatically in `train_val_test`.  
  - Bias/variance/fit classification stored in `model_classification.txt`.  
- **Readable tree output**: `print_tree()` leverages `sklearn.tree.export_text`, preserving original and OneHotEncoded feature names.  
- **Drop-in replacement**: Designed as an evolution of the previous [hybrid-id3-from-scratch](https://github.com/Ulises-JPx/hybrid-id3-from-scratch) implementation, now powered by scikit-learn’s pipeline ecosystem for greater flexibility.

## Requirements

```text
matplotlib>=3.8
numpy>=1.26
scikit-learn>=1.3
pandas>=2.0
```

## Author

**Ulises Jaramillo Portilla (A01798380)** — Ulises-JPx  
Course: **TC3006C – Advanced AI for Data Science I (G101), Tecnológico de Monterrey (CEM).**
