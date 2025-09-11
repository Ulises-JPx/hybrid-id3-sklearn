# Hybrid ID3 — Scikit-learn Enhanced

An improved implementation of a **hybrid ID3 decision tree** built on top of **scikit-learn**, with robust preprocessing and support for both categorical and numerical data.  
Developed for the implementation portfolio of **TC3006C – Advanced AI for Data Science I (G101)** at Tecnológico de Monterrey, Campus Estado de México.

## Features

- **Data normalization**: Cleans tokens consistently (removes quotes, converts empty strings to `None`).  
- **Robust missing-value handling**:  
  - Numeric → imputed with the column median.  
  - Categorical → imputed with `"__MISSING__"`.  
- **Categorical encoding**: OneHotEncoder with `handle_unknown="ignore"`.  
- **ID3 alignment**: Uses `criterion="entropy"` in `DecisionTreeClassifier` to replicate information gain.  
- **Optional optimization**: `GridSearchCV` available (disabled by default) for tuning `ccp_alpha`, `max_depth`, and `min_samples_leaf`.  
- **Readable tree output**: `print_tree()` uses `sklearn.tree.export_text`, preserving original feature names (post-OHE).  
- **Drop-in replacement**: Designed as a replacement for the previous `DecisionTreeID3Plus` [See related repository: hybrid-id3-from-scratch](https://github.com/Ulises-JPx/hybrid-id3-from-scratch)  class, but now powered by sklearn’s pipeline ecosystem.  

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