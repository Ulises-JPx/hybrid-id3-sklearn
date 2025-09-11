# model — Hybrid ID3 (scikit-learn)

This package contains the **DecisionTreeID3Sklearn** implementation, built on top of **scikit-learn pipelines**.  
It extends ID3 with robust preprocessing and integration of categorical + numeric features.

## Highlights
- **Data normalization**: automatic cleaning of tokens (removes quotes, empty → `None`).  
- **Missing-value handling**:  
  - Numeric → imputed with median.  
  - Categorical → imputed with `"__MISSING__"`.  
- **Categorical encoding**: OneHotEncoder with `handle_unknown="ignore"`.  
- **ID3 alignment**: decision tree trained with `criterion="entropy"` to replicate information gain.  
- **Optional tuning**: `GridSearchCV` support for `max_depth`, `min_samples_leaf`, and `ccp_alpha`.  
- **Readable tree**: `print_tree()` uses `sklearn.tree.export_text`, preserving the expanded (post-OHE) feature names.  
- **Drop-in replacement**: mirrors the API of the *from scratch* `DecisionTreeID3Plus` while adding sklearn’s ecosystem.  