"""
DecisionTreeID3Sklearn (improved):
- Normaliza valores como tu versión sin frameworks (quita comillas, vacíos -> None)
- Manejo robusto de missing: imputación (num: mediana, cat: "__MISSING__")
- Codificación de categóricos con OneHotEncoder(handle_unknown="ignore")
- Árbol de sklearn con entropía (info gain) para alinearlo con ID3
- Opción de GridSearchCV (desactivada por defecto) con ccp_alpha, max_depth, min_samples_leaf
- print_tree() via sklearn.tree.export_text (muestra nombres originales de features)

Drop-in replacement de tu clase anterior `DecisionTreeID3Sklearn`.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import GridSearchCV


def _clean_token(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if len(s) >= 2 and ((s[0] == s[-1] == "'") or (s[0] == s[-1] == '"')):
        s = s[1:-1].strip()
    return s if s != "" else None


def _normalize_table(X: List[List[Any]], y: List[Any]):
    Xn = [[_clean_token(v) for v in row] for row in X]
    yn = [_clean_token(t) for t in y]
    return Xn, yn


def _detect_feature_types(X: List[List[Any]]) -> List[str]:
    types: List[str] = []
    for col in zip(*X):
        is_numeric = True
        for v in col:
            vv = _clean_token(v)
            if vv is None:
                continue
            try:
                float(vv)
            except ValueError:
                is_numeric = False
                break
        types.append("numeric" if is_numeric else "categorical")
    return types


class DecisionTreeID3Sklearn:
    def __init__(
        self,
        criterion: str = "entropy",  # align with ID3 (information gain)
        random_state: Optional[int] = 42,
    ):
        self.criterion = criterion
        self.random_state = random_state
        self.pipeline: Optional[Pipeline] = None
        self.clf_: Optional[DecisionTreeClassifier] = None
        self.feature_names_: List[str] = []
        self.feature_types_: List[str] = []
        self._ohe_feature_names_: Optional[List[str]] = None

    def _build_pipeline(self, X: List[List[Any]], feature_names: List[str]) -> Pipeline:
        self.feature_types_ = _detect_feature_types(X)
        self.feature_names_ = list(feature_names)

        num_idx = [i for i, t in enumerate(self.feature_types_) if t == "numeric"]
        cat_idx = [i for i, t in enumerate(self.feature_types_) if t == "categorical"]

        num_tf = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        cat_tf = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        pre = ColumnTransformer(
            transformers=[
                ("num", num_tf, num_idx),
                ("cat", cat_tf, cat_idx),
            ],
            remainder="drop",
        )

        clf = DecisionTreeClassifier(
            criterion=self.criterion,
            random_state=self.random_state,
        )

        pipe = Pipeline(
            steps=[
                ("pre", pre),
                ("clf", clf),
            ]
        )
        return pipe

    def train(
        self,
        X: List[List[Any]],
        y: List[str],
        feature_names: List[str],
        use_gridsearch: bool = False,
    ) -> None:
        Xn, yn = _normalize_table(X, y)
        self.pipeline = self._build_pipeline(Xn, feature_names)

        if use_gridsearch:
            param_grid = {
                "clf__max_depth": [None, 4, 6, 10, 15],
                "clf__min_samples_leaf": [1, 2, 5, 10],
                "clf__ccp_alpha": [0.0, 1e-4, 1e-3, 1e-2],
            }
            gs = GridSearchCV(
                self.pipeline,
                param_grid=param_grid,
                cv=5,
                n_jobs=-1,
                scoring="accuracy",
                refit=True,
            )
            gs.fit(Xn, yn)
            self.pipeline = gs.best_estimator_
        else:
            self.pipeline.fit(Xn, yn)

        # guardar referencia directa al árbol entrenado
        self.clf_ = self.pipeline.named_steps["clf"]

        # capturar nombres finales de features (post-OHE) para print_tree
        pre: ColumnTransformer = self.pipeline.named_steps["pre"]
        ohe = None
        try:
            ohe = pre.named_transformers_["cat"].named_steps["ohe"]
        except Exception:
            ohe = None
        ohe_names = []
        if ohe is not None and hasattr(ohe, "get_feature_names_out"):
            cat_cols = [self.feature_names_[i] for i, t in enumerate(self.feature_types_) if t == "categorical"]
            ohe_names = list(ohe.get_feature_names_out(cat_cols))
        num_names = [self.feature_names_[i] for i, t in enumerate(self.feature_types_) if t == "numeric"]
        self._ohe_feature_names_ = num_names + ohe_names

    def predict_batch(self, X: List[List[Any]]) -> List[str]:
        if self.pipeline is None:
            raise RuntimeError("Model is not trained.")
        Xn, _ = _normalize_table(X, [None] * len(X))
        preds = self.pipeline.predict(Xn)
        return list(preds)

    def predict_proba(self, X: List[List[Any]]) -> List[Dict[str, float]]:
        if self.pipeline is None:
            raise RuntimeError("Model is not trained.")
        Xn, _ = _normalize_table(X, [None] * len(X))
        proba = self.pipeline.predict_proba(Xn)
        classes = list(self.pipeline.classes_)
        out: List[Dict[str, float]] = []
        for p in proba:
            out.append({c: float(pp) for c, pp in zip(classes, p)})
        return out

    def print_tree(self) -> str:
        """Exporta el árbol como texto usando las features post-transformación.
        Nota: export_text requiere las columnas **después** del preprocesamiento.
        """
        if self.clf_ is None:
            return "<Modelo no entrenado>"
        feature_names = self._ohe_feature_names_
        try:
            txt = export_text(self.clf_, feature_names=feature_names)
        except Exception:
            txt = export_text(self.clf_)
        return txt
