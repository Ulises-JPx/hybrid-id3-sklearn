"""
@author Ulises Jaramillo Portilla | A01798380 | Ulises-JPx

This file implements a Decision Tree classifier using the ID3 algorithm logic,
leveraging scikit-learn's DecisionTreeClassifier with entropy criterion to align
with information gain. The class DecisionTreeID3Sklearn provides robust data
preprocessing, including normalization of input values, handling of missing data
through imputation (median for numeric, constant for categorical), and encoding
of categorical features using OneHotEncoder. It supports optional hyperparameter
optimization via GridSearchCV.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import GridSearchCV

def clean_token(value: Any) -> Optional[str]:
    """
    Cleans and normalizes a single token from the dataset.

    Parameters:
        value (Any): The input value to clean.

    Returns:
        Optional[str]: The cleaned string, or None if empty or invalid.
    """
    if value is None:
        return None
    string_value = str(value).strip()
    # Remove surrounding quotes if present
    if len(string_value) >= 2 and ((string_value[0] == string_value[-1] == "'") or (string_value[0] == string_value[-1] == '"')):
        string_value = string_value[1:-1].strip()
    # Return None for empty strings
    return string_value if string_value != "" else None

def normalize_table(features: List[List[Any]], targets: List[Any]) -> tuple[list[list[Any]], list[Any]]:
    """
    Normalizes the entire feature and target tables by cleaning each token.

    Parameters:
        features (List[List[Any]]): The input feature matrix.
        targets (List[Any]): The target labels.

    Returns:
        tuple[list[list[Any]], list[Any]]: Normalized features and targets.
    """
    normalized_features = [[clean_token(value) for value in row] for row in features]
    normalized_targets = [clean_token(target) for target in targets]
    return normalized_features, normalized_targets

def detect_feature_types(features: List[List[Any]]) -> List[str]:
    """
    Detects the type of each feature column (numeric or categorical).

    Parameters:
        features (List[List[Any]]): The input feature matrix.

    Returns:
        List[str]: List of feature types ("numeric" or "categorical").
    """
    feature_types: List[str] = []
    # Iterate over columns to determine type
    for column in zip(*features):
        is_numeric = True
        for value in column:
            cleaned_value = clean_token(value)
            if cleaned_value is None:
                continue
            try:
                float(cleaned_value)
            except ValueError:
                is_numeric = False
                break
        feature_types.append("numeric" if is_numeric else "categorical")
    return feature_types

class DecisionTreeID3Sklearn:
    """
    DecisionTreeID3Sklearn implements a decision tree classifier using scikit-learn,
    with preprocessing and feature handling aligned to the ID3 algorithm.

    Attributes:
        criterion (str): Splitting criterion for the tree ("entropy" for ID3).
        random_state (Optional[int]): Random seed for reproducibility.
        pipeline (Optional[Pipeline]): The complete scikit-learn pipeline.
        clf_ (Optional[DecisionTreeClassifier]): Reference to the trained classifier.
        feature_names_ (List[str]): Original feature names.
        feature_types_ (List[str]): Detected feature types.
        _ohe_feature_names_ (Optional[List[str]]): Feature names after OneHotEncoding.
    """

    def __init__(
        self,
        criterion: str = "entropy",
        random_state: Optional[int] = 42,
        ccp_alpha: float = 0.0
    ):
        """
        Initializes the DecisionTreeID3Sklearn instance.

        Parameters:
            criterion (str): Splitting criterion ("entropy" for information gain).
            random_state (Optional[int]): Random seed for reproducibility.
        """
        self.criterion = criterion
        self.random_state = random_state
        self.ccp_alpha = ccp_alpha
        self.pipeline: Optional[Pipeline] = None
        self.clf_: Optional[DecisionTreeClassifier] = None
        self.feature_names_: List[str] = []
        self.feature_types_: List[str] = []
        self._ohe_feature_names_: Optional[List[str]] = None

    def build_pipeline(self, features: List[List[Any]], feature_names: List[str]) -> Pipeline:
        """
        Constructs the preprocessing and classification pipeline.

        Parameters:
            features (List[List[Any]]): The input feature matrix.
            feature_names (List[str]): List of feature names.

        Returns:
            Pipeline: The constructed scikit-learn pipeline.
        """
        # Detect feature types and store feature names
        self.feature_types_ = detect_feature_types(features)
        self.feature_names_ = list(feature_names)

        # Identify indices for numeric and categorical features
        numeric_indices = [i for i, feature_type in enumerate(self.feature_types_) if feature_type == "numeric"]
        categorical_indices = [i for i, feature_type in enumerate(self.feature_types_) if feature_type == "categorical"]

        # Define numeric transformer: impute missing values with median
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        # Define categorical transformer: impute missing with constant, then OneHotEncode
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        # Combine transformers using ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_indices),
                ("cat", categorical_transformer, categorical_indices),
            ],
            remainder="drop",
        )

        # Initialize DecisionTreeClassifier with specified criterion and random state
        classifier = DecisionTreeClassifier(
            criterion=self.criterion,
            random_state=self.random_state,
            ccp_alpha=self.ccp_alpha
        )

        # Build the complete pipeline: preprocessing followed by classification
        pipeline = Pipeline(
            steps=[
                ("pre", preprocessor),
                ("clf", classifier),
            ]
        )
        return pipeline

    def train(
        self,
        features: List[List[Any]],
        targets: List[str],
        feature_names: List[str],
        use_gridsearch: bool = False,
    ) -> None:
        """
        Trains the decision tree model on the provided data.

        Parameters:
            features (List[List[Any]]): The input feature matrix.
            targets (List[str]): The target labels.
            feature_names (List[str]): List of feature names.
            use_gridsearch (bool): Whether to perform hyperparameter optimization.
        """
        # Normalize features and targets
        normalized_features, normalized_targets = normalize_table(features, targets)
        # Build the pipeline for preprocessing and classification
        self.pipeline = self.build_pipeline(normalized_features, feature_names)

        if use_gridsearch:
            # Define hyperparameter grid for GridSearchCV
            param_grid = { 
                "clf__max_depth": [None, 3, 5, 7, 10, 15, 20], 
                "clf__min_samples_leaf": [1, 2, 4, 8, 16], 
                "clf__min_samples_split": [2, 4, 8, 16], 
                "clf__ccp_alpha": [0.0, 1e-5, 1e-4, 1e-3, 1e-2], 
                "clf__max_features": [None, "sqrt", "log2"], 
            }
            # Ajustar cv dinámicamente según la clase menos poblada
            import numpy as np
            from collections import Counter
            class_counts = Counter(normalized_targets)
            min_class_count = min(class_counts.values())
            n_splits = min(5, min_class_count)
            if n_splits < 2:
                import warnings
                warnings.warn("No se puede usar GridSearchCV: la clase menos poblada tiene menos de 2 muestras. Se entrenará sin búsqueda de hiperparámetros.")
                self.pipeline.fit(normalized_features, normalized_targets)
            else:
                grid_search = GridSearchCV(
                    self.pipeline,
                    param_grid=param_grid,
                    cv=n_splits,
                    n_jobs=-1,
                    scoring="accuracy",
                    refit=True,
                )
                grid_search.fit(normalized_features, normalized_targets)
                # Use the best estimator found by grid search
                self.pipeline = grid_search.best_estimator_
        else:
            # Fit the pipeline directly without hyperparameter search
            self.pipeline.fit(normalized_features, normalized_targets)

        # Store reference to the trained classifier for later use
        self.clf_ = self.pipeline.named_steps["clf"]

        # Extract final feature names after preprocessing (including OHE)
        preprocessor: ColumnTransformer = self.pipeline.named_steps["pre"]
        one_hot_encoder = None
        try:
            one_hot_encoder = preprocessor.named_transformers_["cat"].named_steps["ohe"]
        except Exception:
            one_hot_encoder = None
        one_hot_feature_names = []
        if one_hot_encoder is not None and hasattr(one_hot_encoder, "get_feature_names_out"):
            categorical_columns = [self.feature_names_[i] for i, feature_type in enumerate(self.feature_types_) if feature_type == "categorical"]
            one_hot_feature_names = list(one_hot_encoder.get_feature_names_out(categorical_columns))
        numeric_feature_names = [self.feature_names_[i] for i, feature_type in enumerate(self.feature_types_) if feature_type == "numeric"]
        self._ohe_feature_names_ = numeric_feature_names + one_hot_feature_names

    def predict_batch(self, features: List[List[Any]]) -> List[str]:
        """
        Predicts class labels for a batch of input samples.

        Parameters:
            features (List[List[Any]]): The input feature matrix.

        Returns:
            List[str]: Predicted class labels for each sample.
        """
        if self.pipeline is None:
            raise RuntimeError("Model is not trained.")
        # Normalize input features
        normalized_features, _ = normalize_table(features, [None] * len(features))
        # Predict using the trained pipeline
        predictions = self.pipeline.predict(normalized_features)
        return list(predictions)

    def predict_proba(self, features: List[List[Any]]) -> List[Dict[str, float]]:
        """
        Predicts class probabilities for a batch of input samples.

        Parameters:
            features (List[List[Any]]): The input feature matrix.

        Returns:
            List[Dict[str, float]]: List of dictionaries mapping class labels to probabilities.
        """
        if self.pipeline is None:
            raise RuntimeError("Model is not trained.")
        # Normalize input features
        normalized_features, _ = normalize_table(features, [None] * len(features))
        # Get probability predictions from the pipeline
        probabilities = self.pipeline.predict_proba(normalized_features)
        class_labels = list(self.pipeline.classes_)
        output: List[Dict[str, float]] = []
        # Map each probability vector to a dictionary of class: probability
        for probability_vector in probabilities:
            output.append({class_label: float(prob) for class_label, prob in zip(class_labels, probability_vector)})
        return output

    def print_tree(self) -> str:
        """
        Exports the trained decision tree as a text representation.

        Returns:
            str: Textual representation of the decision tree structure.
        """
        if self.clf_ is None:
            return "<Model not trained>"
        feature_names = self._ohe_feature_names_
        try:
            # Attempt to export tree with feature names after preprocessing
            tree_text = export_text(self.clf_, feature_names=feature_names)
        except Exception:
            # Fallback to exporting without feature names
            tree_text = export_text(self.clf_)
        return tree_text