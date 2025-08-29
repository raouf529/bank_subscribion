from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
class AddDropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_drop, missing_features, addfearures=True):
        self.features_to_drop = features_to_drop if features_to_drop is not None else []
        self.missing_features = missing_features if missing_features is not None else []
        self.addfearures = addfearures
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X_transformed = X.copy()

        #add missing_col for features with most missing values
        if self.addfearures:
            for col in self.missing_features:
                if col in X.columns:
                    if X[col].dtype == 'object':
                        X_transformed[f"missing_{col}"] = X[col]=="unknown"
                    elif X[col].dtype in ['int64', 'float64']:
                        X_transformed[f"missing_{col}"] = X[col]==-1

        #drop not needed feature
        for feature in self.features_to_drop:
            if feature in X_transformed.columns:
                X_transformed.drop(feature, axis=1, inplace=True)

        return X_transformed