from pathlib import Path
import sys
import joblib
import pandas as pd

# ----------------------------------------------------------
# Résolution des imports locaux
# ----------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

for p in [str(CURRENT_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Important pour que joblib puisse recharger les objets correctement
from preprocessing import Parseur, RetailPreprocessor, FeatureEngineer
from utils import CategoricalEncoder, CollinearityReducer, FeatureScaler, PCAReducer


# ----------------------------------------------------------
# Chemins
# ----------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


# ----------------------------------------------------------
# Chargement des objets sauvegardés
# ----------------------------------------------------------
parseur = joblib.load(MODELS_DIR / "parseur.joblib")
preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
feature_engineer = joblib.load(MODELS_DIR / "feature_engineer.joblib")
encoder = joblib.load(MODELS_DIR / "encoder.joblib")
reducer = joblib.load(MODELS_DIR / "reducer.joblib")
scaler = joblib.load(MODELS_DIR / "scaler.joblib")
model = joblib.load(MODELS_DIR / "best_svm_model_02_Pipeline.joblib")

pca_path = MODELS_DIR / "pca_reducer.joblib"
use_pca = pca_path.exists()
pca_reducer = joblib.load(pca_path) if use_pca else None


# ----------------------------------------------------------
# Outils
# ----------------------------------------------------------
def _to_dataframe(X_new):
    if isinstance(X_new, pd.DataFrame):
        return X_new.copy()
    if isinstance(X_new, dict):
        return pd.DataFrame([X_new])
    if isinstance(X_new, list):
        return pd.DataFrame(X_new)
    raise TypeError("X_new doit être un DataFrame, un dict ou une liste de dicts.")


# ----------------------------------------------------------
# Pipeline complet de prédiction
# ----------------------------------------------------------
def preprocess_for_prediction(X_new):
    X_new = _to_dataframe(X_new)

    # Si jamais la cible est présente, on la retire
    if "Churn" in X_new.columns:
        X_new = X_new.drop(columns=["Churn"])

    # 1) Parseur
    X_new = parseur.transform(X_new)

    # 2) Processing
    X_new = preprocessor.transform(X_new)

    # 3) Feature engineering
    X_new = feature_engineer.transform(X_new)

    # 4) Encoding
    X_new = encoder.transform(X_new)

    # 5) Réduction de colinéarité
    X_new = reducer.transform(X_new)

    # 6) Suppression des colonnes brutes restantes si présentes
    X_new = X_new.drop(columns=["RegistrationDate", "LastLoginIP"], errors="ignore")

    # 7) Scaling
    X_new = scaler.transform(X_new)

    # 8) PCA si utilisée
    if use_pca:
        X_new = pca_reducer.transform(X_new)

    return X_new


def predict_from_dataframe(X_new):
    X_processed = preprocess_for_prediction(X_new)
    predictions = model.predict(X_processed)

    result = _to_dataframe(X_new)
    if "Churn" in result.columns:
        result = result.drop(columns=["Churn"])

    result["Prediction"] = predictions
    return result


def predict_one(record: dict):
    result_df = predict_from_dataframe(record)
    return result_df.iloc[0].to_dict()