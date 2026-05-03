from pathlib import Path
import joblib
import pandas as pd
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

for p in [str(CURRENT_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from preprocessing import Parseur, RetailPreprocessor, FeatureEngineer
from utils import CategoricalEncoder, CollinearityReducer, FeatureScaler, PCAReducer



# ==========================================================
# Chemins
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_PATH = BASE_DIR / "data" / "raw" / "retail_customers_COMPLETE_CATEGORICAL.csv"
TRAIN_TEST_DIR = BASE_DIR / "data" / "train_test"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

TRAIN_TEST_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Chargement des données
# ==========================================================
df = pd.read_csv(RAW_PATH)

X = df.drop(columns=["Churn"])
y = df["Churn"]


# ==========================================================
# Split train / test
# ==========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train.to_csv(TRAIN_TEST_DIR / "X_train.csv", index=False)
X_test.to_csv(TRAIN_TEST_DIR / "X_test.csv", index=False)
y_train.to_csv(TRAIN_TEST_DIR / "y_train.csv", index=False)
y_test.to_csv(TRAIN_TEST_DIR / "y_test.csv", index=False)

print("Split terminé et fichiers sauvegardés dans data/train_test/")


# ==========================================================
# Pipeline de preprocessing
# ==========================================================
parseur = Parseur(
    registration_col="RegistrationDate",
    ip_col="LastLoginIP",
    drop_original_date=False,
    drop_original_ip=False
)

preprocessor = RetailPreprocessor(n_neighbors=5)
feature_engineer = FeatureEngineer()
encoder = CategoricalEncoder()
reducer = CollinearityReducer(corr_threshold=0.8)
scaler = FeatureScaler()

# Si tu veux garder la PCA, mets USE_PCA = True
USE_PCA = True
pca_reducer = PCAReducer(n_components=17) if USE_PCA else None


# 1) Parseur
X_train_parsed = parseur.fit_transform(X_train)
X_test_parsed = parseur.transform(X_test)

# 2) Processing
X_train_clean = preprocessor.fit_transform(X_train_parsed)
X_test_clean = preprocessor.transform(X_test_parsed)

# 3) Feature engineering
X_train_fe = feature_engineer.fit_transform(X_train_clean)
X_test_fe = feature_engineer.transform(X_test_clean)

# 4) Encoding
X_train_enc = encoder.fit_transform(X_train_fe, y_train)
X_test_enc = encoder.transform(X_test_fe)

# 5) Collinearity reduction
X_train_red = reducer.fit_transform(X_train_enc)
X_test_red = reducer.transform(X_test_enc)

# 6) Suppression des colonnes brutes restantes
cols_to_drop = ["RegistrationDate", "LastLoginIP"]
X_train_red = X_train_red.drop(columns=cols_to_drop, errors="ignore")
X_test_red = X_test_red.drop(columns=cols_to_drop, errors="ignore")

# 7) Scaling
X_train_scaled = scaler.fit_transform(X_train_red)
X_test_scaled = scaler.transform(X_test_red)

# 8) PCA éventuelle
if USE_PCA:
    X_train_final = pca_reducer.fit_transform(X_train_scaled)
    X_test_final = pca_reducer.transform(X_test_scaled)
else:
    X_train_final = X_train_scaled
    X_test_final = X_test_scaled


# ==========================================================
# Sauvegarde des données processed
# ==========================================================
X_train_final.to_csv(PROCESSED_DIR / "X_train_processed.csv", index=False)
X_test_final.to_csv(PROCESSED_DIR / "X_test_processed.csv", index=False)

print("Données transformées sauvegardées dans data/processed/")


# ==========================================================
# Entraînement du modèle final SVM
# Hyperparamètres retenus :
# {'C': 10, 'class_weight': None, 'gamma': 0.01, 'kernel': 'rbf'}
# ==========================================================
best_svm = SVC(
    C=10,
    class_weight=None,
    gamma=0.01,
    kernel="rbf",
    random_state=42
)

best_svm.fit(X_train_final, y_train)

y_pred = best_svm.predict(X_test_final)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Résultats SVM ")
print("Accuracy :", acc)
print("Precision:", prec)
print("Recall   :", rec)
print("F1-score :", f1)
print("Classification report :")
print(classification_report(y_test, y_pred))


# ==========================================================
# Sauvegarde du pipeline + modèle
# ==========================================================
joblib.dump(parseur, MODELS_DIR / "parseur.joblib")
joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
joblib.dump(feature_engineer, MODELS_DIR / "feature_engineer.joblib")
joblib.dump(encoder, MODELS_DIR / "encoder.joblib")
joblib.dump(reducer, MODELS_DIR / "reducer.joblib")
joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

if USE_PCA:
    joblib.dump(pca_reducer, MODELS_DIR / "pca_reducer.joblib")

joblib.dump(best_svm, MODELS_DIR / "best_svm_model_02_Pipeline.joblib")

print("\nModèle et objets du pipeline sauvegardés dans models/")