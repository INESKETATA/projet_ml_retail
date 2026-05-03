from pathlib import Path
import sys
import pandas as pd

from flask import Flask, request, jsonify
from flask_cors import CORS

# ----------------------------------------------------------
# Résolution des imports du projet
# ----------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

for p in [str(PROJECT_ROOT), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from predict import predict_one, predict_from_dataframe
from utils import get_important_features

# ----------------------------------------------------------
# Charger les données d'entraînement et identifier les features importantes
# ----------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_DATA_PATH = DATA_DIR / "retail_customers_COMPLETE_CATEGORICAL.csv"

df_train = pd.read_csv(TRAIN_DATA_PATH)
important_features_dict = get_important_features(df_train, target_col="Churn", correlation_threshold=0.20)
IMPORTANT_NUMERIC = important_features_dict['numeric_features']
IMPORTANT_CATEGORICAL = important_features_dict['categorical_features']
ALL_IMPORTANT = IMPORTANT_NUMERIC + IMPORTANT_CATEGORICAL


# ----------------------------------------------------------
# App Flask
# ----------------------------------------------------------
app = Flask(__name__)

# Mets ici les URLs de ton front
ALLOWED_ORIGINS = [
    "http://localhost:5173",   # Vite
    "http://127.0.0.1:5173",
    "http://localhost:3000",   # React éventuel
    "http://127.0.0.1:3000",
]

CORS(
    app,
    resources={r"/api/*": {"origins": ALLOWED_ORIGINS}},
    supports_credentials=True
)


# ----------------------------------------------------------
# Routes
# ----------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "API Flask opérationnelle"
    }), 200


@app.route("/api/features", methods=["GET"])
def get_features():
    """
    Retourne la liste des features importantes (forte corrélation avec le churn)
    """
    return jsonify({
        "success": True,
        "important_numeric_features": IMPORTANT_NUMERIC,
        "important_categorical_features": IMPORTANT_CATEGORICAL,
        "all_important_features": ALL_IMPORTANT,
        "total": {
            "numeric": len(IMPORTANT_NUMERIC),
            "categorical": len(IMPORTANT_CATEGORICAL),
            "total": len(ALL_IMPORTANT)
        }
    }), 200


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Prédiction du churn client.
    Accepte uniquement les features importantes.
    Les autres features sont ignorées ou remplies avec des valeurs par défaut.
    """
    try:
        data = request.get_json()

        if data is None:
            return jsonify({
                "success": False,
                "error": "Aucune donnée JSON reçue."
            }), 400

        # Cas 1 : une seule ligne (dict)
        if isinstance(data, dict):
            # Nettoyer les données : garder seulement les features importantes
            cleaned_data = {k: v for k, v in data.items() if k in ALL_IMPORTANT or k == "Churn"}
            
            # Remplir les données avec les valeurs par défaut
            for col in df_train.columns:
                if col not in cleaned_data and col != 'Churn':
                    if df_train[col].dtype in ['int64', 'float64']:
                        cleaned_data[col] = 0
                    else:
                        cleaned_data[col] = "Unknown"
            
            result = predict_one(cleaned_data)
            
            return jsonify({
                "success": True,
                "prediction": result,
                "input_features_used": list(cleaned_data.keys()),
                "note": "Seules les features importantes ont été utilisées pour la prédiction."
            }), 200

        # Cas 2 : plusieurs lignes (list)
        if isinstance(data, list):
            # Nettoyer chaque ligne
            cleaned_data_list = []
            for item in data:
                if isinstance(item, dict):
                    cleaned_item = {k: v for k, v in item.items() if k in ALL_IMPORTANT or k == "Churn"}
                    
                    # Remplir avec valeurs par défaut
                    for col in df_train.columns:
                        if col not in cleaned_item and col != 'Churn':
                            if df_train[col].dtype in ['int64', 'float64']:
                                cleaned_item[col] = 0
                            else:
                                cleaned_item[col] = "Unknown"
                    
                    cleaned_data_list.append(cleaned_item)
            
            result_df = predict_from_dataframe(cleaned_data_list)
            
            return jsonify({
                "success": True,
                "predictions": result_df.to_dict(orient="records"),
                "count": len(result_df),
                "note": "Seules les features importantes ont été utilisées pour les prédictions."
            }), 200

        return jsonify({
            "success": False,
            "error": "Le format des données doit être un objet JSON ou une liste d'objets JSON."
        }), 400

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ----------------------------------------------------------
# Lancement local
# ----------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)