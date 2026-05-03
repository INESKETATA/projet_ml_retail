from pathlib import Path
import sys
import pandas as pd
import streamlit as st

# ----------------------------------------------------------
# Résolution des imports du projet
# ----------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

for p in [str(PROJECT_ROOT), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from predict import predict_one

# ----------------------------------------------------------
# Charger les données d'entraînement
# ----------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_DATA_PATH = DATA_DIR / "retail_customers_COMPLETE_CATEGORICAL.csv"

@st.cache_data
def load_training_data():
    return pd.read_csv(TRAIN_DATA_PATH)

df_train = load_training_data()

# ----------------------------------------------------------
# Configuration page
# ----------------------------------------------------------
st.set_page_config(
    page_title="Retail ML App",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------
# CSS custom
# ----------------------------------------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #0b1020 0%, #111827 100%);
}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}

h1, h2, h3 {
    color: #f8fafc !important;
}

p, label, div, span {
    color: #e5e7eb;
}

.section-box {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 18px;
    margin-bottom: 18px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.18);
}

.result-box {
    background: linear-gradient(90deg, rgba(37,99,235,0.18), rgba(124,58,237,0.18));
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 18px;
    padding: 18px;
    margin-top: 10px;
}

.stButton > button {
    width: 100%;
    border: none;
    border-radius: 12px;
    padding: 0.8rem 1rem;
    font-size: 1rem;
    font-weight: 700;
    color: white;
    background: linear-gradient(90deg, #2563eb, #7c3aed);
}

.stButton > button:hover {
    color: white;
    opacity: 0.95;
}

.small-note {
    color: #cbd5e1;
    font-size: 0.92rem;
}

[data-testid="stSidebar"] {
    background: #1e293b;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: #f1f5f9 !important;
}

/* Selectbox - fond blanc texte noir */
div[data-baseweb="select"] {
    background-color: #ffffff !important;
}

div[data-baseweb="select"] * {
    color: #0f172a !important;
    background-color: #ffffff !important;
}

/* Dropdown ouvert */
div[data-baseweb="popover"] * {
    color: #0f172a !important;
    background-color: #ffffff !important;
}

div[data-baseweb="popover"] li:hover {
    background-color: #e2e8f0 !important;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
def required_text(label: str, key: str):
    return st.text_input(label, value="", key=key)

def required_select(label: str, options: list, key: str):
    return st.selectbox(label, [""] + options, index=0, key=key)

def parse_float(value: str, field_name: str):
    value = value.strip().replace(",", ".")
    if value == "":
        raise ValueError(f"Le champ '{field_name}' est obligatoire.")
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"Le champ '{field_name}' doit être un nombre.")

def parse_choice(value: str, field_name: str):
    if value == "":
        raise ValueError(f"Le champ '{field_name}' est obligatoire.")
    return value


# ----------------------------------------------------------
# Header
# ----------------------------------------------------------
st.markdown("""
<div class="section-box">
    <h1>🛍️ Retail Customer Churn Prediction</h1>
    <p class="small-note">
        Interface intelligente pour prédire le churn client en utilisant les features les plus importantes.
    </p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# Sidebar
# ----------------------------------------------------------
with st.sidebar:
    st.title("📌 Menu")
    st.write("Saisir les features importantes pour prédire le churn.")
    st.info(
        "🔍 **Features utilisées:**\n\n"
        "• **Numériques (3):** Recency, CustomerTenureDays, PreferredMonth\n\n"
        "• **Catégorielles (2):** ChurnRiskCategory, LoyaltyLevel"
    )

# ----------------------------------------------------------
# Formulaire - Features réduites
# ----------------------------------------------------------
with st.form("retail_prediction_form"):

    # ---- Numériques ----
    st.markdown("## 🔢 Features numériques importantes")
    col1, col2, col3 = st.columns(3)

    numeric_values = {}
    with col1:
        numeric_values["Recency"] = required_text("Recency (jours)", "numeric_Recency")
    with col2:
        numeric_values["CustomerTenureDays"] = required_text("CustomerTenureDays (jours)", "numeric_CustomerTenureDays")
    with col3:
        numeric_values["PreferredMonth"] = required_text("PreferredMonth (1-12)", "numeric_PreferredMonth")

    # ---- Catégorielles ----
    st.markdown("## 🏷️ Features catégorielles importantes")
    col1, col2 = st.columns(2)

    categorical_values = {}
    with col1:
        categorical_values["ChurnRiskCategory"] = required_select(
            "ChurnRiskCategory",
            ["Faible", "Moyen", "Élevé", "Critique"],
            "cat_ChurnRiskCategory"
        )
    with col2:
        categorical_values["LoyaltyLevel"] = required_select(
            "LoyaltyLevel",
            ["Nouveau", "Jeune", "Établi", "Ancien", "Inconnu"],
            "cat_LoyaltyLevel"
        )

    # ---- Features supplémentaires obligatoires ----
    st.markdown("## 📝 Features supplémentaires")
    col1, col2, col3 = st.columns(3)
    with col1:
        categorical_values["Country"] = required_text("Country", "country")
    with col2:
        categorical_values["RegistrationDate"] = required_text("RegistrationDate (dd/mm/yyyy)", "registration_date")
    with col3:
        categorical_values["LastLoginIP"] = required_text("LastLoginIP", "last_login_ip")

    submitted = st.form_submit_button("🚀 Lancer la prédiction")


# ----------------------------------------------------------
# Prédiction
# ----------------------------------------------------------
if submitted:
    try:
        input_data = {}

        # Features numériques
        for feature, value in numeric_values.items():
            if value and value.strip():
                input_data[feature] = parse_float(value, feature)

        # Features catégorielles
        for feature, value in categorical_values.items():
            if value and value.strip():
                input_data[feature] = parse_choice(value, feature)

        # Valeurs par défaut pour toutes les autres features
        for col in df_train.columns:
            if col not in input_data and col != "Churn":
                if df_train[col].dtype in ["int64", "float64"]:
                    input_data[col] = 0
                else:
                    input_data[col] = "Unknown"

        result = predict_one(input_data)
        prediction_value = result.get("Prediction", "Inconnue")

        st.markdown("""
        <div class="result-box">
            <h2>✅ Résultat de prédiction</h2>
        </div>
        """, unsafe_allow_html=True)

        if prediction_value == 1:
            st.error("⚠️ Le modèle prédit : **CHURN** - Client à risque de départ ⚠️")
        else:
            st.success("🎉 Le modèle prédit : **NON-CHURN** - Client fidèle 🎉")

        st.markdown("### 📊 Résultats détaillés")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Recency", result.get("Recency", "-"))
        with col2:
            st.metric("CustomerTenureDays", result.get("CustomerTenureDays", "-"))
        with col3:
            st.metric("PreferredMonth", result.get("PreferredMonth", "-"))

        with st.expander("🔍 Voir toutes les données envoyées"):
            st.json(result)

    except Exception as e:
        st.error(f"❌ Erreur : {e}")