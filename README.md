<<<<<<< HEAD
# 🛍️ Retail Customer Behavior Analysis & Churn Prediction

## 📌 Overview

Ce projet est une application complète de **Machine Learning appliqué au retail (e-commerce)** permettant :

* 📊 d’analyser le comportement des clients
* 🎯 de prédire le **churn (départ client)**
* 🧠 d’exploiter un pipeline ML complet (préprocessing → feature engineering → modélisation → déploiement)
* 🌐 de proposer une **interface interactive avec Streamlit**

---

## 🚀 Features

* 🔍 Analyse de données clients (52 features)
* 🧹 Préprocessing avancé :

  * Imputation des valeurs manquantes
  * Traitement des outliers (IQR)
  * Parsing des dates (`RegistrationDate`)
  * Feature engineering (`LastLoginIP`, ratios, etc.)
* 🧠 Modèle de Machine Learning :

  * **SVM (Support Vector Machine)**
* ⚙️ Pipeline complet :

  * Parseur → Preprocessing → Feature Engineering → Encoding → Réduction → Scaling → PCA → Modèle
* 🌐 Interface utilisateur moderne avec **Streamlit**
* 📦 Modèles sauvegardés avec `joblib`

---

## 🏗️ Project Structure

```
projet_ml_retail/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── train_test/
│
├── notebooks/
│
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│   └── utils.py
│
├── models/
│   ├── parseur.joblib
│   ├── preprocessor.joblib
│   ├── feature_engineer.joblib
│   ├── encoder.joblib
│   ├── reducer.joblib
│   ├── scaler.joblib
│   ├── pca_reducer.joblib
│   └── best_svm_model_02_Pipeline.joblib
│
├── app/
│   ├── streamlit_app.py   # Interface principale
│   └── app.py             # (ancienne version Flask - optionnelle)
│
├── reports/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### 1️⃣ Cloner le projet

```bash
git clone <repo_url>
cd projet_ml_retail
```

### 2️⃣ Créer un environnement virtuel

```bash
python -m venv venv
```

### 3️⃣ Activer l’environnement

Windows :

```bash
venv\Scripts\activate
```

### 4️⃣ Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## ▶️ Lancer l’application

### 🟢 Backend / ML

```bash
.\venv\Scripts\python.exe .\app\app.py
```

📌 Port backend : **5000**

---

### 🟣 Frontend (Streamlit)

```bash
streamlit run app/streamlit_app.py
```

📌 L’application s’ouvre automatiquement dans le navigateur

---

## 🧠 Machine Learning Pipeline

```
Input Data
   ↓
Parseur
   ↓
RetailPreprocessor
   ↓
FeatureEngineer
   ↓
CategoricalEncoder
   ↓
CollinearityReducer
   ↓
FeatureScaler
   ↓
PCAReducer (optionnel)
   ↓
SVM Model
   ↓
Prediction
```

---

## 📊 Features utilisées

Le dataset contient **52 features** réparties en :

### 🔢 Numériques

* Recency
* Frequency
* MonetaryTotal / Avg / Std
* Quantity metrics
* CustomerTenure
* AvgDaysBetweenPurchases
* etc.

### 🏷️ Catégorielles

* RFMSegment
* AgeCategory
* CustomerType
* LoyaltyLevel
* Region
* Gender
* etc.

### 🔄 Features transformées

* `RegistrationDate` → parsing
* `LastLoginIP` → feature engineering
* `NewsletterSubscribed` → supprimée

---

## 🌐 Interface Streamlit

L'application permet :

* 📝 Saisie complète des données client
* ⚠️ Validation des champs obligatoires
* 🔮 Prédiction en temps réel
* 📊 Affichage du résultat
* 📋 Visualisation des données envoyées

---

## 🧪 Exemple d’utilisation

1. Remplir tous les champs
2. Cliquer sur **"Lancer la prédiction"**
3. Résultat affiché :

   * ✅ Client fidèle
   * ⚠️ Client en churn

---

## 📈 Objectifs du projet

* Améliorer la **rétention client**
* Optimiser les stratégies marketing
* Identifier les clients à risque
* Construire un pipeline ML robuste

---

## 🧰 Technologies utilisées

* Python 🐍
* Pandas / NumPy
* Scikit-learn
* Streamlit
* Joblib

---

## 📌 Améliorations possibles

* 📊 Dashboard avancé (Plotly / Power BI)
* ☁️ Déploiement cloud (AWS / Azure)
* 📉 Monitoring du modèle
