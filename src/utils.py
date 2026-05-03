import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class CategoricalEncoder:
    def __init__(self):
        self.one_hot_candidates = [
            "RFMSegment",
            "CustomerType",
            "FavoriteSeason",
            "Region",
            "ProductDiversity",
            "Gender",
            "AccountStatus",
        ]
        self.fitted_ = False

    # ----------------------------------------------------------
    # Outils
    # ----------------------------------------------------------
    def _to_dataframe(self, X):
        if isinstance(X, pd.DataFrame):
            return X.copy()
        raise TypeError("X doit être un DataFrame.")

    def _clean_text(self, x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        if s == "":
            return np.nan
        return s

    def _clean_series(self, s):
        return s.apply(self._clean_text)

    def _mode_or_default(self, s, default=0):
        s = s.dropna()
        if s.empty:
            return default
        return s.mode().iloc[0]

    def _validate_input_columns(self, X):
        input_set = set(X.columns)
        train_set = set(self.input_columns_)

        missing_cols = [col for col in self.input_columns_ if col not in input_set]
        extra_cols = [col for col in X.columns if col not in train_set]

        if missing_cols or extra_cols:
            parts = []
            if missing_cols:
                parts.append(f"Colonnes manquantes par rapport au train : {missing_cols}")
            if extra_cols:
                parts.append(f"Colonnes supplémentaires non vues au train : {extra_cols}")

            raise ValueError(
                "Les colonnes de X ne correspondent pas à celles vues au fit(). "
                + " | ".join(parts)
            )

    # ----------------------------------------------------------
    # Mappings ordinals
    # ----------------------------------------------------------
    def _build_ordinal_mappings(self, X):
        ordinal_mappings = {}

        if "AgeCategory" in X.columns:
            ordinal_mappings["AgeCategory"] = {
                "18-24": 0,
                "25-34": 1,
                "35-44": 2,
                "45-54": 3,
                "55-64": 4,
                "65+": 5,
    
            }

        if "SpendingCategory" in X.columns:
            ordinal_mappings["SpendingCategory"] = {
                "Low": 0,
                "Medium": 1,
                "High": 2,
                "VIP": 3,
            }

        if "PreferredTimeOfDay" in X.columns:
            ordinal_mappings["PreferredTimeOfDay"] = {
                "Matin": 0,
                "Midi": 1,
                "Après-midi": 2,
                "Apres-midi": 2,
                "Soir": 3,
                "Nuit": 4,
            }

        if "LoyaltyLevel" in X.columns:
            ordinal_mappings["LoyaltyLevel"] = {
                "Nouveau": 0,
                "Jeune": 1,
                "Établi": 2,
                "Etabli": 2,
                "Ancien": 3,
              
            }

        if "ChurnRiskCategory" in X.columns:
            ordinal_mappings["ChurnRiskCategory"] = {
                "Faible": 0,
                "Moyen": 1,
                "Élevé": 2,
                "Eleve": 2,
                "Critique": 3,
            }

        if "BasketSizeCategory" in X.columns:
            ordinal_mappings["BasketSizeCategory"] = {
                "Petit": 0,
                "Moyen": 1,
                "Grand": 2,
                
            }

        return ordinal_mappings

    # ----------------------------------------------------------
    # Transformation interne
    # ----------------------------------------------------------
    def _transform_internal(self, X, fit_mode=False):
        X = X.copy()
        X = X[self.input_columns_].copy()

        # Ordinal
        for col, mapping in self.ordinal_mappings_.items():
            s = self._clean_series(X[col])
            X[col] = s.map(mapping)
            X[col] = X[col].fillna(self.ordinal_fill_values_.get(col, 0))

        # Target Encoding pour Country
        if self.country_col_ is not None:
            s = self._clean_series(X[self.country_col_])
            X["Country_TE"] = s.map(self.country_target_map_)
            X["Country_TE"] = X["Country_TE"].fillna(self.country_global_mean_)
            X = X.drop(columns=[self.country_col_])

        # One-Hot
        for col, categories in self.one_hot_categories_.items():
            s = self._clean_series(X[col])

            for cat in categories:
                new_col = f"{col}__{cat}"
                X[new_col] = (s == cat).astype(int)

            X = X.drop(columns=[col])

        # Pendant fit : on ne réordonne pas encore avec self.output_columns_
        if not fit_mode:
            X = X[self.output_columns_].copy()

        return X

    # ----------------------------------------------------------
    # FIT
    # ----------------------------------------------------------
    def fit(self, X, y=None):
        X = self._to_dataframe(X)
        self.input_columns_ = X.columns.tolist()

        # Ordinal
        self.ordinal_mappings_ = self._build_ordinal_mappings(X)

        self.ordinal_fill_values_ = {}
        for col, mapping in self.ordinal_mappings_.items():
            s = self._clean_series(X[col]).map(mapping)
            self.ordinal_fill_values_[col] = self._mode_or_default(s, default=0)

        # One-Hot
        self.one_hot_cols_ = [c for c in self.one_hot_candidates if c in X.columns]
        self.one_hot_categories_ = {}

        for col in self.one_hot_cols_:
            s = self._clean_series(X[col]).dropna()
            self.one_hot_categories_[col] = sorted(s.unique().tolist())

        # Target Encoding pour Country
        self.country_col_ = "Country" if "Country" in X.columns else None
        self.country_target_map_ = {}
        self.country_global_mean_ = np.nan

        if self.country_col_ is not None:
            if y is None:
                raise ValueError("y_train est requis dans fit() pour le Target Encoding de Country.")

            y_series = pd.Series(y).reset_index(drop=True)
            if len(y_series) != len(X):
                raise ValueError("X et y doivent avoir la même longueur.")

            country_s = self._clean_series(X[self.country_col_]).reset_index(drop=True)
            df_te = pd.DataFrame({
                "Country": country_s,
                "target": pd.to_numeric(y_series, errors="coerce")
            })

            self.country_global_mean_ = df_te["target"].mean()
            self.country_target_map_ = df_te.groupby("Country")["target"].mean().to_dict()

        # Ici fit_mode=True
        X_encoded = self._transform_internal(X, fit_mode=True)
        self.output_columns_ = X_encoded.columns.tolist()

        self.fitted_ = True
        return self

    # ----------------------------------------------------------
    # TRANSFORM
    # ----------------------------------------------------------
    def transform(self, X):
        if not self.fitted_:
            raise RuntimeError("Le CategoricalEncoder doit être fit avant transform.")

        X = self._to_dataframe(X)
        self._validate_input_columns(X)

        X = self._transform_internal(X, fit_mode=False)
        return X

    # ----------------------------------------------------------
    # FIT_TRANSFORM
    # ----------------------------------------------------------
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)



class CollinearityReducer:
    def __init__(self, corr_threshold=0.8, protected_features=None):
        self.corr_threshold = corr_threshold
        self.protected_features = set(protected_features or [])
        self.fitted_ = False

    # ----------------------------------------------------------
    # Outils
    # ----------------------------------------------------------
    def _to_dataframe(self, X):
        if isinstance(X, pd.DataFrame):
            return X.copy()
        raise TypeError("X doit être un DataFrame.")

    def _validate_input_columns(self, X):
        input_set = set(X.columns)
        train_set = set(self.input_columns_)

        missing_cols = [col for col in self.input_columns_ if col not in input_set]
        extra_cols = [col for col in X.columns if col not in train_set]

        if missing_cols or extra_cols:
            parts = []
            if missing_cols:
                parts.append(f"Colonnes manquantes : {missing_cols}")
            if extra_cols:
                parts.append(f"Colonnes supplémentaires : {extra_cols}")

            raise ValueError(
                "Les colonnes de X ne correspondent pas à celles vues au fit(). "
                + " | ".join(parts)
            )

    def _get_numeric_df(self, X):
        return X.select_dtypes(include=[np.number]).copy()

        

    def _find_correlated_to_drop(self, corr_matrix):
        abs_corr = corr_matrix.abs()
        upper = abs_corr.where(np.triu(np.ones(abs_corr.shape), k=1).astype(bool))

        to_drop = set()
        reasons = []

        for col in upper.columns:
            high_corr_rows = upper.index[upper[col] > self.corr_threshold].tolist()

            for row_feature in high_corr_rows:
                col_feature = col
                corr_value = upper.loc[row_feature, col_feature]

                if row_feature in self.protected_features and col_feature not in self.protected_features:
                    to_drop.add(col_feature)
                    reasons.append((col_feature, row_feature, corr_value))

                elif col_feature in self.protected_features and row_feature not in self.protected_features:
                    to_drop.add(row_feature)
                    reasons.append((row_feature, col_feature, corr_value))

                elif row_feature in self.protected_features and col_feature in self.protected_features:
                    continue

                else:
                    # règle simple : on supprime la colonne de droite
                    to_drop.add(col_feature)
                    reasons.append((col_feature, row_feature, corr_value))

        # déduplication
        clean_reasons = []
        seen = set()
        for dropped, kept, corr_value in reasons:
            key = (dropped, kept)
            if key not in seen:
                seen.add(key)
                clean_reasons.append((dropped, kept, corr_value))

        return sorted(to_drop), clean_reasons

    # ----------------------------------------------------------
    # FIT
    # ----------------------------------------------------------
    def fit(self, X, y=None):
        X = self._to_dataframe(X)
        self.input_columns_ = X.columns.tolist()

        # Corrélation avant
        X_num = self._get_numeric_df(X)
        self.numeric_input_columns_ = X_num.columns.tolist()
        self.corr_before_ = X_num.corr(numeric_only=True)


        # Colonnes à supprimer
        self.dropped_columns_, self.drop_reasons_ = self._find_correlated_to_drop(self.corr_before_)

        # Colonnes finales
        self.output_columns_ = [c for c in self.input_columns_ if c not in self.dropped_columns_]
        self.retained_columns_ = self.output_columns_

        # Corrélation après
        X_after = X[self.output_columns_].copy()
        X_after_num = self._get_numeric_df(X_after)
        self.corr_after_ = X_after_num.corr(numeric_only=True)

        

        self.fitted_ = True
        return self

    # ----------------------------------------------------------
    # TRANSFORM
    # ----------------------------------------------------------
    def transform(self, X):
        if not self.fitted_:
            raise RuntimeError("Le CollinearityReducer doit être fit avant transform.")

        X = self._to_dataframe(X)
        self._validate_input_columns(X)

        X = X[self.input_columns_].copy()
        X = X[self.output_columns_].copy()
        return X

    # ----------------------------------------------------------
    # FIT_TRANSFORM
    # ----------------------------------------------------------
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

 



class FeatureScaler:
    def __init__(self, columns_to_scale=None):
        """
        columns_to_scale :
        - None  -> détection automatique des colonnes à scaler
        - liste -> colonnes explicitement fournies
        """
        self.columns_to_scale = columns_to_scale
        self.fitted_ = False

    # ----------------------------------------------------------
    # Outils
    # ----------------------------------------------------------
    def _to_dataframe(self, X):
        if isinstance(X, pd.DataFrame):
            return X.copy()
        raise TypeError("X doit être un DataFrame.")

    def _validate_input_columns(self, X):
        input_set = set(X.columns)
        train_set = set(self.input_columns_)

        missing_cols = [col for col in self.input_columns_ if col not in input_set]
        extra_cols = [col for col in X.columns if col not in train_set]

        if missing_cols or extra_cols:
            parts = []
            if missing_cols:
                parts.append(f"Colonnes manquantes : {missing_cols}")
            if extra_cols:
                parts.append(f"Colonnes supplémentaires : {extra_cols}")

            raise ValueError(
                "Les colonnes de X ne correspondent pas à celles vues au fit(). "
                + " | ".join(parts)
            )

    def _infer_columns_to_scale(self, X):
        """
        Détection automatique :
        - garde les colonnes numériques
        - exclut les colonnes one-hot (présence de '__')
        - exclut les colonnes IP dérivées
        - exclut les colonnes binaires 0/1
        """
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        excluded_cols = {
            "IsValidIP",
            "IsPrivateIP",
            "IPVersion",
        }

        cols_to_scale = []

        for col in numeric_cols:
            # exclure one-hot
            if "__" in col:
                continue

            # exclure certaines colonnes discrètes/binaries IP
            if col in excluded_cols:
                continue

            s = pd.to_numeric(X[col], errors="coerce").dropna()

            # exclure binaire 0/1
            unique_vals = set(s.unique().tolist())
            if unique_vals.issubset({0, 1}):
                continue

            cols_to_scale.append(col)

        return cols_to_scale

    # ----------------------------------------------------------
    # FIT
    # ----------------------------------------------------------
    def fit(self, X, y=None):
        X = self._to_dataframe(X)
        self.input_columns_ = X.columns.tolist()

        if self.columns_to_scale is None:
            self.columns_to_scale_ = self._infer_columns_to_scale(X)
        else:
            self.columns_to_scale_ = [col for col in self.columns_to_scale if col in X.columns]

        self.scaler_ = StandardScaler()
        self.scaler_.fit(X[self.columns_to_scale_])

        self.output_columns_ = self.input_columns_.copy()
        self.fitted_ = True
        return self

    # ----------------------------------------------------------
    # TRANSFORM
    # ----------------------------------------------------------
    def transform(self, X):
        if not self.fitted_:
            raise RuntimeError("Le FeatureScaler doit être fit avant transform.")

        X = self._to_dataframe(X)
        self._validate_input_columns(X)

        X = X[self.input_columns_].copy()

        if self.columns_to_scale_:
            X[self.columns_to_scale_] = self.scaler_.transform(X[self.columns_to_scale_])

        return X

    # ----------------------------------------------------------
    # FIT_TRANSFORM
    # ----------------------------------------------------------
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    # ----------------------------------------------------------
    # Résumé
    # ----------------------------------------------------------
    def summary(self):
        if not self.fitted_:
            raise RuntimeError("Le FeatureScaler doit être fit avant summary().")

        print("Colonnes scalées :")
        print(self.columns_to_scale_)


from sklearn.decomposition import PCA


class PCAReducer:
    def __init__(self, n_components=None, explained_variance_threshold=None):
        """
        n_components :
            - int : nombre fixe de composantes
            - None : utilisé seulement si explained_variance_threshold est donné

        explained_variance_threshold :
            - float entre 0 et 1, ex: 0.95 pour garder 95% de variance
            - None si on veut imposer n_components
        """
        self.n_components = n_components
        self.explained_variance_threshold = explained_variance_threshold
        self.fitted_ = False

    # ----------------------------------------------------------
    # Outils
    # ----------------------------------------------------------
    def _to_dataframe(self, X):
        if isinstance(X, pd.DataFrame):
            return X.copy()
        raise TypeError("X doit être un DataFrame.")

    def _validate_input_columns(self, X):
        input_set = set(X.columns)
        train_set = set(self.input_columns_)

        missing_cols = [col for col in self.input_columns_ if col not in input_set]
        extra_cols = [col for col in X.columns if col not in train_set]

        if missing_cols or extra_cols:
            parts = []
            if missing_cols:
                parts.append(f"Colonnes manquantes : {missing_cols}")
            if extra_cols:
                parts.append(f"Colonnes supplémentaires : {extra_cols}")

            raise ValueError(
                "Les colonnes de X ne correspondent pas à celles vues au fit(). "
                + " | ".join(parts)
            )

    def _build_output_dataframe(self, X_pca, index):
        cols = [f"PC{i+1}" for i in range(X_pca.shape[1])]
        return pd.DataFrame(X_pca, columns=cols, index=index)

    # ----------------------------------------------------------
    # FIT
    # ----------------------------------------------------------
    def fit(self, X, y=None):
        X = self._to_dataframe(X)
        self.input_columns_ = X.columns.tolist()

        if self.explained_variance_threshold is not None:
            self.pca_ = PCA(n_components=self.explained_variance_threshold)
        else:
            self.pca_ = PCA(n_components=self.n_components)

        self.pca_.fit(X)

        self.n_components_selected_ = self.pca_.n_components_
        self.explained_variance_ratio_ = self.pca_.explained_variance_ratio_
        self.cumulative_explained_variance_ = np.cumsum(self.explained_variance_ratio_)

        self.output_columns_ = [f"PC{i+1}" for i in range(self.n_components_selected_)]
        self.fitted_ = True
        return self

    # ----------------------------------------------------------
    # TRANSFORM
    # ----------------------------------------------------------
    def transform(self, X):
        if not self.fitted_:
            raise RuntimeError("Le PCAReducer doit être fit avant transform.")

        X = self._to_dataframe(X)
        self._validate_input_columns(X)

        X = X[self.input_columns_].copy()
        X_pca = self.pca_.transform(X)

        return self._build_output_dataframe(X_pca, X.index)

    # ----------------------------------------------------------
    # FIT_TRANSFORM
    # ----------------------------------------------------------
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    # ----------------------------------------------------------
    # Résumé
    # ----------------------------------------------------------
    def summary(self):
        if not self.fitted_:
            raise RuntimeError("Le PCAReducer doit être fit avant summary().")

        summary_df = pd.DataFrame({
            "Composante": self.output_columns_,
            "Variance expliquée": self.explained_variance_ratio_,
            "Variance cumulée": self.cumulative_explained_variance_
        })
        return summary_df


# ===========================================================
# Fonction pour identifier les features importantes
# ===========================================================
def get_important_features(df, target_col="Churn", correlation_threshold=0.1):
    """
    Identifie UNIQUEMENT les features avec une forte corrélation avec la cible.
    
    Parameters:
    - df : DataFrame avec toutes les colonnes
    - target_col : nom de la colonne cible
    - correlation_threshold : seuil de corrélation (en valeur absolue)
    
    Returns:
    - dict avec 'numeric_features' et 'categorical_features'
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Retirer la cible
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    # Calculer corrélation pour features numériques
    important_numeric = []
    if len(numeric_cols) > 0 and target_col in df.columns:
        corr = df[numeric_cols + [target_col]].corr()[target_col].drop(target_col)
        important_numeric = corr[abs(corr) > correlation_threshold].index.tolist()
        important_numeric = sorted(important_numeric, key=lambda x: abs(corr[x]), reverse=True)
    
    # Pour les features catégorielles : utiliser Cramér's V ou point-biserial
    # Pour simplifier, on va calculer la corrélation encodée
    important_categorical = []
    
    try:
        from sklearn.preprocessing import LabelEncoder
        
        # Créer une copie et encoder les catégories
        df_encoded = df.copy()
        
        # Encoder la cible si elle est catégorique
        if target_col in categorical_cols or df[target_col].dtype == 'object':
            le_target = LabelEncoder()
            target_encoded = le_target.fit_transform(df_encoded[target_col].astype(str))
        else:
            target_encoded = df_encoded[target_col].values
        
        # Encoder chaque feature catégorique et calculer corrélation
        for col in categorical_cols:
            try:
                le = LabelEncoder()
                col_encoded = le.fit_transform(df_encoded[col].astype(str).fillna("Missing"))
                
                # Calculer corrélation
                corr_value = abs(np.corrcoef(col_encoded, target_encoded)[0, 1])
                
                if corr_value > correlation_threshold:
                    important_categorical.append((col, corr_value))
            except:
                continue
        
        # Trier par corrélation décroissante
        important_categorical = sorted(important_categorical, key=lambda x: x[1], reverse=True)
        important_categorical = [col for col, _ in important_categorical]
    
    except Exception as e:
        print(f"Erreur lors du calcul des corrélations catégorielles: {e}")
        important_categorical = []
    
    return {
        'numeric_features': important_numeric,
        'categorical_features': important_categorical,
        'all_numeric_features': numeric_cols,
        'all_categorical_features': categorical_cols
    }

