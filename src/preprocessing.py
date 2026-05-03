import numpy as np
import re
import pandas as pd
from sklearn.impute import KNNImputer
import ipaddress
from datetime import datetime


class Parseur:
    def __init__(
        self,
        registration_col="RegistrationDate",
        ip_col="LastLoginIP",
        drop_original_date=False,
        drop_original_ip=False
    ):
        self.registration_col = registration_col
        self.ip_col = ip_col
        self.drop_original_date = drop_original_date
        self.drop_original_ip = drop_original_ip
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
                parts.append(f"Colonnes manquantes par rapport au train : {missing_cols}")
            if extra_cols:
                parts.append(f"Colonnes supplémentaires non vues au train : {extra_cols}")

            raise ValueError(
                "Les colonnes de X ne correspondent pas à celles vues au fit(). "
                + " | ".join(parts)
            )

    def _is_missing_text(self, x):
        if pd.isna(x):
            return True
        return str(x).strip().lower() in {
            "", "unknown", "inconnu", "na", "n/a", "none", "null", "missing", "?"
        }

    def _clean_text(self, x):
        if self._is_missing_text(x):
            return np.nan
        return str(x).strip()

    # ----------------------------------------------------------
    # RegistrationDate : uniformiser puis convertir
    # Formats gérés :
    # - dd/mm/yy
    # - dd/mm/yyyy
    # - yyyy-mm-dd
    # ----------------------------------------------------------
    def _standardize_registration_date_value(self, x):
        if self._is_missing_text(x):
            return np.nan

        s = str(x).strip()

        try:
            # yyyy-mm-dd
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
                dt = datetime.strptime(s, "%Y-%m-%d")
                return dt.strftime("%Y-%m-%d")

            # dd/mm/yyyy
            if re.fullmatch(r"\d{2}/\d{2}/\d{4}", s):
                dt = datetime.strptime(s, "%d/%m/%Y")
                return dt.strftime("%Y-%m-%d")

            # dd/mm/yy
            if re.fullmatch(r"\d{2}/\d{2}/\d{2}", s):
                dt = datetime.strptime(s, "%d/%m/%y")
                return dt.strftime("%Y-%m-%d")

            return np.nan

        except ValueError:
            return np.nan

    def _standardize_registration_date_series(self, s):
        return s.apply(self._standardize_registration_date_value)

    # ----------------------------------------------------------
    # Parsing LastLoginIP
    # ----------------------------------------------------------
    def _extract_ip_features(self, x):
        if self._is_missing_text(x):
            return pd.Series(
                {
                    "IsValidIP": np.nan,
                    "IsPrivateIP": np.nan,
                    "IPVersion": np.nan,
                }
            )

        s = str(x).strip()

        try:
            ip_obj = ipaddress.ip_address(s)
            return pd.Series(
                {
                    "IsValidIP": 1,
                    "IsPrivateIP": int(ip_obj.is_private),
                    "IPVersion": ip_obj.version,
                }
            )
        except ValueError:
            return pd.Series(
                {
                    "IsValidIP": 0,
                    "IsPrivateIP": np.nan,
                    "IPVersion": np.nan,
                }
            )

    # ----------------------------------------------------------
    # Fit
    # ----------------------------------------------------------
    def fit(self, X, y=None):
        X = self._to_dataframe(X)
        self.input_columns_ = X.columns.tolist()

        output_columns = self.input_columns_.copy()

        if self.registration_col in self.input_columns_:
            for col in ["RegYear", "RegMonth", "RegDay", "RegWeekday"]:
                if col not in output_columns:
                    output_columns.append(col)

            if self.drop_original_date and self.registration_col in output_columns:
                output_columns.remove(self.registration_col)

        if self.ip_col in self.input_columns_:
            for col in ["IsValidIP", "IsPrivateIP", "IPVersion"]:
                if col not in output_columns:
                    output_columns.append(col)

            if self.drop_original_ip and self.ip_col in output_columns:
                output_columns.remove(self.ip_col)

        self.output_columns_ = output_columns
        self.fitted_ = True
        return self

    # ----------------------------------------------------------
    # Transform
    # ----------------------------------------------------------
    def transform(self, X):
        if not self.fitted_:
            raise RuntimeError("Le parseur doit être fit avant transform.")

        X = self._to_dataframe(X)
        self._validate_input_columns(X)

        X = X[self.input_columns_].copy()

        # -------------------------
        # RegistrationDate
        # -------------------------
        if self.registration_col in X.columns:
            X[self.registration_col] = X[self.registration_col].apply(self._clean_text)

            # 1) Uniformiser en texte YYYY-MM-DD
            standardized_dates = self._standardize_registration_date_series(X[self.registration_col])

            # 2) Convertir en vrai datetime pandas
            X[self.registration_col] = pd.to_datetime(
                standardized_dates,
                format="%Y-%m-%d",
                errors="coerce"
            )

            # 3) Extraire les features
            X["RegYear"] = X[self.registration_col].dt.year
            X["RegMonth"] = X[self.registration_col].dt.month
            X["RegDay"] = X[self.registration_col].dt.day
            X["RegWeekday"] = X[self.registration_col].dt.weekday

        # -------------------------
        # LastLoginIP
        # -------------------------
        if self.ip_col in X.columns:
            X[self.ip_col] = X[self.ip_col].apply(self._clean_text)
            ip_features = X[self.ip_col].apply(self._extract_ip_features)
            X = pd.concat([X, ip_features], axis=1)

        # -------------------------
        # Drop éventuel des colonnes brutes
        # -------------------------
        if self.drop_original_date and self.registration_col in X.columns:
            X = X.drop(columns=[self.registration_col])

        if self.drop_original_ip and self.ip_col in X.columns:
            X = X.drop(columns=[self.ip_col])

        X = X[self.output_columns_].copy()
        return X

    # ----------------------------------------------------------
    # Fit transform
    # ----------------------------------------------------------
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
    



class RetailPreprocessor:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

        # Colonnes à supprimer
        self.columns_to_drop = [
            "CustomerID",
            "WeekendPreference",
            "NewsletterSubscribed"
        ]

        # Colonnes métier
        self.age_col = "Age"
        self.agecat_col = "AgeCategory"
        self.gender_col = "Gender"
        self.region_col = "Region"
        self.country_col = "Country"
        self.support_col = "SupportTicketsCount"
        self.satisfaction_col = "SatisfactionScore"
        self.avgdays_col = "AvgDaysBetweenPurchases"

        # Colonnes créées par Parseur : date
        self.date_feature_cols = [
            "RegYear",
            "RegMonth",
            "RegDay",
            "RegWeekday"
        ]

        # Colonnes créées par Parseur : IP
        self.ip_valid_col = "IsValidIP"
        self.ip_private_col = "IsPrivateIP"
        self.ip_version_col = "IPVersion"
        self.ip_feature_cols = [
            self.ip_valid_col,
            self.ip_private_col,
            self.ip_version_col
        ]

        # Variables les plus dépendantes pour KNN sur AvgDaysBetweenPurchases
        self.knn_num_features = [
            "Frequency",
            "CustomerTenureDays",
            "UniqueInvoices"
        ]

        self.text_missing_values = {
            "", "unknown", "inconnu", "na", "n/a", "none", "null",
            "missing", "?", "unspecified"
        }

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
                parts.append(f"Colonnes manquantes par rapport au train : {missing_cols}")
            if extra_cols:
                parts.append(f"Colonnes supplémentaires non vues au train : {extra_cols}")

            raise ValueError(
                "Les colonnes de X ne correspondent pas à celles vues au fit(). "
                + " | ".join(parts)
            )

    def _is_text_missing(self, x):
        if pd.isna(x):
            return True
        return str(x).strip().lower() in self.text_missing_values

    def _clean_text(self, x):
        if self._is_text_missing(x):
            return np.nan
        return str(x).strip()

    def _mode_or_default(self, series, default=np.nan):
        s = series.dropna()
        if s.empty:
            return default
        return s.mode().iloc[0]

    def _safe_median(self, series, default=np.nan):
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return default
        return s.median()

    def _safe_discrete_median(self, series, default=np.nan):
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return default
        return int(round(s.median()))

    def _age_to_category(self, age):
        if pd.isna(age):
            return "Inconnu"

        age = float(age)

        if 18 <= age <= 24:
            return "18-24"
        elif 25 <= age <= 34:
            return "25-34"
        elif 35 <= age <= 44:
            return "35-44"
        elif 45 <= age <= 54:
            return "45-54"
        elif 55 <= age <= 64:
            return "55-64"
        elif age >= 65:
            return "65+"
        else:
            return "Inconnu"

    # ----------------------------------------------------------
    # Outliers : bornes IQR apprises sur train
    # Exclure les colonnes IP dérivées
    # ----------------------------------------------------------
    def _learn_outlier_bounds(self, X):
        self.outlier_bounds_ = {}

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [
            c for c in numeric_cols
            if c not in self.columns_to_drop and c not in self.ip_feature_cols
        ]

        for col in numeric_cols:
            s = pd.to_numeric(X[col], errors="coerce").dropna()

            if s.empty or s.nunique() <= 1:
                continue

            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1

            if iqr == 0:
                continue

            lower_limit = q1 - 1.5 * iqr
            upper_limit = q3 + 1.5 * iqr

            self.outlier_bounds_[col] = (lower_limit, upper_limit)

    def _cap_outliers(self, X):
        X = X.copy()

        for col, (lower_limit, upper_limit) in self.outlier_bounds_.items():
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")
                X[col] = np.where(
                    X[col] >= upper_limit,
                    upper_limit,
                    np.where(
                        X[col] <= lower_limit,
                        lower_limit,
                        X[col]
                    )
                )
        return X

    # ----------------------------------------------------------
    # Matrice KNN pour AvgDaysBetweenPurchases
    # ----------------------------------------------------------
    def _build_knn_matrix(self, X, fit_mode=False):
        cols = [self.avgdays_col] + [c for c in self.knn_num_features if c in X.columns]

        knn_df = pd.DataFrame(index=X.index)

        for col in cols:
            if col in X.columns:
                knn_df[col] = pd.to_numeric(X[col], errors="coerce")
            else:
                knn_df[col] = np.nan

        if fit_mode:
            self.knn_matrix_columns_ = knn_df.columns.tolist()
        else:
            for col in self.knn_matrix_columns_:
                if col not in knn_df.columns:
                    knn_df[col] = np.nan
            knn_df = knn_df[self.knn_matrix_columns_]

        return knn_df

    # ----------------------------------------------------------
    # FIT
    # ----------------------------------------------------------
    def fit(self, X, y=None):
        X = self._to_dataframe(X)
        self.input_columns_ = X.columns.tolist()

        self.output_columns_ = [c for c in self.input_columns_ if c not in self.columns_to_drop]

        self.numeric_columns_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.numeric_columns_ = [c for c in self.numeric_columns_ if c not in self.columns_to_drop]

        for col in self.date_feature_cols + self.ip_feature_cols:
            if col in X.columns and col not in self.numeric_columns_ and col not in self.columns_to_drop:
                self.numeric_columns_.append(col)

        self.categorical_columns_ = [c for c in self.input_columns_ if c not in self.numeric_columns_]
        self.categorical_columns_ = [c for c in self.categorical_columns_ if c not in self.columns_to_drop]

        # Age
        if self.age_col in X.columns:
            age_series = pd.to_numeric(X[self.age_col], errors="coerce")
            self.age_median_ = age_series.median()
        else:
            self.age_median_ = np.nan

        self.age_median_category_ = self._age_to_category(self.age_median_)

        # Gender
        if self.gender_col in X.columns:
            gender_series = X[self.gender_col].apply(self._clean_text)
            self.gender_mode_ = self._mode_or_default(gender_series, default="Unknown")
        else:
            self.gender_mode_ = "Unknown"

        # Region
        if self.region_col in X.columns:
            region_series = X[self.region_col].apply(self._clean_text)
            region_series = region_series[region_series != "Autre"]
            self.region_mode_ = self._mode_or_default(region_series, default=np.nan)
        else:
            self.region_mode_ = np.nan

        # Country
        if self.country_col in X.columns:
            country_series = X[self.country_col].apply(self._clean_text)
            self.country_mode_ = self._mode_or_default(country_series, default=np.nan)
        else:
            self.country_mode_ = np.nan

        # SupportTicketsCount
        if self.support_col in X.columns:
            support = pd.to_numeric(X[self.support_col], errors="coerce")
            support_valid = support[support.between(0, 15, inclusive="both")]
            self.support_median_ = support_valid.median() if not support_valid.empty else 0
        else:
            self.support_median_ = 0

        # SatisfactionScore
        if self.satisfaction_col in X.columns:
            satisfaction = pd.to_numeric(X[self.satisfaction_col], errors="coerce")
            satisfaction_valid = satisfaction[satisfaction.between(1, 5, inclusive="both")]
            self.satisfaction_median_ = satisfaction_valid.median() if not satisfaction_valid.empty else 3
        else:
            self.satisfaction_median_ = 3

        # Colonnes date dérivées
        self.date_feature_medians_ = {}
        for col in self.date_feature_cols:
            if col in X.columns:
                self.date_feature_medians_[col] = self._safe_discrete_median(X[col], default=np.nan)

        # Colonnes IP dérivées
        self.isvalidip_fill_ = 0

        if self.ip_private_col in X.columns:
            s = pd.to_numeric(X[self.ip_private_col], errors="coerce")
            s = s[s.isin([0, 1])]
            self.isprivateip_mode_ = self._mode_or_default(s, default=0)
        else:
            self.isprivateip_mode_ = 0

        if self.ip_version_col in X.columns:
            s = pd.to_numeric(X[self.ip_version_col], errors="coerce")
            s = s[s.isin([4, 6])]
            self.ipversion_mode_ = self._mode_or_default(s, default=4)
        else:
            self.ipversion_mode_ = 4

        # Médianes génériques des autres numériques
        self.numeric_medians_ = {}

        for col in self.numeric_columns_:
            if col in self.ip_feature_cols:
                continue
            if col in self.date_feature_cols:
                self.numeric_medians_[col] = self.date_feature_medians_.get(col, np.nan)
            elif col == self.support_col:
                self.numeric_medians_[col] = self.support_median_
            elif col == self.satisfaction_col:
                self.numeric_medians_[col] = self.satisfaction_median_
            elif col == self.age_col:
                self.numeric_medians_[col] = self.age_median_
            else:
                self.numeric_medians_[col] = self._safe_median(X[col], default=np.nan)

        if self.avgdays_col in X.columns:
            self.avgdays_median_ = self._safe_median(X[self.avgdays_col], default=np.nan)
        else:
            self.avgdays_median_ = np.nan

        # Modes génériques des autres catégorielles
        self.categorical_modes_ = {}

        for col in self.categorical_columns_:
            series = X[col].apply(self._clean_text)

            if col == self.region_col:
                series = series[series != "Autre"]

            self.categorical_modes_[col] = self._mode_or_default(series, default=np.nan)

        # Bornes IQR apprises sur train
        X_bounds = X.copy()

        if self.support_col in X_bounds.columns:
            X_bounds[self.support_col] = pd.to_numeric(X_bounds[self.support_col], errors="coerce")
            X_bounds.loc[
                ~X_bounds[self.support_col].between(0, 15, inclusive="both"),
                self.support_col
            ] = np.nan

        if self.satisfaction_col in X_bounds.columns:
            X_bounds[self.satisfaction_col] = pd.to_numeric(X_bounds[self.satisfaction_col], errors="coerce")
            X_bounds.loc[
                ~X_bounds[self.satisfaction_col].between(1, 5, inclusive="both"),
                self.satisfaction_col
            ] = np.nan

        for col in self.numeric_columns_:
            if col in X_bounds.columns and col not in self.ip_feature_cols:
                X_bounds[col] = pd.to_numeric(X_bounds[col], errors="coerce")

        self._learn_outlier_bounds(X_bounds)

        # KNN pour AvgDaysBetweenPurchases
        if self.avgdays_col in X.columns:
            X_knn = X.copy()

            for col in [self.avgdays_col] + self.knn_num_features:
                if col in X_knn.columns:
                    X_knn[col] = pd.to_numeric(X_knn[col], errors="coerce")

            for col in self.knn_num_features:
                if col in X_knn.columns:
                    median_col = self.numeric_medians_.get(col, np.nan)
                    X_knn[col] = X_knn[col].fillna(median_col)

            X_knn = self._cap_outliers(X_knn)

            knn_matrix = self._build_knn_matrix(X_knn, fit_mode=True)
            self.knn_imputer_ = KNNImputer(n_neighbors=self.n_neighbors)
            self.knn_imputer_.fit(knn_matrix)
        else:
            self.knn_imputer_ = None
            self.knn_matrix_columns_ = []

        self.fitted_ = True
        return self

    # ----------------------------------------------------------
    # TRANSFORM
    # ----------------------------------------------------------
    def transform(self, X):
        if not self.fitted_:
            raise RuntimeError("Le préprocesseur doit être fit avant transform.")

        X = self._to_dataframe(X)
        self._validate_input_columns(X)

        X = X[self.input_columns_].copy()

        # Nettoyage texte catégorielles
        for col in self.categorical_columns_:
            if col in X.columns:
                X[col] = X[col].apply(self._clean_text)

        # Age
        if self.age_col in X.columns:
            X[self.age_col] = pd.to_numeric(X[self.age_col], errors="coerce")
            X[self.age_col] = X[self.age_col].fillna(self.age_median_)

        # AgeCategory
        if self.agecat_col in X.columns:
            X[self.agecat_col] = X[self.agecat_col].fillna(self.age_median_category_)

        # Gender
        if self.gender_col in X.columns:
            X[self.gender_col] = X[self.gender_col].fillna(self.gender_mode_)

        # Region
        if self.region_col in X.columns:
            X.loc[X[self.region_col] == "Autre", self.region_col] = np.nan
            X[self.region_col] = X[self.region_col].fillna(self.region_mode_)

        # Country
        if self.country_col in X.columns:
            X[self.country_col] = X[self.country_col].fillna(self.country_mode_)

        # Autres catégorielles
        for col in self.categorical_columns_:
            if col in X.columns and col not in [self.agecat_col, self.gender_col, self.region_col, self.country_col]:
                X[col] = X[col].fillna(self.categorical_modes_.get(col, np.nan))

        # SupportTicketsCount
        if self.support_col in X.columns:
            X[self.support_col] = pd.to_numeric(X[self.support_col], errors="coerce")
            X.loc[
                ~X[self.support_col].between(0, 15, inclusive="both"),
                self.support_col
            ] = np.nan
            X[self.support_col] = X[self.support_col].fillna(self.support_median_)

        # SatisfactionScore
        if self.satisfaction_col in X.columns:
            X[self.satisfaction_col] = pd.to_numeric(X[self.satisfaction_col], errors="coerce")
            X.loc[
                ~X[self.satisfaction_col].between(1, 5, inclusive="both"),
                self.satisfaction_col
            ] = np.nan
            X[self.satisfaction_col] = X[self.satisfaction_col].fillna(self.satisfaction_median_)

        # Colonnes date dérivées
        for col in self.date_feature_cols:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")
                X[col] = X[col].fillna(self.date_feature_medians_.get(col, np.nan))

        # Colonnes IP dérivées
        if self.ip_valid_col in X.columns:
            X[self.ip_valid_col] = pd.to_numeric(X[self.ip_valid_col], errors="coerce")
            X.loc[~X[self.ip_valid_col].isin([0, 1]), self.ip_valid_col] = np.nan
            X[self.ip_valid_col] = X[self.ip_valid_col].fillna(self.isvalidip_fill_)

        if self.ip_private_col in X.columns:
            X[self.ip_private_col] = pd.to_numeric(X[self.ip_private_col], errors="coerce")
            X.loc[~X[self.ip_private_col].isin([0, 1]), self.ip_private_col] = np.nan
            X[self.ip_private_col] = X[self.ip_private_col].fillna(self.isprivateip_mode_)

        if self.ip_version_col in X.columns:
            X[self.ip_version_col] = pd.to_numeric(X[self.ip_version_col], errors="coerce")
            X.loc[~X[self.ip_version_col].isin([4, 6]), self.ip_version_col] = np.nan
            X[self.ip_version_col] = X[self.ip_version_col].fillna(self.ipversion_mode_)

        # Autres numériques
        excluded_cols = {
            self.age_col,
            self.support_col,
            self.satisfaction_col,
            self.avgdays_col,
            *self.date_feature_cols,
            *self.ip_feature_cols
        }

        for col in self.numeric_columns_:
            if col in X.columns and col not in excluded_cols:
                X[col] = pd.to_numeric(X[col], errors="coerce")
                X[col] = X[col].fillna(self.numeric_medians_.get(col, np.nan))

        if self.avgdays_col in X.columns:
            X[self.avgdays_col] = pd.to_numeric(X[self.avgdays_col], errors="coerce")

        # Capping IQR
        X = self._cap_outliers(X)

        # KNN pour AvgDaysBetweenPurchases
        if self.knn_imputer_ is not None and self.avgdays_col in X.columns:
            knn_matrix = self._build_knn_matrix(X, fit_mode=False)
            knn_imputed = self.knn_imputer_.transform(knn_matrix)
            knn_imputed_df = pd.DataFrame(
                knn_imputed,
                columns=self.knn_matrix_columns_,
                index=X.index
            )
            X[self.avgdays_col] = knn_imputed_df[self.avgdays_col]
            X[self.avgdays_col] = X[self.avgdays_col].fillna(self.avgdays_median_)

        # Drop colonnes inutiles
        cols_to_drop_now = [c for c in self.columns_to_drop if c in X.columns]
        X = X.drop(columns=cols_to_drop_now)

        X = X[self.output_columns_].copy()
        return X

    # ----------------------------------------------------------
    # FIT_TRANSFORM
    # ----------------------------------------------------------
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)




class FeatureEngineer:
    def __init__(self):
        # Colonnes sources
        self.monetary_col = "MonetaryTotal"
        self.recency_col = "Recency"
        self.frequency_col = "Frequency"
        self.tenure_col = "CustomerTenureDays"

        # Nouvelles features créées
        self.created_feature_cols = [
            "MonetaryPerDay",
            "AvgBasketValue",
            "TenureRatio"
        ]

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
                parts.append(f"Colonnes manquantes par rapport au train : {missing_cols}")
            if extra_cols:
                parts.append(f"Colonnes supplémentaires non vues au train : {extra_cols}")

            raise ValueError(
                "Les colonnes de X ne correspondent pas à celles vues au fit(). "
                + " | ".join(parts)
            )

    def _safe_divide(self, numerator, denominator):
        numerator = pd.to_numeric(numerator, errors="coerce")
        denominator = pd.to_numeric(denominator, errors="coerce")

        result = np.where(
            denominator.isna() | (denominator == 0),
            np.nan,
            numerator / denominator
        )
        return pd.Series(result, index=numerator.index)

    def _build_features(self, X):
        X = X.copy()

        # 1) Ratio dépenses / récence
        if self.monetary_col in X.columns and self.recency_col in X.columns:
            monetary = pd.to_numeric(X[self.monetary_col], errors="coerce")
            recency = pd.to_numeric(X[self.recency_col], errors="coerce")
            X["MonetaryPerDay"] = self._safe_divide(monetary, recency + 1)

        # 2) Panier moyen
        if self.monetary_col in X.columns and self.frequency_col in X.columns:
            monetary = pd.to_numeric(X[self.monetary_col], errors="coerce")
            frequency = pd.to_numeric(X[self.frequency_col], errors="coerce")
            X["AvgBasketValue"] = self._safe_divide(monetary, frequency)

        # 3) Ancienneté vs activité récente
        if self.recency_col in X.columns and self.tenure_col in X.columns:
            recency = pd.to_numeric(X[self.recency_col], errors="coerce")
            tenure = pd.to_numeric(X[self.tenure_col], errors="coerce")
            X["TenureRatio"] = self._safe_divide(recency, tenure)

        return X

    # ----------------------------------------------------------
    # FIT
    # ----------------------------------------------------------
    def fit(self, X, y=None):
        X = self._to_dataframe(X)
        self.input_columns_ = X.columns.tolist()

        X_feat = self._build_features(X)

        # Médianes des nouvelles features pour sécuriser le transform
        self.created_feature_medians_ = {}
        for col in self.created_feature_cols:
            if col in X_feat.columns:
                s = pd.to_numeric(X_feat[col], errors="coerce").dropna()
                self.created_feature_medians_[col] = s.median() if not s.empty else np.nan
            else:
                self.created_feature_medians_[col] = np.nan

        self.output_columns_ = X_feat.columns.tolist()
        self.fitted_ = True
        return self

    # ----------------------------------------------------------
    # TRANSFORM
    # ----------------------------------------------------------
    def transform(self, X):
        if not self.fitted_:
            raise RuntimeError("Le FeatureEngineer doit être fit avant transform.")

        X = self._to_dataframe(X)
        self._validate_input_columns(X)

        X = X[self.input_columns_].copy()

        # Création des nouvelles features
        X = self._build_features(X)

        # Sécurisation des NaN éventuels dus aux divisions
        for col in self.created_feature_cols:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")
                X[col] = X[col].fillna(self.created_feature_medians_.get(col, np.nan))

        X = X[self.output_columns_].copy()
        return X

    # ----------------------------------------------------------
    # FIT_TRANSFORM
    # ----------------------------------------------------------
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)