import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 1. Chargement du dataset
df = pd.read_csv("fraude_bancaire_synthetique_final.csv")

# 2. Nettoyage des valeurs manquantes
df = df.dropna(subset=["fraude"])  # si la colonne 'fraude' contient des NaN
df["fraude"] = df["fraude"].astype(int)

# 3. Encodage des variables catégorielles
df = pd.get_dummies(df, drop_first=True)

# 4. Suppression des lignes restantes avec NaN (facultatif : imputation moyenne possible)
df = df.fillna(df.mean(numeric_only=True))

# 5. Séparation X / y
X = df.drop("fraude", axis=1)
y = df["fraude"]

# 6. Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. Rééquilibrage avec SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)

# 8. Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, stratify=y_res, random_state=42
)

# 9. Entraînement du modèle XGBoost
model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# 10. Prédiction
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# 11. Évaluation
print("\n📊 Rapport de Classification :\n")
print(classification_report(y_test, y_pred))
print(f"AUC Score : {roc_auc_score(y_test, y_proba):.4f}")

# 12. Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Matrice de Confusion")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.tight_layout()
plt.show()

# 13. Sauvegarde du modèle et du scaler
joblib.dump(model, "xgboost_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Sauvegarde des noms des colonnes utilisées pour l'entraînement
joblib.dump(X.columns.tolist(), "feature_columns.pkl")
