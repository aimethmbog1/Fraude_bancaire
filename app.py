import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from datetime import datetime

# Chargement des modèles
model = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# Configuration
st.set_page_config(page_title="🛡️ Détection de Fraude", layout="centered")
st.title("🔍 Détection de Fraude Bancaire")

# Navigation
st.sidebar.title("🧭 Navigation")
st.sidebar.markdown("- 📝 Saisie des données")
st.sidebar.markdown("- 📅 Historique & Filtrage")
st.sidebar.markdown("- 📈 Tendance et Statistiques")

# Initialisation historique
if "history" not in st.session_state:
    st.session_state["history"] = pd.DataFrame()

# Valeurs possibles
regions = ["Paris", "Lyon", "Marseille"]
genres = ["Homme", "Femme"]
cartes = ["Visa", "Mastercard", "Amex"]
canaux = ["Mobile", "Web", "Agence"]

# 📝 Formulaire utilisateur
st.subheader("📝 Saisie des données transactionnelles")
input_dict = {
    "montant": st.number_input("Montant", min_value=0.0, format="%.2f"),
    "solde": st.number_input("Solde", min_value=0.0, format="%.2f"),
    "age": st.number_input("Âge", min_value=18.0, format="%.0f"),
    "anciennete": st.number_input("Ancienneté", min_value=0.0, format="%.1f"),
    "region": st.selectbox("Région", regions),
    "genre": st.selectbox("Genre", genres),
    "type_carte": st.selectbox("Type de carte", cartes),
    "canal": st.selectbox("Canal", canaux)
}

if st.button("🚀 Prédire la fraude"):
    input_df = pd.DataFrame([input_dict])

    # Colonnes catégorielles à encoder
    cat_vars = [col for col in ["region", "genre", "type_carte", "canal"] if col in input_df.columns]

    # Encodage
    input_encoded = pd.get_dummies(input_df, columns=cat_vars)

    # Alignement avec les colonnes du modèle
    for col in feature_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[feature_columns]

    # Mise à l'échelle
    input_scaled = scaler.transform(input_encoded)

    # Prédiction
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]

    # Résultat
    if prediction == 1:
        st.error(f"🚨 FRAUDE détectée ! (probabilité : {proba:.2%})")
    else:
        st.success(f"✅ Transaction normale (probabilité : {proba:.2%})")

    # Historique
    log = input_df.copy()
    log["Prédiction"] = "FRAUDE" if prediction == 1 else "NORMALE"
    log["Probabilité"] = f"{proba:.2%}"
    log["Horodatage"] = pd.Timestamp.now()
    st.session_state["history"] = pd.concat([st.session_state["history"], log], ignore_index=True)

    with st.expander("📋 Détail de la transaction analysée"):
        st.dataframe(log)

# 📅 Historique
if not st.session_state["history"].empty:
    st.markdown("---")
    st.subheader("📅 Historique des prédictions")

    df_hist = st.session_state["history"].copy()
    df_hist["Horodatage"] = pd.to_datetime(df_hist["Horodatage"])

    # Filtrage par date
    min_date, max_date = df_hist["Horodatage"].min().date(), df_hist["Horodatage"].max().date()
    start_date, end_date = st.date_input("📆 Filtrer par date", [min_date, max_date])
    mask = (df_hist["Horodatage"].dt.date >= start_date) & (df_hist["Horodatage"].dt.date <= end_date)
    filtered = df_hist.loc[mask]

    st.dataframe(filtered, use_container_width=True)

    # 📈 Tendance temporelle des fraudes
    st.subheader("📈 Tendance des fraudes (7 derniers jours)")
    recent = filtered[(filtered["Prédiction"] == "FRAUDE") &
                      (filtered["Horodatage"] >= pd.Timestamp.now() - pd.Timedelta(days=6))].copy()
    recent["Date"] = recent["Horodatage"].dt.date
    daily = recent.groupby("Date").size().reset_index(name="Nombre de fraudes")

    if not daily.empty:
        fig_trend = px.line(daily, x="Date", y="Nombre de fraudes", markers=True,
                            title="📊 Nombre de fraudes détectées")
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("Aucune fraude détectée durant les 7 derniers jours.")

    # 📊 Répartition
    st.subheader("📊 Répartition des prédictions")
    fig_hist = px.histogram(filtered, x="Prédiction", color="Prédiction",
                            text_auto=True,
                            color_discrete_map={"FRAUDE": "red", "NORMALE": "green"})
    st.plotly_chart(fig_hist, use_container_width=True)

    # Export CSV
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Télécharger l'historique filtré", csv, "historique_fraude.csv", "text/csv")
