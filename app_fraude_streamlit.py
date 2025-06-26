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

# Configuration de l'app
st.set_page_config(page_title="🛡️ Détection de Fraude", layout="centered")

# Sidebar navigation
st.sidebar.title("🧭 Navigation")
st.sidebar.markdown("**Sections disponibles :**")
st.sidebar.markdown("- 📝 Saisie des données")
st.sidebar.markdown("- 📅 Historique & Filtrage")
st.sidebar.markdown("- 📈 Tendance et Statistiques")

st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🔍 Détection de Fraude Bancaire</h1>", unsafe_allow_html=True)

# Initialisation historique
if "history" not in st.session_state:
    st.session_state["history"] = pd.DataFrame()

# Champs dynamiques
regions = ["Paris", "Lyon", "Marseille"]
genres = ["Homme", "Femme"]
cartes = ["Visa", "Mastercard", "Amex"]
canaux = ["Mobile", "Web", "Agence"]

# Formulaire de saisie
st.subheader("📝 Saisie des données transactionnelles")
input_dict = {}
for col in feature_columns:
    if "montant" in col or "solde" in col or "age" in col or "anciennete" in col:
        input_dict[col] = st.number_input(col, min_value=0.0, format="%.2f")
    elif "region" in col:
        input_dict[col] = st.selectbox(col, regions)
    elif "genre" in col:
        input_dict[col] = st.selectbox(col, genres)
    elif "type_carte" in col:
        input_dict[col] = st.selectbox(col, cartes)
    elif "canal" in col:
        input_dict[col] = st.selectbox(col, canaux)
    else:
        input_dict[col] = st.number_input(col, format="%.2f")

if st.button("🚀 Prédire la fraude"):
    input_df = pd.DataFrame([input_dict])

    # Encodage one-hot pour les variables catégorielles
    cat_vars = ["region", "genre", "type_carte", "canal"]
    input_df_encoded = pd.get_dummies(input_df, columns=cat_vars)

    # Aligner avec les colonnes d'entraînement
    for col in feature_columns:
        if col not in input_df_encoded.columns:
            input_df_encoded[col] = 0
    input_df_encoded = input_df_encoded[feature_columns]

    # Mise à l’échelle
    input_scaled = scaler.transform(input_df_encoded)

    # Prédiction
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.markdown(f"<h3 style='color: red;'>🚨 FRAUDE détectée ! (probabilité : {proba:.2%})</h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='color: green;'>✅ Transaction normale (probabilité de fraude : {proba:.2%})</h3>", unsafe_allow_html=True)

    # Historique
    result_row = input_df_encoded.copy()
    result_row["Prédiction"] = "FRAUDE" if prediction == 1 else "NORMALE"
    result_row["Probabilité"] = f"{proba:.2%}"
    result_row["Horodatage"] = datetime.now().date()
    st.session_state["history"] = pd.concat([st.session_state["history"], result_row], ignore_index=True)

    with st.expander("📋 Détail de la transaction analysée"):
        st.dataframe(result_row)

# 📊 Historique
if not st.session_state["history"].empty:
    st.markdown("---")
    st.subheader("📅 Historique des prédictions")

    if st.button("🗑️ Réinitialiser l'historique"):
        st.session_state["history"] = pd.DataFrame()
        st.success("L’historique a été vidé.")

    df_hist = st.session_state["history"].copy()
    df_hist["Horodatage"] = pd.to_datetime(df_hist["Horodatage"])
    min_date, max_date = df_hist["Horodatage"].min(), df_hist["Horodatage"].max()
    start_date, end_date = st.date_input("📆 Filtrer par date", [min_date, max_date])
    filtered = df_hist[(df_hist["Horodatage"] >= pd.to_datetime(start_date)) & (df_hist["Horodatage"] <= pd.to_datetime(end_date))]

    st.dataframe(filtered.style.format("{:.2f}"), use_container_width=True)

    # 📈 Tendance temporelle
    st.subheader("📅 Tendance temporelle des fraudes (7 derniers jours)")
    recent_days = filtered[filtered["Prédiction"] == "FRAUDE"].copy()
    recent_days = recent_days[recent_days["Horodatage"] >= pd.to_datetime(datetime.now().date()) - pd.Timedelta(days=6)]
    if not recent_days.empty:
        daily_trend = recent_days.groupby("Horodatage").size().reset_index(name="Nombre de fraudes")
        daily_trend = daily_trend.sort_values("Horodatage")
        fig_trend = px.line(daily_trend, x="Horodatage", y="Nombre de fraudes", markers=True,
                            title="📈 Nombre de fraudes détectées - 7 derniers jours")
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("Aucune fraude détectée durant les 7 derniers jours.")

    # 📊 Répartition
    st.subheader("📊 Répartition des prédictions")
    fig = px.histogram(filtered, x="Prédiction", title="Répartition (après filtrage)",
                       color="Prédiction", color_discrete_map={"FRAUDE": "red", "NORMALE": "green"},
                       text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

    # 📥 Export CSV
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Télécharger l'historique filtré", csv, "historique_filtré.csv", "text/csv")
# 📈 Statistiques   
st.subheader("📊 Statistiques des transactions"
             )