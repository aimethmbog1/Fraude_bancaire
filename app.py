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

# Navigation
st.sidebar.title("🧭 Navigation")
st.sidebar.markdown("**Sections disponibles :**")
st.sidebar.markdown("- 📝 Saisie des données")
st.sidebar.markdown("- 📅 Historique & Filtrage")
st.sidebar.markdown("- 📈 Tendance et Statistiques")

st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🔍 Détection de Fraude Bancaire</h1>", unsafe_allow_html=True)

# Historique
if "history" not in st.session_state:
    st.session_state["history"] = pd.DataFrame()

# Champs catégoriels prédéfinis
regions = ["Paris", "Lyon", "Marseille"]
genres = ["Homme", "Femme"]
cartes = ["Visa", "Mastercard", "Amex"]
canaux = ["Mobile", "Web", "Agence"]

# Données utilisateur
st.subheader("📝 Saisie des données transactionnelles")
input_dict = {}

# Ajout manuel des champs pour éviter erreurs
input_dict["montant"] = st.number_input("montant", min_value=0.0, format="%.2f")
input_dict["solde"] = st.number_input("solde", min_value=0.0, format="%.2f")
input_dict["age"] = st.number_input("age", min_value=18.0, format="%.0f")
input_dict["anciennete"] = st.number_input("anciennete", min_value=0.0, format="%.1f")
input_dict["region"] = st.selectbox("region", regions)
input_dict["genre"] = st.selectbox("genre", genres)
input_dict["type_carte"] = st.selectbox("type_carte", cartes)
input_dict["canal"] = st.selectbox("canal", canaux)

# Ajoute d’autres variables numériques selon ton modèle ici si nécessaire

if st.button("🚀 Prédire la fraude"):
    input_df = pd.DataFrame([input_dict])
    cat_vars = ["region", "genre", "type_carte", "canal"]

    # Encodage one-hot et alignement
    input_encoded = pd.get_dummies(input_df, columns=cat_vars)

    for col in feature_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0  # colonnes manquantes

    input_encoded = input_encoded[feature_columns]

    input_scaled = scaler.transform(input_encoded)

    # Prédiction
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]

    # Affichage résultat
    if prediction == 1:
        st.error(f"🚨 FRAUDE détectée ! (probabilité : {proba:.2%})")
    else:
        st.success(f"✅ Transaction normale (probabilité de fraude : {proba:.2%})")

    # Ajout à l'historique
    result_row = input_df.copy()
    result_row["Prédiction"] = "FRAUDE" if prediction == 1 else "NORMALE"
    result_row["Probabilité"] = f"{proba:.2%}"
    result_row["Horodatage"] = datetime.now().date()
    st.session_state["history"] = pd.concat([st.session_state["history"], result_row], ignore_index=True)

    with st.expander("📋 Détail de la transaction analysée"):
        st.dataframe(result_row)

# Affichage Historique
if not st.session_state["history"].empty:
    st.markdown("---")
    st.subheader("📅 Historique des prédictions")

    if st.button("🗑️ Réinitialiser l'historique"):
        st.session_state["history"] = pd.DataFrame()
        st.success("Historique vidé.")

    df_hist = st.session_state["history"].copy()
    df_hist["Horodatage"] = pd.to_datetime(df_hist["Horodatage"])

    min_date, max_date = df_hist["Horodatage"].min(), df_hist["Horodatage"].max()
    start_date, end_date = st.date_input("📆 Filtrer par date", [min_date, max_date])
    filtered = df_hist[(df_hist["Horodatage"] >= pd.to_datetime(start_date)) & (df_hist["Horodatage"] <= pd.to_datetime(end_date))]

    st.dataframe(filtered, use_container_width=True)

    # 📈 Tendance temporelle
    st.subheader("📅 Tendance temporelle des fraudes (7 derniers jours)")
    recent = filtered[(filtered["Prédiction"] == "FRAUDE") & 
                  (filtered["Horodatage"] >= pd.Timestamp.now() - pd.Timedelta(days=6))]
    recent["Horodatage"] = recent["Horodatage"].dt.date  # Convertir en date pour regrouper par jour
    recent = recent.groupby("Horodatage").size().reset_index(name="Count")
    recent["Horodatage"] = pd.to_datetime(recent["Horodatage"])
    recent = recent.sort_values("Horodatage")
    recent = recent[recent["Count"] > 0]  # Filtrer les jours avec au moins une fraude
    st.write(f"Nombre de fraudes détectées : {len(recent)}")
    st.write(f"Période : {start_date} à {end_date}")
    st.write("Graphique des fraudes détectées par jour :")
    st.write("Ce graphique montre le nombre de fraudes détectées par jour au cours des 7 derniers jours.")
    
    if not recent.empty:
        trend = recent.groupby("Horodatage").size().reset_index(name="Fraudes")
        fig = px.line(trend, x="Horodatage", y="Fraudes", markers=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Aucune fraude détectée ces 7 derniers jours.")

    # 📊 Répartition
    st.subheader("📊 Répartition des prédictions")
    fig = px.histogram(filtered, x="Prédiction", color="Prédiction", text_auto=True,
                       color_discrete_map={"FRAUDE": "red", "NORMALE": "green"})
    st.plotly_chart(fig, use_container_width=True)

    # 📥 Téléchargement
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Télécharger historique filtré", csv, "historique_fraude.csv", "text/csv")
