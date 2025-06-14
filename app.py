# app_final.py
import streamlit as st 
import joblib
import pandas as pd
import numpy as np
import plotly.express as px 
import shap 
import matplotlib.pyplot as plt 
import streamlit.components.v1 as components 
import os
import tempfile
import uuid 




def st_shap(plot, height=None):
    """Fonction wrapper améliorée pour afficher les plots SHAP dans Streamlit,
       en utilisant une sauvegarde HTML temporaire pour les plots interactifs."""
    
    # Vérifier si c'est l'objet spécifique du force_plot HTML/JS
    if isinstance(plot, shap.plots._force.AdditiveForceVisualizer):
        try:
            # Créer un nom de fichier temporaire unique
            # Utiliser tempfile pour gérer le fichier temporaire
            with tempfile.NamedTemporaryFile(mode='w', suffix=".html", delete=False) as tmp_file:
                # Sauvegarder le graphique SHAP dans le fichier HTML temporaire
                shap.save_html(tmp_file.name, plot)
                # Lire le contenu du fichier HTML
                with open(tmp_file.name, 'r') as f:
                    html_content = f.read()
            
            # Afficher le contenu HTML lu dans Streamlit
            components.html(html_content, height=height, scrolling=True)
            
            # Supprimer le fichier temporaire après lecture (optionnel si delete=True fonctionne bien)
            if os.path.exists(tmp_file.name):
                 os.remove(tmp_file.name)

        except Exception as e:
            st.error(f"Erreur lors de la sauvegarde/lecture/affichage du composant HTML SHAP : {e}")
            st.write("Affichage de l'objet brut:")
            st.write(plot)
            
    # Vérifier si c'est un graphique Matplotlib
    elif isinstance(plot, plt.Figure):
        try:
            st.pyplot(plot, bbox_inches="tight", clear_figure=True)
            plt.close(plot)
        except Exception as e:
            st.error(f"Erreur lors de l'affichage du graphique Matplotlib SHAP : {e}")
            
    # Fallback si le type n'est pas reconnu
    else:
        st.warning(f"Type de graphique SHAP non géré ({type(plot)}). Tentative d'affichage direct.")
        st.write(plot)



# --- Configuration ---
PIPELINE_PATH = "best_pipeline_full.pkl"
PREPROCESSOR_PATH = "preprocessor.pkl" # Path to the saved preprocessor

# ---Chargement du Pipeline Complet et du Préprocesseur ---
@st.cache_resource # Cache les ressources chargées
def load_pipeline_and_preprocessor(pipeline_path, preprocessor_path):
    pipeline = None
    preprocessor = None
    try:
        pipeline = joblib.load(pipeline_path)
        st.success(f"Pipeline chargé depuis {pipeline_path}")
    except FileNotFoundError:
        st.error(f"Erreur : Le fichier pipeline {pipeline_path} est introuvable.")
        st.stop()
    except Exception as e:
        st.error(f"Erreur lors du chargement du pipeline : {e}")
        st.stop()

    try:
        preprocessor = joblib.load(preprocessor_path)
        st.success(f"Préprocesseur chargé depuis {preprocessor_path}")
    except FileNotFoundError:
        st.warning(f"Fichier préprocesseur {preprocessor_path} non trouvé. L\"analyse SHAP sera limitée.")
    except Exception as e:
        st.error(f"Erreur lors du chargement du préprocesseur : {e}")
        # Ne pas arrêter l\"app, mais SHAP pourrait ne pas fonctionner

    return pipeline, preprocessor

pipeline, preprocessor = load_pipeline_and_preprocessor(PIPELINE_PATH, PREPROCESSOR_PATH)

# Extraire le modèle du pipeline (pour SHAP)
model = None
if pipeline:
    try:
        # Tenter d\"extraire l\"étape nommée \"classifier\"
        model = pipeline.named_steps["classifier"]
    except KeyError:
        # Sinon, supposer que c\"est la dernière étape
        model = pipeline.steps[-1][1]
    except Exception as e:
        st.warning(f"Impossible d\"extraire le modèle du pipeline pour SHAP : {e}")

# --- Fonction pour faire des prédictions --- 
def make_prediction(features_df):
    try:
        prediction = pipeline.predict(features_df)
        probability = pipeline.predict_proba(features_df)
        probability_percent = np.round(probability * 100, 2)
        return prediction, probability, probability_percent
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
        return None, None, None

# --- Fonction SHAP améliorée ---
@st.cache_data
def explain_prediction(input_df):
    if preprocessor is None or model is None:
        st.warning("Préprocesseur ou modèle non disponible pour l'analyse SHAP.")
        return None, None, None
    
    try:
        # Prétraitement
        input_processed = preprocessor.transform(input_df)
        
        # Noms des features
        try:
            feature_names = preprocessor.get_feature_names_out()
        except AttributeError:
            feature_names = [f"feature_{i}" for i in range(input_processed.shape[1])]
        
        input_processed_df = pd.DataFrame(input_processed, columns=feature_names)

        # Sélection de l'explainer
        model_type = str(type(model)).lower()
        
        if 'randomforest' in model_type or 'gradientboosting' in model_type:
            explainer = shap.TreeExplainer(model)
        elif 'logisticregression' in model_type:
            explainer = shap.LinearExplainer(model, input_processed_df)
        else:
            predict_func = lambda x: pipeline.predict_proba(x)[:, 1]
            explainer = shap.KernelExplainer(predict_func, input_processed_df)

        # Calcul SHAP
        shap_values = explainer.shap_values(input_processed_df) 
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Pour classification binaire
            expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        else:
            expected_value = explainer.expected_value

        return shap_values, expected_value, input_processed_df

    except Exception as e:
        st.error(f"Erreur SHAP : {str(e)}")
        return None, None, None

# --- Interface Streamlit --- 

st.title("Application de Prédiction et d\'Explication de Défaut de Paiement")
st.write("Utilisez les options dans la barre latérale pour entrer les informations d\"un client et obtenir une prédiction de défaut de paiement, ainsi qu\"une explication de cette prédiction.")

st.sidebar.header("Informations sur le client")

# --- Saisie des caractéristiques du client --- 
# (Code de saisie identique à app_modified_step3.py)
col1, col2 = st.sidebar.columns(2)

with col1:
    limit_bal = st.number_input("Montant du crédit (LIMIT_BAL)", min_value=0, value=50000, step=1000)
    sex = st.selectbox("Sexe (SEX)", ["Female", "Male"]) # Correction : "Female"
    # Utiliser les labels pour l\"UI, mais les noms de colonnes bruts pour le DataFrame
    education_options = {"Graduate school": "Graduate school", "University": "University", "High school": "High school", "Others": "Others"}
    education_label = st.selectbox("Niveau d\"éducation (EDUCATION)", list(education_options.keys()))
    
    marriage_options = {"Married": "Married", "Single": "Single", "Others": "Others"}
    marriage_label = st.selectbox("Statut matrimonial (MARRIAGE)", list(marriage_options.keys()))
    
    age = st.number_input("Age (AGE)", min_value=18, max_value=100, value=30)

with col2:
    st.markdown("**Statuts de Paiement (PAY_X)**")
    payment_options = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    default_pay_index = payment_options.index(0)
    payment_status_sep = st.selectbox("Septembre (PAY_0)", options=payment_options, index=default_pay_index)
    payment_status_aug = st.selectbox("Août (PAY_2)", options=payment_options, index=default_pay_index)
    payment_status_jul = st.selectbox("Juillet (PAY_3)", options=payment_options, index=default_pay_index)
    payment_status_jun = st.selectbox("Juin (PAY_4)", options=payment_options, index=default_pay_index)
    payment_status_may = st.selectbox("Mai (PAY_5)", options=payment_options, index=default_pay_index)
    payment_status_apr = st.selectbox("Avril (PAY_6)", options=payment_options, index=default_pay_index)

st.sidebar.markdown("--- ")
st.sidebar.header("Historique Financier (6 derniers mois)")

col3, col4 = st.sidebar.columns(2)

with col3:
    st.markdown("**Relevés de Facturation (BILL_AMT)**")
    bill_statement_sep = st.number_input("Septembre (BILL_AMT1)", value=10000)
    bill_statement_aug = st.number_input("Août (BILL_AMT2)", value=10000)
    bill_statement_jul = st.number_input("Juillet (BILL_AMT3)", value=10000)
    bill_statement_jun = st.number_input("Juin (BILL_AMT4)", value=10000)
    bill_statement_may = st.number_input("Mai (BILL_AMT5)", value=10000)
    bill_statement_apr = st.number_input("Avril (BILL_AMT6)", value=10000)

with col4:
    st.markdown("**Paiements Précédents (PAY_AMT)**")
    previous_payment_sep = st.number_input("Septembre (PAY_AMT1)", min_value=0, value=5000)
    previous_payment_aug = st.number_input("Août (PAY_AMT2)", min_value=0, value=5000)
    previous_payment_jul = st.number_input("Juillet (PAY_AMT3)", min_value=0, value=5000)
    previous_payment_jun = st.number_input("Juin (PAY_AMT4)", min_value=0, value=5000)
    previous_payment_may = st.number_input("Mai (PAY_AMT5)", min_value=0, value=5000)
    previous_payment_apr = st.number_input("Avril (PAY_AMT6)", min_value=0, value=5000)

# --- Création du DataFrame d\"entrée --- 
input_dict = {
    "limit_bal": [limit_bal],
    "sex": [sex],
    "education": [education_label],
    "marriage": [marriage_label],
    "age": [age],
    # Assurez-vous que les noms PAY_0, PAY_2, etc. correspondent à ceux du dataset original
    # Si le dataset original utilisait payment_status_sep, etc., il faut utiliser ces noms ici.
    # Je suppose que le pipeline s\"attend aux noms originaux (ex: PAY_0, BILL_AMT1)
    "payment_status_sep": [payment_status_sep], 
    "payment_status_aug": [payment_status_aug],
    "payment_status_jul": [payment_status_jul],
    "payment_status_jun": [payment_status_jun],
    "payment_status_may": [payment_status_may],
    "payment_status_apr": [payment_status_apr],
    "bill_statement_sep": [bill_statement_sep],
    "bill_statement_aug": [bill_statement_aug],
    "bill_statement_jul": [bill_statement_jul],
    "bill_statement_jun": [bill_statement_jun],
    "bill_statement_may": [bill_statement_may],
    "bill_statement_apr": [bill_statement_apr],
    "previous_payment_sep": [previous_payment_sep],
    "previous_payment_aug": [previous_payment_aug],
    "previous_payment_jul": [previous_payment_jul],
    "previous_payment_jun": [previous_payment_jun],
    "previous_payment_may": [previous_payment_may],
    "previous_payment_apr": [previous_payment_apr],
}

input_data = pd.DataFrame(input_dict)

# --- Prédiction et Affichage --- 
if st.sidebar.button("Prédire", type="primary"):
    prediction, probability_raw, probability_percent = make_prediction(input_data)

    if prediction is not None:
        st.header("Résultats de la Prédiction")
        res_col1, res_col2 = st.columns([1, 2]) 

        with res_col1:
            st.subheader("Prédiction")
            prob_default = probability_percent[0][1]
            prob_no_default = probability_percent[0][0]

            if prediction[0] == 1:
                st.error(f"**Défaut Prédit**")
                st.metric(label="Probabilité de Défaut", value=f"{prob_default:.2f}%")
                # Ajout du paragraphe explicite
                st.write("**Conclusion : Le client présente un risque élevé de défaut de paiement.**") 
            else:
                st.success(f"**Pas de Défaut Prédit**")
                st.metric(label="Probabilité de Non-Défaut", value=f"{prob_no_default:.2f}%")
                # Ajout du paragraphe explicite
                st.write("**Conclusion : Le client ne présente pas de risque élevé de défaut de paiement.**")

        with res_col2:
            st.subheader("Probabilités Prédites")
            prob_df = pd.DataFrame({
                "Classe": ["Pas de Défaut", "Défaut"],
                "Probabilité (%)": [prob_no_default, prob_default]
            })
            fig_prob = px.bar(prob_df, x="Classe", y="Probabilité (%)", 
                         text="Probabilité (%)", 
                         color="Classe", 
                         color_discrete_map={"Pas de Défaut": "#28a745", "Défaut": "#dc3545"},
                         labels={"Probabilité (%)": "Probabilité (%)"})
            fig_prob.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            fig_prob.update_layout(uniformtext_minsize=8, uniformtext_mode="hide", showlegend=False)
            st.plotly_chart(fig_prob, use_container_width=True)
        
    else:
        st.warning("La prédiction n'a pas pu être effectuée.")

else:
    st.info("Veuillez saisir les informations du client dans la barre latérale et cliquer sur 'Prédire'.")

# Ajouter une section "A propos"
st.sidebar.markdown("--- ")
st.sidebar.header("À Propos")
st.sidebar.info(
    "Application de démonstration pour la prédiction de défaut de paiement. "
    "Développée avec Streamlit, Scikit-learn. "
    "Basée sur le dataset UCI Credit Card Default."
)
