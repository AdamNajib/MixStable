# app.py
import streamlit as st
import contextvars
from io import BytesIO
import numpy as np
import warnings

# Suppress R warnings for cleaner Streamlit output
warnings.filterwarnings("ignore", message="R is not initialized by the main thread")

# 🔧 Initialize RPy2 conversions once at startup
import interface.rpy2_init

# Import robust wrappers instead of original functions
from alpha_stable_mixture.visualization import plot_final_mixture_fit
from alpha_stable_mixture.metrics import compute_model_metrics
from alpha_stable_mixture.testing_data import analyse_stable_distribution

# Import the ORIGINAL EM methods from alpha_stable_mixture (DON'T CHANGE)
from alpha_stable_mixture.em_methode import (
    em_estimate_stable_recursive_ecf,
    em_estimate_stable_kernel_ecf,
    em_estimate_stable_weighted_ols,
    em_estimate_stable_from_cdf
)

# Import our INTERFACE wrapper (this is what we control)
from interface.em_robust import robust_em_stable_mixture

from interface.preprocess import read_csv_with_auto_delimiter, extract_serial_intervals

# -------------------- Language Configuration --------------------
LANGUAGES = {
    "en": {
        "title": "📊 Mixstable — Alpha-Stable Mixture Estimator",
        "sidebar_title": "Mixstable",
        "upload_file": "📁 Upload a CSV file",
        "stability_test": "⚗️ Stability Test",
        "em_estimation": "📊 EM Estimation",
        "language": "🌐 Language",
        "theme": "🎨 Theme",
        "light": "☀️ Light",
        "dark": "🌙 Dark",
        "run_stability": "Run Stability Test",
        "run_em": "Run EM Algorithm",
        "running_stability": "Running stability analysis via R...",
        "running_em": "🔄 Running EM algorithm...",
        "stability_complete": "✅ Stability test complete.",
        "em_complete": "🎯 EM estimation complete.",
        "no_numeric": "❌ No numeric column found. Please include a serial interval column.",
        "using_column": "📈 Using column:",
        "no_serial_found": "⚠️ No serial interval column found. Using:",
        "analyzing_points": "📊 Analyzing {} data points",
        "no_valid_data": "❌ No valid data points found after removing NaN values.",
        "download_plot": "📥 Download Plot",
        "plot_not_generated": "⚠️ Plot file not generated",
        "error_stability": "❌ Error during stability analysis:",
        "error_em": "❌ Error during EM estimation:",
        "error_processing": "❌ Error processing file:",
        "error_extracting": "❌ Error extracting data:",
        "r_interface_check": "This might be due to R interface issues. Please check your R installation and required packages.",
        "select_data": "📂 Please select a data source, upload a CSV file, and choose a mode from the sidebar.",
        "how_to_use": "📖 How to use this app:",
        "upload_csv": "Upload a CSV file containing your data",
        "choose_mode": "Choose a mode:",
        "click_button": "Click the respective button to run the analysis",
        "expected_format": "📊 Expected data format:",
        "csv_should_contain": "Your CSV should contain a numeric column with one of these names:",
        "or_any_numeric": "Or any other numeric column (the app will use the first numeric column found).",
        "parameters": "📋 Parameters",
        "weights": "Weights:",
        "log_likelihood": "Log-likelihood:",
        "fit_plot": "📉 Fit Plot",
        "download_fit_plot": "📥 Download Fit Plot",
        "model_metrics": "📊 Model Metrics",
        "component": "Component",
        "choose_method": "Choose estimation method",
        "could_not_compute": "⚠️ Could not compute metrics:",
        "plot_not_found": "⚠️ Plot file not found"
    },
    "fr": {
        "title": "📊 Mixstable — Estimateur de Mélange Alpha-Stable",
        "sidebar_title": "Mixstable",
        "upload_file": "📁 Télécharger un fichier CSV",
        "stability_test": "⚗️ Test de Stabilité",
        "em_estimation": "📊 Estimation EM",
        "language": "🌐 Langue",
        "theme": "🎨 Thème",
        "light": "☀️ Clair",
        "dark": "🌙 Sombre",
        "run_stability": "Lancer le Test de Stabilité",
        "run_em": "Lancer l'Algorithme EM",
        "running_stability": "Exécution de l'analyse de stabilité via R...",
        "running_em": "🔄 Exécution de l'algorithme EM...",
        "stability_complete": "✅ Test de stabilité terminé.",
        "em_complete": "🎯 Estimation EM terminée.",
        "no_numeric": "❌ Aucune colonne numérique trouvée. Veuillez inclure une colonne d'intervalle sériel.",
        "using_column": "📈 Utilisation de la colonne:",
        "no_serial_found": "⚠️ Aucune colonne d'intervalle sériel trouvée. Utilisation de:",
        "analyzing_points": "📊 Analyse de {} points de données",
        "no_valid_data": "❌ Aucun point de données valide trouvé après suppression des valeurs NaN.",
        "download_plot": "📥 Télécharger le Graphique",
        "plot_not_generated": "⚠️ Fichier graphique non généré",
        "error_stability": "❌ Erreur lors de l'analyse de stabilité:",
        "error_em": "❌ Erreur lors de l'estimation EM:",
        "error_processing": "❌ Erreur de traitement du fichier:",
        "error_extracting": "❌ Erreur d'extraction des données:",
        "r_interface_check": "Cela peut être dû à des problèmes d'interface R. Veuillez vérifier votre installation R et les packages requis.",
        "select_data": "📂 Veuillez sélectionner une source de données, télécharger un fichier CSV et choisir un mode dans la barre latérale.",
        "how_to_use": "📖 Comment utiliser cette application:",
        "upload_csv": "Téléchargez un fichier CSV contenant vos données",
        "choose_mode": "Choisissez un mode:",
        "click_button": "Cliquez sur le bouton respectif pour lancer l'analyse",
        "expected_format": "📊 Format de données attendu:",
        "csv_should_contain": "Votre CSV doit contenir une colonne numérique avec l'un de ces noms:",
        "or_any_numeric": "Ou toute autre colonne numérique (l'application utilisera la première colonne numérique trouvée).",
        "parameters": "📋 Paramètres",
        "weights": "Poids:",
        "log_likelihood": "Log-vraisemblance:",
        "fit_plot": "📉 Graphique d'Ajustement",
        "download_fit_plot": "📥 Télécharger le Graphique d'Ajustement",
        "model_metrics": "📊 Métriques du Modèle",
        "component": "Composant",
        "choose_method": "Choisir la méthode d'estimation",
        "could_not_compute": "⚠️ Impossible de calculer les métriques:",
        "plot_not_found": "⚠️ Fichier graphique non trouvé"
    }
}

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="Mixstable — Alpha-Stable Mixture Estimator",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# -------------------- Theme and Language Setup --------------------
# Initialize session state for theme and language
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'
if 'language' not in st.session_state:
    st.session_state.language = 'en'

# Apply theme CSS
def apply_theme():
    if st.session_state.theme == 'dark':
        st.markdown("""
        <style>
        .stApp {
            background-color: #0e1117;
            color: #fafafa;
        }
        .stSidebar {
            background-color: #262730;
        }
        .stSelectbox > div > div {
            background-color: #262730;
            color: #fafafa;
        }
        .stButton > button {
            background-color: #262730;
            color: #fafafa;
            border: 1px solid #4a4a4a;
        }
        .stButton > button:hover {
            background-color: #4a4a4a;
            border: 1px solid #6a6a6a;
        }
        </style>
        """, unsafe_allow_html=True)

apply_theme()

# Get current language texts
def t(key):
    return LANGUAGES[st.session_state.language].get(key, key)

# -------------------- Sidebar --------------------
st.sidebar.title(t("sidebar_title"))

# Language selector
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("🇬🇧 EN" if st.session_state.language == 'fr' else "🇫🇷 FR"):
        st.session_state.language = 'en' if st.session_state.language == 'fr' else 'fr'
        st.rerun()

with col2:
    if st.button("🌙" if st.session_state.theme == 'light' else "☀️"):
        st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
        apply_theme()
        st.rerun()

st.sidebar.markdown("---")

mode = st.sidebar.radio(
    "Choose mode:", 
    [t("stability_test"), t("em_estimation")]
)

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader(t("upload_file"), type="csv")

# -------------------- Main Panel --------------------
st.title(t("title"))

if uploaded_file:
    try:
        df = read_csv_with_auto_delimiter(uploaded_file)

        if mode == t("stability_test"):
            st.subheader("🧪 " + t("stability_test"))

            possible_cols = ["serial_interval", "serial_interval_mean_based", "mean_serial_interval"]
            serial_col = next((col for col in possible_cols if col in df.columns), None)

            if serial_col:
                x = df[serial_col].astype(float)
                st.info(f"{t('using_column')} `{serial_col}`")
            else:
                numeric_cols = df.select_dtypes(include=[float, int]).columns
                if len(numeric_cols) == 0:
                    st.error(t("no_numeric"))
                    st.stop()
                x = df[numeric_cols[0]].astype(float)
                st.warning(f"{t('no_serial_found')} `{numeric_cols[0]}`")

            # Remove NaN values
            x = x.dropna()
            if len(x) == 0:
                st.error(t("no_valid_data"))
                st.stop()

            st.info(t("analyzing_points").format(len(x)))

            if st.button(t("run_stability")):
                with st.spinner(t("running_stability")):
                    try:
                        # Direct call without complex context handling for stability test
                        result = analyse_stable_distribution(x.values, "serial_interval_result")
                        
                        # Handle both old format (tuple) and new format (string)
                        if isinstance(result, tuple):
                            result_text, fig_path = result
                        else:
                            result_text = result
                            fig_path = "stability_test_plot.png"

                        st.success(t("stability_complete"))
                        st.markdown(result_text, unsafe_allow_html=False)
                        
                        # Show plot if it exists
                        try:
                            st.image("stability_test_plot.png", caption="Distribution Comparison")
                            
                            # Offer download
                            with open("stability_test_plot.png", "rb") as f:
                                st.download_button(
                                    t("download_plot"), 
                                    f.read(), 
                                    file_name="stability_analysis_plot.png", 
                                    mime="image/png"
                                )
                        except FileNotFoundError:
                            st.warning(t("plot_not_generated"))
                            
                    except Exception as e:
                        st.error(f"{t('error_stability')} {str(e)}")
                        st.error(t("r_interface_check"))

        elif mode == t("em_estimation"):
            st.subheader("⚙️ " + t("em_estimation"))
            
            try:
                data = extract_serial_intervals(df)
                st.info(t("analyzing_points").format(len(data)))
                
                # Method selection - using the ORIGINAL functions for display
                methods = {
                    "ECF - Recursive": em_estimate_stable_recursive_ecf,
                    "ECF - Kernel": em_estimate_stable_kernel_ecf,
                    "ECF - Weighted OLS": em_estimate_stable_weighted_ols,
                    "ECF - From CDF": em_estimate_stable_from_cdf
                }
                
                selected_method = st.selectbox(t("choose_method"), list(methods.keys()))
                estimator_func = methods[selected_method]

                if st.button(t("run_em")):
                    with st.spinner(t("running_em")):
                        try:
                            print(f"🔄 Starting EM with method: {selected_method}")
                            
                            # Use our ROBUST WRAPPER instead of calling the original directly
                            result = robust_em_stable_mixture(
                                data, 
                                u=None,  # Not used by the actual functions
                                estimator_func=estimator_func,
                                max_iter=300, 
                                epsilon=1e-3
                            )

                            st.success(t("em_complete"))
                            
                            # Display results
                            st.markdown(f"### {t('parameters')}")
                            for i, param in enumerate(result["params_list"]):
                                if isinstance(param, dict):
                                    # If params are returned as dict
                                    param_str = f"α={param['alpha']:.3f}, β={param['beta']:.3f}, γ={param['gamma']:.3f}, δ={param['delta']:.3f}"
                                else:
                                    # If params are returned as list
                                    param_str = f"α={param[0]:.3f}, β={param[1]:.3f}, γ={param[2]:.3f}, δ={param[3]:.3f}"
                                st.markdown(f"**{t('component')} {i+1}:** `{param_str}`")
                            
                            st.write(f"**{t('weights')}**: {result['weights']}")
                            st.write(f"**{t('log_likelihood')}**: {result['log_likelihood']:.3f}")
                            
                            # Show method used
                            if 'method' in result:
                                st.info(f"**Method used**: {result['method']}")

                            # Generate and show plot
                            st.markdown(f"### {t('fit_plot')}")
                            try:
                                # Convert dict params to list format for plotting
                                plot_params = []
                                for param in result["params_list"]:
                                    if isinstance(param, dict):
                                        plot_params.append([param['alpha'], param['beta'], param['gamma'], param['delta']])
                                    else:
                                        plot_params.append(param)
                                
                                # Call plot function with correct signature (p1, p2, w) - NO weights keyword
                                plot_final_mixture_fit(data, plot_params[0], plot_params[1], result["weights"][0])
                                
                                # Show plot
                                st.image("mixture_alpha_stable_fit_final.png")
                                with open("mixture_alpha_stable_fit_final.png", "rb") as f:
                                    st.download_button(
                                        t("download_fit_plot"), 
                                        f.read(), 
                                        file_name="mixture_fit.png", 
                                        mime="image/png"
                                    )
                            except FileNotFoundError:
                                st.warning(t("plot_not_found"))
                            except Exception as plot_error:
                                st.warning(f"Could not generate plot: {plot_error}")

                            # Compute metrics - FIX: Handle the parameter format correctly
                            st.markdown(f"### {t('model_metrics')}")
                            try:
                                # Your compute_model_metrics expects (data, params) where params is [alpha, beta, scale, location]
                                # But you have a mixture model, so we need to compute metrics for each component
                                
                                metrics_results = {}
                                for i, params in enumerate(result["params_list"]):
                                    if isinstance(params, dict):
                                        param_list = [params['alpha'], params['beta'], params['gamma'], params['delta']]
                                    else:
                                        param_list = params
                                    
                                    component_metrics = compute_model_metrics(data, param_list)
                                    metrics_results[f"Component_{i+1}"] = component_metrics
                                
                                # Add overall mixture metrics
                                mixture_ll = result['log_likelihood']
                                n_params = 8  # 4 params per component * 2 components
                                n_data = len(data)
                                
                                metrics_results["Mixture"] = {
                                    "log_likelihood": mixture_ll,
                                    "AIC": -2 * mixture_ll + 2 * n_params,
                                    "BIC": -2 * mixture_ll + n_params * np.log(n_data),
                                    "weights": result["weights"]
                                }
                                
                                st.json(metrics_results)
                            except Exception as e:
                                st.warning(f"{t('could_not_compute')} {str(e)}")
                                st.write("Error details:", str(e))
                                
                        except Exception as e:
                            st.error(f"{t('error_em')} {str(e)}")
                            st.info("The algorithm used fallback methods due to R interface issues.")
                            
            except Exception as e:
                st.error(f"{t('error_extracting')} {str(e)}")

    except Exception as e:
        st.error(f"{t('error_processing')} {e}")
        st.stop()
else:
    st.info(t("select_data"))
    
    # Add some helpful information
    st.markdown(f"""
    ### {t('how_to_use')}
    
    1. **{t('upload_csv')}**
    2. **{t('choose_mode')}**
       - **{t('stability_test')}**: Test if your data follows an α-stable distribution
       - **{t('em_estimation')}**: Fit a mixture of α-stable distributions
    3. **{t('click_button')}**
    
    ### {t('expected_format')}
    {t('csv_should_contain')}
    - `serial_interval`
    - `serial_interval_mean_based` 
    - `mean_serial_interval`
    
    {t('or_any_numeric')}
    """)