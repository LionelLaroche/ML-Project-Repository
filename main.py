import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import io

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
)

# =========================
# CONFIG APP
# =========================
st.set_page_config(
    page_title="Data App – Auto MPG & Census & Fraude",
    page_icon="📊",
    layout="wide"
)

# =========================
# STYLES GLOBAUX
# =========================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .main-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.6rem;
        font-weight: 800;
        color: #0d1b2a;
        text-align: center;
        letter-spacing: -0.03em;
        margin-bottom: 0.3rem;
    }
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .prediction-card {
        padding: 1.5rem 2rem;
        border-radius: 1rem;
        background: linear-gradient(135deg, #0d1b2a, #1a6b72);
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin-bottom: 1rem;
    }
    .prediction-value {
        font-family: 'Syne', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        margin-top: 0.5rem;
        letter-spacing: -0.02em;
    }
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        flex: 1;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 0.75rem;
        padding: 1rem;
        text-align: center;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    .metric-val {
        font-family: 'Syne', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #0d1b2a;
    }
    .section-title {
        font-family: 'Syne', sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        color: #0d1b2a;
        margin: 1.5rem 0 0.75rem 0;
        border-left: 4px solid #1a6b72;
        padding-left: 0.75rem;
    }
    .info-box {
        background: #f0fdf9;
        border: 1px solid #a7f3d0;
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
        color: #065f46;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
        color: #92400e;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">📊 Plateforme de Prédiction ML</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Exploration interactive · Modèles entraînés · Prédictions en temps réel</div>',
    unsafe_allow_html=True
)

# =========================
# FONCTIONS AUTO MPG
# =========================
@st.cache_data
def load_auto_data():
    df = pd.read_csv("auto-mpg.csv")
    if df["horsepower"].dtype == object:
        df = df[df["horsepower"] != "?"]
        df["horsepower"] = df["horsepower"].astype(float)
    features = ["cylinders", "displacement", "horsepower",
                "weight", "acceleration", "model year", "origin"]
    target = "mpg"
    X = df[features]
    y = df[target]
    return df, X, y, features, target

@st.cache_resource
def train_auto_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = model.score(X_test, y_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return model, r2, mae, rmse, X_test, y_test, y_pred

# =========================
# FONCTIONS CENSUS
# =========================
@st.cache_data
def load_census_data():
    df = pd.read_csv("Census.csv")
    df.replace(" ?", np.nan, inplace=True)
    df.dropna(inplace=True)
    target = "income"
    y = df[target]
    X = df.drop(columns=[target])
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    return df, X, y, target, numeric_features, categorical_features

@st.cache_resource
def train_census_model(X, y, numeric_features, categorical_features):
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    clf = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, max_depth=3, random_state=42)
    model = Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = model.score(X_test, y_test)
    return model, acc, X_test, y_test, y_pred

# =========================
# HELPER : export CSV
# =========================
def df_to_csv_bytes(df):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()

# =========================
# ONGLETS
# =========================
# =========================
# FONCTIONS FRAUDE
# =========================
@st.cache_data
def load_fraud_data():
    df = pd.read_csv("creditcard.csv")
    return df

@st.cache_resource
def train_fraud_model(df):
    X = df.drop("Class", axis=1)
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test  = X_test.copy()
    X_train[["Time", "Amount"]] = scaler.fit_transform(X_train[["Time", "Amount"]])
    X_test[["Time", "Amount"]]  = scaler.transform(X_test[["Time", "Amount"]])

    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train, y_train)

    prob      = model.predict_proba(X_test)
    y_prob    = prob[:, 1]
    y_pred_05 = model.predict(X_test)

    # Courbe ROC + seuil optimal (FPR ≤ 5%)
    fpr_arr, tpr_arr, roc_thresholds = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    valid_idx = np.where(fpr_arr <= 0.05)[0]
    opt_idx   = valid_idx[-1]
    opt_thresh = float(roc_thresholds[opt_idx])

    # Courbe PR + équilibre
    prec_curve, rec_curve, pr_thresholds = precision_recall_curve(y_test, y_prob)
    avg_prec = average_precision_score(y_test, y_prob)
    gaps     = np.abs(prec_curve[:-1] - rec_curve[:-1])
    eq_thresh = float(pr_thresholds[np.argmin(gaps)])

    return (model, scaler, y_test, y_prob, y_pred_05,
            fpr_arr, tpr_arr, auc, opt_thresh,
            prec_curve, rec_curve, pr_thresholds, avg_prec, eq_thresh)

tab1, tab2, tab3 = st.tabs(["🚗 Auto MPG", "👤 Census Income", "💳 Détection de Fraude"])

# ──────────────────────────────────────────────────
# ONGLET 1 : AUTO MPG
# ──────────────────────────────────────────────────
with tab1:
    df_auto, X_auto, y_auto, FEATURES_AUTO, TARGET_AUTO = load_auto_data()
    model_auto, r2_auto, mae_auto, rmse_auto, X_test_auto, y_test_auto, y_pred_auto = train_auto_model(X_auto, y_auto)

    # ─── Métriques du modèle ───
    st.markdown('<div class="section-title">📈 Performances du modèle</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² Score", f"{r2_auto:.3f}")
    c2.metric("MAE", f"{mae_auto:.2f} mpg")
    c3.metric("RMSE", f"{rmse_auto:.2f} mpg")
    c4.metric("Taille dataset", f"{len(df_auto)} voitures")

    # ─── Sections : Prédiction | Exploration ───
    pred_col, explore_col = st.columns([1, 1], gap="large")

    with pred_col:
        st.markdown('<div class="section-title">🔮 Prédiction</div>', unsafe_allow_html=True)

        with st.form("auto_form"):
            a1, a2 = st.columns(2)
            with a1:
                cylinders    = st.number_input("Cylindres",       3,    8,    4,    1)
                displacement = st.number_input("Cylindrée",       60.0, 500.0,150.0,1.0)
                horsepower   = st.number_input("Puissance (HP)",  40.0, 250.0, 90.0,1.0)
            with a2:
                weight       = st.number_input("Poids (lbs)",    1500.0,5500.0,2500.0,10.0)
                acceleration = st.number_input("Accélération 0-60", 8.0,25.0,15.0,0.1)
                model_year   = st.number_input("Année (70–82)",   70,   82,   76,   1)

            origin = st.selectbox("Origine", [1, 2, 3],
                                  format_func=lambda x: {1:"🇺🇸 USA", 2:"🇪🇺 Europe", 3:"🇯🇵 Japon"}[x])
            submitted_auto = st.form_submit_button("🔮 Prédire le MPG", use_container_width=True)

        if submitted_auto:
            input_auto = pd.DataFrame(
                [[cylinders, displacement, horsepower, weight, acceleration, model_year, origin]],
                columns=FEATURES_AUTO
            )
            pred_mpg = model_auto.predict(input_auto)[0]
            avg_mpg  = float(y_auto.mean())
            delta_pct = (pred_mpg - avg_mpg) / avg_mpg * 100

            st.markdown(
                f"""
                <div class="prediction-card">
                    Consommation estimée
                    <div class="prediction-value">{pred_mpg:.2f} MPG</div>
                    <div style="margin-top:0.5rem;font-size:0.9rem;opacity:0.85;">
                        Moyenne du dataset : {avg_mpg:.1f} MPG &nbsp;|&nbsp;
                        Écart : {delta_pct:+.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Comparaison avec véhicules similaires (±1 cylindre)
            similar = df_auto[df_auto["cylinders"] == cylinders]["mpg"]
            if len(similar) > 0:
                st.markdown(
                    f'<div class="info-box">🔍 Parmi les {len(similar)} véhicules à {cylinders} cylindres dans le dataset, '
                    f'le MPG moyen est <strong>{similar.mean():.1f}</strong> '
                    f'(min {similar.min():.1f} – max {similar.max():.1f}).</div>',
                    unsafe_allow_html=True
                )

            # Export résultat
            result_df = input_auto.copy()
            result_df["Predicted MPG"] = pred_mpg
            st.download_button(
                "⬇️ Télécharger ce résultat (CSV)",
                data=df_to_csv_bytes(result_df),
                file_name="prediction_mpg.csv",
                mime="text/csv"
            )

    with explore_col:
        st.markdown('<div class="section-title">🔬 Exploration du dataset</div>', unsafe_allow_html=True)

        # Importance des features
        feat_imp = pd.Series(model_auto.feature_importances_, index=FEATURES_AUTO).sort_values()
        fig_imp, ax_imp = plt.subplots(figsize=(5, 3))
        colors = ["#1a6b72" if v == feat_imp.max() else "#93c5d6" for v in feat_imp.values]
        feat_imp.plot(kind="barh", ax=ax_imp, color=colors)
        ax_imp.set_title("Importance des features", fontsize=10, fontweight="bold")
        ax_imp.set_xlabel("Importance")
        ax_imp.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_imp)
        plt.close()

        # Scatter prédictions vs réalité
        fig_sc, ax_sc = plt.subplots(figsize=(5, 3))
        ax_sc.scatter(y_test_auto, y_pred_auto, alpha=0.5, color="#1a6b72", s=20)
        mn, mx = min(y_test_auto.min(), y_pred_auto.min()), max(y_test_auto.max(), y_pred_auto.max())
        ax_sc.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Parfait")
        ax_sc.set_xlabel("MPG réel")
        ax_sc.set_ylabel("MPG prédit")
        ax_sc.set_title("Prédictions vs Réalité", fontsize=10, fontweight="bold")
        ax_sc.spines[["top","right"]].set_visible(False)
        ax_sc.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_sc)
        plt.close()

    # ─── Exploration interactive ───
    st.markdown('<div class="section-title">📊 Analyse des données</div>', unsafe_allow_html=True)
    ea1, ea2 = st.columns(2)

    with ea1:
        x_var = st.selectbox("Variable X", FEATURES_AUTO, index=3, key="auto_x")
    with ea2:
        color_var = st.selectbox("Couleur par", ["origin", "cylinders"], key="auto_color")

    fig_exp, ax_exp = plt.subplots(figsize=(6, 3.5))
    scatter = ax_exp.scatter(
        df_auto[x_var], df_auto["mpg"],
        c=df_auto[color_var], cmap="viridis", alpha=0.6, s=18
    )
    plt.colorbar(scatter, ax=ax_exp, label=color_var)
    ax_exp.set_xlabel(x_var)
    ax_exp.set_ylabel("MPG")
    ax_exp.set_title(f"MPG vs {x_var}", fontsize=10, fontweight="bold")
    ax_exp.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig_exp)
    plt.close()

    # Aperçu + téléchargement dataset
    with st.expander("📄 Aperçu du dataset Auto MPG"):
        st.dataframe(df_auto, use_container_width=True)
        st.download_button(
            "⬇️ Télécharger le dataset complet",
            data=df_to_csv_bytes(df_auto),
            file_name="auto_mpg.csv",
            mime="text/csv"
        )

# ──────────────────────────────────────────────────
# ONGLET 2 : CENSUS INCOME
# ──────────────────────────────────────────────────
with tab2:
    df_census, X_census, y_census, TARGET_CENSUS, num_cols, cat_cols = load_census_data()
    model_census, acc_census, X_test_cens, y_test_cens, y_pred_cens = train_census_model(
        X_census, y_census, num_cols, cat_cols
    )

    # ─── Métriques ───
    st.markdown('<div class="section-title">📈 Performances du modèle</div>', unsafe_allow_html=True)
    from sklearn.metrics import f1_score, roc_auc_score
    classes = model_census.classes_
    proba_test = model_census.predict_proba(X_test_cens)
    idx_high = list(classes).index(">50K") if ">50K" in classes else 0
    auc = roc_auc_score((y_test_cens == ">50K").astype(int), proba_test[:, idx_high])
    f1  = f1_score(y_test_cens, y_pred_cens, pos_label=">50K")

    cb1, cb2, cb3, cb4 = st.columns(4)
    cb1.metric("Accuracy", f"{acc_census:.3f}")
    cb2.metric("F1-Score (>50K)", f"{f1:.3f}")
    cb3.metric("AUC-ROC", f"{auc:.3f}")
    cb4.metric("Taille dataset", f"{len(df_census):,}")

    # ─── Deux colonnes : Formulaire | Résultat + Stats ───
    pc1, pc2 = st.columns([1, 1], gap="large")

    with pc1:
        st.markdown('<div class="section-title">🧮 Prédiction de revenu</div>', unsafe_allow_html=True)

        with st.form("census_form"):
            b1, b2 = st.columns(2)
            with b1:
                age            = st.number_input("Âge",              17,  90,  35, 1)
                education_num  = st.number_input("Niveau éducation",   1,  16,   9, 1)
                hours_per_week = st.number_input("H travail/semaine",  1,  99,  40, 1)
                capital_gain   = st.number_input("Capital gain",       0,  99999, 0, 100)
                capital_loss   = st.number_input("Capital loss",       0,  99999, 0, 100)
            with b2:
                workclass      = st.selectbox("Workclass",      sorted(df_census["workclass"].unique()))
                education      = st.selectbox("Education",      sorted(df_census["education"].unique()))
                marital_status = st.selectbox("Statut marital", sorted(df_census["marital.status"].unique()))
                occupation     = st.selectbox("Occupation",     sorted(df_census["occupation"].unique()))
                relationship   = st.selectbox("Relation",       sorted(df_census["relationship"].unique()))
                race           = st.selectbox("Race",           sorted(df_census["race"].unique()))
                sex            = st.selectbox("Sexe",           sorted(df_census["sex"].unique()))
                native_country = st.selectbox("Pays d'origine", sorted(df_census["native.country"].unique()))

            submitted_census = st.form_submit_button("🧮 Prédire le revenu", use_container_width=True)

    with pc2:
        st.markdown('<div class="section-title">📊 Résultat & Analyse</div>', unsafe_allow_html=True)

        if submitted_census:
            data_dict = {col: [None] for col in X_census.columns}
            data_dict.update({
                "age": [age], "education.num": [education_num],
                "hours.per.week": [hours_per_week], "capital.gain": [capital_gain],
                "capital.loss": [capital_loss], "workclass": [workclass],
                "education": [education], "marital.status": [marital_status],
                "occupation": [occupation], "relationship": [relationship],
                "race": [race], "sex": [sex], "native.country": [native_country],
            })
            input_census = pd.DataFrame(data_dict)[X_census.columns]
            for col in num_cols:
                input_census[col] = input_census[col].fillna(X_census[col].median())
            for col in cat_cols:
                input_census[col] = input_census[col].fillna(X_census[col].mode()[0])

            pred_income = model_census.predict(input_census)[0]
            proba       = model_census.predict_proba(input_census)[0]
            prob_high   = proba[idx_high]
            prob_low    = 1 - prob_high

            label = ">50K 💰" if pred_income == ">50K" else "≤50K"
            color = "#1a6b72" if pred_income == ">50K" else "#0d1b2a"

            st.markdown(
                f"""
                <div class="prediction-card" style="background: linear-gradient(135deg, #0d1b2a, {color});">
                    Classe prédite (revenu annuel)
                    <div class="prediction-value">{label}</div>
                    <div style="margin-top:0.5rem;font-size:0.9rem;opacity:0.85;">
                        P(>50K) = {prob_high*100:.1f}% &nbsp;|&nbsp; P(≤50K) = {prob_low*100:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Jauge de probabilité
            fig_gauge, ax_g = plt.subplots(figsize=(5, 1.2))
            ax_g.barh(0, prob_low,  color="#e2e8f0", height=0.5)
            ax_g.barh(0, prob_high, left=prob_low, color="#1a6b72", height=0.5)
            ax_g.set_xlim(0, 1)
            ax_g.set_yticks([])
            ax_g.set_xticks([0, 0.25, 0.5, 0.75, 1])
            ax_g.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
            ax_g.set_title(f"Probabilité >50K : {prob_high*100:.1f}%", fontsize=9)
            ax_g.spines[["top","right","left"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_gauge)
            plt.close()

            # Comparaison profil vs dataset
            st.markdown("**Votre profil vs dataset**")
            comp_data = {
                "Variable": ["Âge", "Heures/sem.", "Éducation (num)"],
                "Votre valeur": [age, hours_per_week, education_num],
                "Moyenne dataset": [
                    round(df_census["age"].mean(), 1),
                    round(df_census["hours.per.week"].mean(), 1),
                    round(df_census["education.num"].mean(), 1)
                ]
            }
            st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

            # Export résultat
            result_df = input_census.copy()
            result_df["Predicted Income"] = pred_income
            result_df["Prob >50K"] = round(prob_high, 4)
            st.download_button(
                "⬇️ Télécharger ce résultat (CSV)",
                data=df_to_csv_bytes(result_df),
                file_name="prediction_Census.csv",
                mime="text/csv"
            )
        else:
            st.info("Remplissez le formulaire et cliquez sur **Prédire** pour voir les résultats.")

            # Matrice de confusion du modèle
            st.markdown("**Matrice de confusion (jeu de test)**")
            cm = confusion_matrix(y_test_cens, y_pred_cens, labels=classes)
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=classes, yticklabels=classes, ax=ax_cm)
            ax_cm.set_xlabel("Prédit")
            ax_cm.set_ylabel("Réel")
            ax_cm.set_title("Matrice de confusion", fontsize=9, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_cm)
            plt.close()

    # ─── Analyse statistique ───
    st.markdown('<div class="section-title">📊 Analyse des données Census</div>', unsafe_allow_html=True)
    stat1, stat2 = st.columns(2)

    with stat1:
        # Distribution âge par classe de revenu
        fig_age, ax_age = plt.subplots(figsize=(5, 3))
        for cls, color in zip(classes, ["#1a6b72", "#f59e0b"]):
            subset = df_census[df_census["income"] == cls]["age"]
            ax_age.hist(subset, bins=20, alpha=0.6, label=cls, color=color, density=True)
        ax_age.set_xlabel("Âge")
        ax_age.set_ylabel("Densité")
        ax_age.set_title("Distribution de l'âge par revenu", fontsize=9, fontweight="bold")
        ax_age.legend(fontsize=8)
        ax_age.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_age)
        plt.close()

    with stat2:
        # Répartition par education.num
        edu_income = df_census.groupby(["education.num", "income"]).size().unstack(fill_value=0)
        edu_income_pct = edu_income.div(edu_income.sum(axis=1), axis=0)
        fig_edu, ax_edu = plt.subplots(figsize=(5, 3))
        edu_income_pct.plot(kind="bar", ax=ax_edu, color=["#1a6b72", "#f59e0b"], alpha=0.85)
        ax_edu.set_xlabel("Niveau éducation (num)")
        ax_edu.set_ylabel("Proportion")
        ax_edu.set_title("Revenu par niveau d'éducation", fontsize=9, fontweight="bold")
        ax_edu.legend(fontsize=8)
        ax_edu.spines[["top","right"]].set_visible(False)
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig_edu)
        plt.close()

    # Aperçu + téléchargement dataset
    with st.expander("📄 Aperçu du dataset Census"):
        # Filtre rapide
        income_filter = st.multiselect(
            "Filtrer par classe de revenu",
            options=list(df_census["income"].unique()),
            default=list(df_census["income"].unique()),
            key="census_filter"
        )
        filtered = df_census[df_census["income"].isin(income_filter)]
        st.write(f"{len(filtered):,} lignes affichées")
        st.dataframe(filtered, use_container_width=True)
        st.download_button(
            "⬇️ Télécharger (filtré)",
            data=df_to_csv_bytes(filtered),
            file_name="census_filtered.csv",
            mime="text/csv"
        )

# ──────────────────────────────────────────────────
# ONGLET 3 : DÉTECTION DE FRAUDE
# ──────────────────────────────────────────────────
with tab3:
    try:
        df_fraud = load_fraud_data()
    except FileNotFoundError:
        st.error("❌ Fichier `creditcard.csv` introuvable. Placez-le à la racine du projet.")
        st.stop()

    (model_fraud, scaler_fraud, y_test_fr, y_prob_fr, y_pred_05_fr,
     fpr_arr, tpr_arr, auc_fr, opt_thresh,
     prec_curve, rec_curve, pr_thresholds, avg_prec, eq_thresh) = train_fraud_model(df_fraud)

    # ── Métriques globales ──
    st.markdown('<div class="section-title">📈 Performances du modèle (Logistic Regression · class_weight=balanced)</div>', unsafe_allow_html=True)

    from sklearn.metrics import f1_score, precision_score, recall_score
    f1_fr   = f1_score(y_test_fr, y_pred_05_fr)
    prec_fr = precision_score(y_test_fr, y_pred_05_fr, zero_division=0)
    rec_fr  = recall_score(y_test_fr, y_pred_05_fr)

    fm1, fm2, fm3, fm4, fm5 = st.columns(5)
    fm1.metric("AUC-ROC",          f"{auc_fr:.4f}")
    fm2.metric("Avg Precision",    f"{avg_prec:.4f}")
    fm3.metric("F1-Score (Fraude)",f"{f1_fr:.4f}")
    fm4.metric("Recall (Fraude)",  f"{rec_fr:.4f}")
    fm5.metric("Transactions test",f"{len(y_test_fr):,}")

    total_fraud = int(y_test_fr.sum())
    total_legit = len(y_test_fr) - total_fraud
    st.markdown(
        f'<div class="info-box">📊 Jeu de test : <strong>{total_legit:,}</strong> transactions légitimes · '
        f'<strong>{total_fraud}</strong> fraudes ({total_fraud/len(y_test_fr)*100:.2f}% du total)</div>',
        unsafe_allow_html=True
    )

    # ── Comparaison des stratégies de seuil ──
    st.markdown('<div class="section-title">⚖️ Comparaison des stratégies de seuil</div>', unsafe_allow_html=True)

    strategies = {
        "Default (0.50)": 0.5,
        f"ROC-Optimal ({opt_thresh:.4f})": opt_thresh,
        f"PR-Équilibre ({eq_thresh:.4f})": eq_thresh,
    }
    rows = []
    for name, thresh in strategies.items():
        preds = (y_prob_fr >= thresh).astype(int)
        rep   = classification_report(y_test_fr, preds, output_dict=True, zero_division=0)
        cm_t  = confusion_matrix(y_test_fr, preds)
        rows.append({
            "Stratégie":          name,
            "Seuil":              round(thresh, 4),
            "Precision (Fraude)": round(rep["1"]["precision"], 4),
            "Recall (Fraude)":    round(rep["1"]["recall"],    4),
            "F1 (Fraude)":        round(rep["1"]["f1-score"],  4),
            "Faux Positifs":      int(cm_t[0][1]),
            "Fraudes manquées":   int(cm_t[1][0]),
        })
    comp_df = pd.DataFrame(rows).set_index("Stratégie")
    st.dataframe(comp_df, use_container_width=True)

    # ── Graphiques : ROC | PR | Confusion matrix ──
    st.markdown('<div class="section-title">📊 Courbes & Matrices</div>', unsafe_allow_html=True)
    gc1, gc2, gc3 = st.columns(3)

    with gc1:
        fig_roc, ax_r = plt.subplots(figsize=(4.5, 3.5))
        ax_r.plot(fpr_arr, tpr_arr, color="#1a6b72", lw=2,
                  label=f"Logistic Reg. (AUC={auc_fr:.4f})")
        ax_r.plot([0,1],[0,1],"k--",lw=1,alpha=0.4,label="Aléatoire")
        ax_r.fill_between(fpr_arr, tpr_arr, alpha=0.07, color="#1a6b72")
        ax_r.axvline(0.05, color="#f59e0b", lw=1.2, ls="--", label="FPR=5%")
        ax_r.set_xlabel("Taux Faux Positifs", fontsize=8)
        ax_r.set_ylabel("Taux Vrais Positifs", fontsize=8)
        ax_r.set_title("Courbe ROC", fontsize=10, fontweight="bold")
        ax_r.legend(fontsize=7)
        ax_r.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_roc)
        plt.close()

    with gc2:
        fig_pr, ax_p = plt.subplots(figsize=(4.5, 3.5))
        ax_p.plot(rec_curve, prec_curve, color="#4CAF50", lw=2,
                  label=f"PR (AP={avg_prec:.4f})")
        ax_p.set_xlabel("Recall (Fraudes détectées)", fontsize=8)
        ax_p.set_ylabel("Precision (Alertes correctes)", fontsize=8)
        ax_p.set_title("Courbe Precision-Recall", fontsize=10, fontweight="bold")
        ax_p.legend(fontsize=7)
        ax_p.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_pr)
        plt.close()

    with gc3:
        chosen_strat = st.selectbox(
            "Matrice de confusion pour :",
            list(strategies.keys()), key="cm_strat"
        )
        thresh_cm = strategies[chosen_strat]
        preds_cm  = (y_prob_fr >= thresh_cm).astype(int)
        cm_val    = confusion_matrix(y_test_fr, preds_cm)
        fig_cm2, ax_cm2 = plt.subplots(figsize=(4.5, 3.5))
        sns.heatmap(cm_val, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Légitime","Fraude"],
                    yticklabels=["Légitime","Fraude"], ax=ax_cm2)
        ax_cm2.set_xlabel("Prédit", fontsize=8)
        ax_cm2.set_ylabel("Réel",   fontsize=8)
        ax_cm2.set_title(f"Matrice ({chosen_strat})", fontsize=9, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_cm2)
        plt.close()

    # ── Tableau d'impact des seuils ──
    st.markdown('<div class="section-title">🎛️ Impact de chaque seuil sur la détection</div>', unsafe_allow_html=True)
    thresh_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    rows_t = []
    for t in thresh_list:
        p = (y_prob_fr >= t).astype(int)
        rep = classification_report(y_test_fr, p, output_dict=True, zero_division=0)
        rows_t.append({
            "Seuil": t,
            "Precision": round(rep["1"]["precision"], 4),
            "Recall":    round(rep["1"]["recall"],    4),
            "F1-Score":  round(rep["1"]["f1-score"],  4),
            "Fraudes détectées": f"{int(p[y_test_fr==1].sum())}/{total_fraud}",
            "Faux positifs":     int(p[y_test_fr==0].sum()),
        })
    st.dataframe(pd.DataFrame(rows_t), use_container_width=True, hide_index=True)

    # ── Prédiction sur une transaction manuelle ──
    st.markdown('<div class="section-title">🔍 Analyser une transaction</div>', unsafe_allow_html=True)

    pred_col_fr, info_col_fr = st.columns([1, 1], gap="large")

    with pred_col_fr:
        st.markdown("**Saisie de la transaction**")
        threshold_choice = st.selectbox(
            "Stratégie de seuil",
            list(strategies.keys()),
            key="thresh_pred"
        )
        active_thresh = strategies[threshold_choice]

        with st.form("fraud_form"):
            st.caption("Entrez les valeurs des features (V1–V28 sont des composantes PCA anonymisées)")
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                f_time   = st.number_input("Time",   value=50000.0, step=1000.0)
                f_amount = st.number_input("Amount", value=50.0,    step=1.0)
                f_v1  = st.number_input("V1",  value=0.0, step=0.1, format="%.3f")
                f_v2  = st.number_input("V2",  value=0.0, step=0.1, format="%.3f")
                f_v3  = st.number_input("V3",  value=0.0, step=0.1, format="%.3f")
                f_v4  = st.number_input("V4",  value=0.0, step=0.1, format="%.3f")
                f_v5  = st.number_input("V5",  value=0.0, step=0.1, format="%.3f")
                f_v6  = st.number_input("V6",  value=0.0, step=0.1, format="%.3f")
                f_v7  = st.number_input("V7",  value=0.0, step=0.1, format="%.3f")
                f_v8  = st.number_input("V8",  value=0.0, step=0.1, format="%.3f")
            with fc2:
                f_v9  = st.number_input("V9",  value=0.0, step=0.1, format="%.3f")
                f_v10 = st.number_input("V10", value=0.0, step=0.1, format="%.3f")
                f_v11 = st.number_input("V11", value=0.0, step=0.1, format="%.3f")
                f_v12 = st.number_input("V12", value=0.0, step=0.1, format="%.3f")
                f_v13 = st.number_input("V13", value=0.0, step=0.1, format="%.3f")
                f_v14 = st.number_input("V14", value=0.0, step=0.1, format="%.3f")
                f_v15 = st.number_input("V15", value=0.0, step=0.1, format="%.3f")
                f_v16 = st.number_input("V16", value=0.0, step=0.1, format="%.3f")
                f_v17 = st.number_input("V17", value=0.0, step=0.1, format="%.3f")
            with fc3:
                f_v18 = st.number_input("V18", value=0.0, step=0.1, format="%.3f")
                f_v19 = st.number_input("V19", value=0.0, step=0.1, format="%.3f")
                f_v20 = st.number_input("V20", value=0.0, step=0.1, format="%.3f")
                f_v21 = st.number_input("V21", value=0.0, step=0.1, format="%.3f")
                f_v22 = st.number_input("V22", value=0.0, step=0.1, format="%.3f")
                f_v23 = st.number_input("V23", value=0.0, step=0.1, format="%.3f")
                f_v24 = st.number_input("V24", value=0.0, step=0.1, format="%.3f")
                f_v25 = st.number_input("V25", value=0.0, step=0.1, format="%.3f")
                f_v26 = st.number_input("V26", value=0.0, step=0.1, format="%.3f")
                f_v27 = st.number_input("V27", value=0.0, step=0.1, format="%.3f")
                f_v28 = st.number_input("V28", value=0.0, step=0.1, format="%.3f")

            submitted_fraud = st.form_submit_button("🔎 Analyser cette transaction", use_container_width=True)

    with info_col_fr:
        st.markdown("**Résultat de l'analyse**")

        if submitted_fraud:
            raw_input = pd.DataFrame([[
                f_time, f_v1, f_v2, f_v3, f_v4, f_v5, f_v6, f_v7, f_v8,
                f_v9, f_v10, f_v11, f_v12, f_v13, f_v14, f_v15, f_v16, f_v17,
                f_v18, f_v19, f_v20, f_v21, f_v22, f_v23, f_v24, f_v25, f_v26,
                f_v27, f_v28, f_amount
            ]], columns=df_fraud.drop("Class", axis=1).columns)

            input_scaled = raw_input.copy()
            input_scaled[["Time", "Amount"]] = scaler_fraud.transform(input_scaled[["Time", "Amount"]])

            prob_fraud = model_fraud.predict_proba(input_scaled)[0][1]
            is_fraud   = prob_fraud >= active_thresh
            label      = "🚨 FRAUDE DÉTECTÉE" if is_fraud else "✅ Transaction Légitime"
            card_color = "#7f1d1d" if is_fraud else "#1a6b72"

            st.markdown(
                f"""
                <div class="prediction-card" style="background: linear-gradient(135deg, #0d1b2a, {card_color});">
                    {label}
                    <div class="prediction-value">{prob_fraud*100:.1f}%</div>
                    <div style="margin-top:0.5rem;font-size:0.9rem;opacity:0.85;">
                        Probabilité de fraude · Seuil : {active_thresh:.4f} ({threshold_choice})
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Mini jauge de risque
            fig_g2, ax_g2 = plt.subplots(figsize=(5, 1))
            ax_g2.barh(0, 1 - prob_fraud, color="#d1fae5", height=0.5)
            ax_g2.barh(0, prob_fraud, left=1 - prob_fraud,
                       color="#ef4444" if is_fraud else "#1a6b72", height=0.5)
            ax_g2.axvline(active_thresh, color="#f59e0b", lw=2, ls="--")
            ax_g2.set_xlim(0, 1)
            ax_g2.set_yticks([])
            ax_g2.set_xticks([0, 0.25, 0.5, 0.75, 1])
            ax_g2.set_xticklabels(["0%","25%","50%","75%","100%"], fontsize=8)
            ax_g2.set_title(f"Risque : {prob_fraud*100:.1f}%  |  Seuil : {active_thresh:.2f}", fontsize=9)
            ax_g2.spines[["top","right","left"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_g2)
            plt.close()

            if is_fraud:
                st.markdown(
                    '<div class="warning-box">⚠️ Cette transaction dépasse le seuil de détection. '
                    'Elle serait bloquée ou mise en révision selon la politique de la banque.</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="info-box">✔️ Transaction en dessous du seuil de risque. '
                    'Elle serait autorisée normalement.</div>',
                    unsafe_allow_html=True
                )

            # Export
            export_row = raw_input.copy()
            export_row["Prob_Fraude"] = round(prob_fraud, 4)
            export_row["Decision"]    = "FRAUDE" if is_fraud else "LEGITIME"
            st.download_button(
                "⬇️ Exporter ce résultat (CSV)",
                data=df_to_csv_bytes(export_row),
                file_name="resultat_fraude.csv",
                mime="text/csv"
            )
        else:
            # Afficher distribution Amount par classe
            st.markdown("**Distribution du montant (Amount) par classe**")
            fig_amt, ax_amt = plt.subplots(figsize=(5, 3.5))
            fraud_vals  = df_fraud[df_fraud["Class"]==1]["Amount"]
            legit_vals  = df_fraud[df_fraud["Class"]==0]["Amount"]
            ax_amt.hist(legit_vals.clip(upper=500),  bins=50, alpha=0.6,
                        color="#1a6b72", label="Légitime", density=True)
            ax_amt.hist(fraud_vals.clip(upper=500),  bins=50, alpha=0.7,
                        color="#ef4444", label="Fraude", density=True)
            ax_amt.set_xlabel("Montant (Amount, tronqué à 500€)", fontsize=8)
            ax_amt.set_ylabel("Densité", fontsize=8)
            ax_amt.set_title("Distribution des montants", fontsize=9, fontweight="bold")
            ax_amt.legend(fontsize=8)
            ax_amt.spines[["top","right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_amt)
            plt.close()

            st.info("Remplissez le formulaire à gauche et cliquez sur **Analyser** pour détecter une fraude.")

    # ── Aperçu dataset ──
    with st.expander("📄 Aperçu du dataset Credit Card"):
        st.write(f"**{len(df_fraud):,} transactions** · {df_fraud['Class'].sum()} fraudes ({df_fraud['Class'].mean()*100:.3f}%)")
        st.dataframe(df_fraud.head(100), use_container_width=True)
        st.download_button(
            "⬇️ Télécharger un échantillon (1000 lignes)",
            data=df_to_csv_bytes(df_fraud.sample(min(1000, len(df_fraud)), random_state=42)),
            file_name="creditcard_sample.csv",
            mime="text/csv"
        )
