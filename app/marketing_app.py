
import streamlit as st
import pandas as pd
import altair as alt
import joblib
from sklearn.metrics import accuracy_score, f1_score

st.set_page_config(
    page_title="Marketing Campaign Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
)

st.title("Campagne marketing - Analyse et Prédiction Dépôt")

#  Charger les modèles et encodeur 
rf_model = joblib.load("app/random_forest_model.pkl")
lr_model = joblib.load("app/logistic_regression_model.pkl")
dt_model = joblib.load("app/decision_tree_model.pkl")
encoder = joblib.load("app/encoder.pkl")

# Layout : 2 colonnes 
left_col, right_col = st.columns([1, 1])


# Colonne gauche : Upload CSV + Métriques + Graphiques

with left_col:
    st.subheader("Importer le dataset")
    uploaded_file = st.file_uploader("Téléchargez le fichier CSV", type="csv")

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.success("Dataset chargé !")

        # Calcul métriques 
        total = len(data)
        yes_count = data['deposit'].value_counts().get('yes', 0)
        no_count = data['deposit'].value_counts().get('no', 0)
        st.metric("Total enregistrements", total)
        st.metric("Nombre YES", yes_count)
        st.metric("Nombre NO", no_count)

        #  Graphique distribution globale 
        st.subheader("Distribution globale du dépôt")
        dist_chart = alt.Chart(data).mark_bar().encode(
            x=alt.X('deposit:N', title='Deposit'),
            y=alt.Y('count():Q', title='Nombre'),
            color='deposit:N',
            tooltip=['deposit', 'count()']
        )
        st.altair_chart(dist_chart, use_container_width=True)

        #  Graphiques par catégories 
        categorical_cols = ["job","marital","education","default","housing","loan","contact","month","poutcome"]
        for col in categorical_cols:
            st.subheader(f"Distribution par {col.capitalize()}")
            chart = alt.Chart(data).mark_bar().encode(
                x=alt.X(f"{col}:N", sort='-y', title=col.capitalize()),
                y=alt.Y('count():Q', title='Nombre'),
                color='deposit:N',
                tooltip=[col, 'count()']
            )
            st.altair_chart(chart, use_container_width=True)

        # Table complète 
        st.subheader("Table complète")
        st.dataframe(data)

        #  Comparaison des modèles 
        st.subheader("Comparaison des modèles")
        X_cat = encoder.transform(data[categorical_cols])
        X = pd.concat([data.drop(columns=categorical_cols + ["deposit"]),
                       pd.DataFrame(X_cat, columns=encoder.get_feature_names_out(categorical_cols))],
                      axis=1)
        y = data['deposit'].map({'no':0, 'yes':1})

        performances = []
        for name, model in [("Random Forest", rf_model),
                            ("Logistic Regression", lr_model),
                            ("Decision Tree", dt_model)]:
            y_pred = model.predict(X)
            acc = accuracy_score(y, y_pred)
            f1 = f1_score(y, y_pred)
            performances.append({"Modèle": name, "Accuracy": acc, "F1-score": f1})

        perf_df = pd.DataFrame(performances)
        st.dataframe(perf_df)

        # Graphique comparaison
        perf_chart = alt.Chart(perf_df.melt(id_vars="Modèle", value_vars=["Accuracy","F1-score"])).mark_bar().encode(
            x="Modèle:N",
            y="value:Q",
            color="variable:N",
            tooltip=["Modèle","variable","value"]
        )
        st.altair_chart(perf_chart, use_container_width=True)

  
# ANALYSE DES CLUSTERS

st.subheader("Segmentation des clients (Clustering)")

try:
    cluster_data = pd.read_csv("data/marketing_with_clusters.csv")

    st.success("Clusters chargés avec succès !")

    # --- Distribution des clusters
    st.subheader("Distribution des clusters")
    cluster_dist = alt.Chart(cluster_data).mark_bar().encode(
        x=alt.X('cluster:N', title='Cluster'),
        y=alt.Y('count():Q', title='Nombre de clients'),
        color='cluster:N',
        tooltip=['cluster', 'count()']
    )
    st.altair_chart(cluster_dist, use_container_width=True)

    # Dépôt par cluster
    st.subheader("Dépôt (Yes/No) par cluster")
    deposit_cluster = alt.Chart(cluster_data).mark_bar().encode(
        x=alt.X('cluster:N'),
        y=alt.Y('count():Q'),
        color='deposit:N',
        tooltip=['cluster', 'deposit', 'count()']
    )
    st.altair_chart(deposit_cluster, use_container_width=True)

    # Balance moyenne par cluster
    st.subheader("Balance moyenne par cluster")
    balance_chart = alt.Chart(cluster_data).mark_bar().encode(
        x='cluster:N',
        y='mean(balance):Q',
        color='cluster:N',
        tooltip=['cluster', 'mean(balance)']
    )
    st.altair_chart(balance_chart, use_container_width=True)

    #  Durée moyenne par cluster
    st.subheader("Durée moyenne des appels par cluster")
    duration_chart = alt.Chart(cluster_data).mark_bar().encode(
        x='cluster:N',
        y='mean(duration):Q',
        color='cluster:N',
        tooltip=['cluster', 'mean(duration)']
    )
    st.altair_chart(duration_chart, use_container_width=True)

except:
    st.warning("⚠️ Lance d'abord kmeans_clustering.py pour générer les clusters")


# Colonne droite : Formulaire Prédiction en temps réel

with right_col:
    st.subheader("Prédiction temps réel pour un nouveau client")

    # Options par défaut ou dataset si uploadé
    if uploaded_file:
        job_options = list(data['job'].unique())
        marital_options = list(data['marital'].unique())
        education_options = list(data['education'].unique())
        default_options = list(data['default'].unique())
        housing_options = list(data['housing'].unique())
        loan_options = list(data['loan'].unique())
        contact_options = list(data['contact'].unique())
        month_options = list(data['month'].unique())
        poutcome_options = list(data['poutcome'].unique())
    else:
        job_options = ["admin.","blue-collar","technician"]
        marital_options = ["married","single","divorced"]
        education_options = ["primary","secondary","tertiary"]
        default_options = ["yes","no"]
        housing_options = ["yes","no"]
        loan_options = ["yes","no"]
        contact_options = ["cellular","telephone"]
        month_options = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
        poutcome_options = ["success","failure","other","unknown"]

    #  Formulaire avec session_state pour réinitialisation
    if 'reset_counter' not in st.session_state:
        st.session_state.reset_counter = 0

    form = st.form("prediction_form")
    key_suffix = st.session_state.reset_counter

    age = form.number_input("Âge", min_value=18, max_value=100, value=30, key=f"age_{key_suffix}")
    job = form.selectbox("Emploi", job_options, key=f"job_{key_suffix}")
    marital = form.selectbox("État civil", marital_options, key=f"marital_{key_suffix}")
    education = form.selectbox("Éducation", education_options, key=f"education_{key_suffix}")
    default = form.selectbox("Crédit par défaut ?", default_options, key=f"default_{key_suffix}")
    balance = form.number_input("Balance", value=1000, key=f"balance_{key_suffix}")
    housing = form.selectbox("Prêt logement ?", housing_options, key=f"housing_{key_suffix}")
    loan = form.selectbox("Prêt personnel ?", loan_options, key=f"loan_{key_suffix}")
    contact = form.selectbox("Contact", contact_options, key=f"contact_{key_suffix}")
    day = form.number_input("Jour du contact", min_value=1, max_value=31, value=15, key=f"day_{key_suffix}")
    month = form.selectbox("Mois du contact", month_options, key=f"month_{key_suffix}")
    duration = form.number_input("Durée du dernier appel", value=100, key=f"duration_{key_suffix}")
    campaign = form.number_input("Nombre de contacts durant cette campagne", value=1, key=f"campaign_{key_suffix}")
    pdays = form.number_input("Nombre de jours depuis le dernier contact", value=-1, key=f"pdays_{key_suffix}")
    previous = form.number_input("Nombre de contacts précédents", value=0, key=f"previous_{key_suffix}")
    poutcome = form.selectbox("Résultat campagne précédente", poutcome_options, key=f"poutcome_{key_suffix}")

    submit = form.form_submit_button("Prédire")
    reset = form.form_submit_button("Réinitialiser le formulaire")

    #  Action prédiction 
    if submit:
        input_df = pd.DataFrame([{
            "age": age,
            "job": job,
            "marital": marital,
            "education": education,
            "default": default,
            "balance": balance,
            "housing": housing,
            "loan": loan,
            "contact": contact,
            "day": day,
            "month": month,
            "duration": duration,
            "campaign": campaign,
            "pdays": pdays,
            "previous": previous,
            "poutcome": poutcome
        }])

        cat_cols = ["job","marital","education","default","housing","loan","contact","month","poutcome"]
        cat_encoded = encoder.transform(input_df[cat_cols])
        cat_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(cat_cols))

        X_input = pd.concat([input_df.drop(columns=cat_cols), cat_df], axis=1)

        prediction = rf_model.predict(X_input)[0]
        proba = rf_model.predict_proba(X_input)[0][1]

        st.success(f"Prédiction : {'Yes' if prediction==1 else 'No'}")
        st.info(f"Probabilité d'acceptation : {round(proba*100,2)}%")

    #  Action réinitialisation 
    if reset:
        st.session_state.reset_counter += 1