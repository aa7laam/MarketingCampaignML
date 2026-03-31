# 📊 Marketing Campaign Prediction App

## 🎯 Objectif du projet
Ce projet vise à développer un système intelligent capable de prédire si un client souscrira à un dépôt bancaire à partir de ses données personnelles et historiques.

---

## 💡 Contexte
Dans les campagnes marketing bancaires, il est important de cibler les clients les plus susceptibles d’accepter une offre.  
Ce projet utilise le Machine Learning pour améliorer la prise de décision.

---

## ⚙️ Fonctionnalités principales

✔ Analyse des données (visualisation interactive)  
✔ Prédiction en temps réel via un formulaire  
✔ Comparaison de plusieurs modèles de Machine Learning  
✔ Segmentation des clients avec clustering (K-Means)  
✔ Visualisation des clusters  

---

## 🧠 Modèles utilisés

### 🔹 Apprentissage supervisé
- Random Forest (modèle principal)
- Logistic Regression
- Decision Tree

### 🔹 Apprentissage non supervisé
- K-Means Clustering

---

## 📊 Résultats

- Identification de segments de clients
- Détection d’un groupe à fort potentiel (clients à cibler)
- Amélioration de la stratégie marketing

---

## 🖥️ Interface utilisateur

Application développée avec **Streamlit** permettant :
- Upload d’un dataset
- Visualisation des données
- Prédiction instantanée
- Analyse des clusters

---

## 📁 Structure du projet
MarketingCampaignML/
│
├── app/
│ ├── marketing_app.py
│ ├── random_forest_model.pkl
│ ├── logistic_regression_model.pkl
│ ├── decision_tree_model.pkl
│ ├── encoder.pkl
│
├── data/
│ ├── marketing_campaign.csv
│ ├── marketing_with_clusters.csv
│
├── train_random_forest.py
├── logistic_regression_model.py
├── decision_tree_model.py
├── kmeans_clustering.py
├── README.md  


## ▶️ Comment exécuter le projet

### 1. Installer les dépendances
```bash
pip install pandas scikit-learn streamlit altair matplotlib joblib
```
###  2. Entraîner les modèles
```bash
python train_random_forest.py
python logistic_regression_model.py
python decision_tree_model.py
```

### 3. Générer les clusters
```bash
python kmeans_clustering.py
```

### 4. Lancer l'application
```bash
streamlit run app/marketing_app.py

```
### Métriques utilisées
Accuracy
F1-score
Matrice de confusion
### Technologies utilisées
Python
Scikit-learn
Pandas
Streamlit
Altair
Matplotlib
# Auteur

Projet réalisé par FIJEH Ahlam, dans le cadre d’un mini-projet de Machine Learning