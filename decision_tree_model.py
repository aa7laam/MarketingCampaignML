
# 1 IMPORT DES LIBRAIRIES

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


# 2 CHARGER LE DATASET

data = pd.read_csv("data/marketing_campaign.csv")
data.columns = data.columns.str.strip()  # Nettoyer les noms de colonnes


# 3 SÉPARER X et y

X = data.drop("deposit", axis=1)
y = data["deposit"].map({"yes": 1, "no": 0})


# 4 ENCODAGE DES VARIABLES CATÉGORIELLES

cat_cols = X.select_dtypes(include=['object']).columns.tolist()
encoder = OneHotEncoder(drop='first', sparse_output=False)
X_encoded = pd.DataFrame(
    encoder.fit_transform(X[cat_cols]),
    columns=encoder.get_feature_names_out(cat_cols)
)
X = X.drop(cat_cols, axis=1)
X = pd.concat([X.reset_index(drop=True), X_encoded.reset_index(drop=True)], axis=1)


# 5 SPLIT TRAIN / TEST

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 6 CRÉATION DU MODÈLE

dt_model = DecisionTreeClassifier(random_state=42)


# 7 ENTRAÎNEMENT

dt_model.fit(X_train, y_train)


# 8 PRÉDICTION

y_pred = dt_model.predict(X_test)


# 9 ÉVALUATION

print("===== DECISION TREE =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 10 ANALYSE

print("\nAnalyse :")
print("- Decision Tree est rapide et interprétable")
print("- Peut overfitter si l'arbre est trop profond")
print("- Utile pour comprendre les décisions du modèle")


# 11 SAUVEGARDE DU MODÈLE

joblib.dump(dt_model, "app/decision_tree_model.pkl")
print("Modèle Decision Tree sauvegardé !")