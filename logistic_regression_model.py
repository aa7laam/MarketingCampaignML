

# 1 IMPORT DES LIBRAIRIES

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


# 2 CHARGER LE DATASET

data = pd.read_csv("data/marketing_campaign.csv")

data.columns = data.columns.str.strip()


# 3SÉPARER LES VARIABLES

X = data.drop("deposit", axis=1)
y = data["deposit"].map({"yes": 1, "no": 0})  # Transformation de la cible


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


# 6 NORMALISATION

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7 CRÉATION DU MODÈLE

lr_model = LogisticRegression(max_iter=1000)


# 8 ENTRAÎNEMENT DU MODÈLE

lr_model.fit(X_train, y_train)


# 9 PRÉDICTION

y_pred = lr_model.predict(X_test)


# 10 ÉVALUATION

print("===== LOGISTIC REGRESSION =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nAnalyse :")
print("- Accuracy montre la performance globale")
print("- F1-score équilibre précision et rappel")
print("- La matrice de confusion montre où le modèle se trompe")


# 11 SAUVEGARDE DU MODÈLE

joblib.dump(lr_model, "app/logistic_regression_model.pkl")
print("Modèle Logistic Regression sauvegardé !")