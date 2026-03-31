
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
import joblib

# 1 Charger les données
data = pd.read_csv("data/marketing_campaign.csv", sep=',')
data.columns = data.columns.str.strip()  


# 2 Définir X et y
X = data.drop("deposit", axis=1)
y = data["deposit"].map({"yes": 1, "no": 0})  

# 3 Identifier les colonnes catégorielles
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# 4 Appliquer One-Hot Encoding sur les colonnes catégorielles
encoder = OneHotEncoder(drop='first', sparse_output=False)
X_encoded = pd.DataFrame(
    encoder.fit_transform(X[categorical_cols]),
    columns=encoder.get_feature_names_out(categorical_cols)
)


X = X.drop(categorical_cols, axis=1)
X = pd.concat([X.reset_index(drop=True), X_encoded.reset_index(drop=True)], axis=1)

# 5 Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6 Créer et entraîner le modèle
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 7 Prédictions et évaluation
y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 8 Sauvegarder le modèle et l'encodeur

import joblib
joblib.dump(rf, "app/random_forest_model.pkl")
joblib.dump(encoder, "app/encoder.pkl")
print("Modèle et encodeur sauvegardés avec succès !")