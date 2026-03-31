import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# 1 CHARGER DATASET

data = pd.read_csv("data/marketing_campaign.csv")
data.columns = data.columns.str.strip()

print("✅ Dataset chargé !")
print("Shape :", data.shape)


# 2 PRÉPARATION

X = data.drop("deposit", axis=1)

cat_cols = X.select_dtypes(include=['object']).columns.tolist()

encoder = OneHotEncoder(drop='first', sparse_output=False)

X_encoded = pd.DataFrame(
    encoder.fit_transform(X[cat_cols]),
    columns=encoder.get_feature_names_out(cat_cols)
)

X = X.drop(cat_cols, axis=1)
X = pd.concat([X.reset_index(drop=True), X_encoded.reset_index(drop=True)], axis=1)

print("✅ Encodage terminé")


# 3 NORMALISATION

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("✅ Normalisation terminée")


# 4 MÉTHODE DU COUDE

print("📊 Calcul du coude...")

inertia = []

for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    print(f"K={k} → Inertia={kmeans.inertia_}")

# Graphique sauvegardé 
plt.figure()
plt.plot(range(1, 10), inertia, marker='o')
plt.xlabel("Nombre de clusters")
plt.ylabel("Inertia")
plt.title("Méthode du coude")

plt.savefig("elbow_plot.png")  # 🔥 IMPORTANT
print("✅ Graphique sauvegardé : elbow_plot.png")


# 5 CLUSTERING FINAL

print("🚀 Clustering avec K=3...")

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

data["cluster"] = clusters


# 6 ANALYSE

print("\n===== ANALYSE DES CLUSTERS =====")

cluster_analysis = data.groupby("cluster").mean(numeric_only=True)
print(cluster_analysis)

print("\nDistribution deposit par cluster :")
print(pd.crosstab(data["cluster"], data["deposit"]))


# 7 SAUVEGARDE

data.to_csv("data/marketing_with_clusters.csv", index=False)
print("\n✅ Dataset avec clusters sauvegardé !")