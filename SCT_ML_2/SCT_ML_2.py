
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


df = pd.read_csv("SCT_ML_2/data/Mall_Customers.csv")

print("First 5 Rows")
print(df.head())

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())


print("\nSummary Statistics")
print(df.describe())


X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

print("\nSelected Features")
print(X.head())



wcss = []

for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        init="k-means++",
        random_state=42
    )
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(True)

plt.savefig("SCT_ML_2_ELBOW.png", dpi=300, bbox_inches="tight")
plt.show()



kmeans = KMeans(
    n_clusters=5,
    init="k-means++",
    random_state=42
)

y_pred = kmeans.fit_predict(X)



plt.figure(figsize=(8, 6))

plt.scatter(
    X.iloc[y_pred == 0, 0],
    X.iloc[y_pred == 0, 1],
    s=80,
    label="Cluster 1"
)

plt.scatter(
    X.iloc[y_pred == 1, 0],
    X.iloc[y_pred == 1, 1],
    s=80,
    label="Cluster 2"
)

plt.scatter(
    X.iloc[y_pred == 2, 0],
    X.iloc[y_pred == 2, 1],
    s=80,
    label="Cluster 3"
)

plt.scatter(
    X.iloc[y_pred == 3, 0],
    X.iloc[y_pred == 3, 1],
    s=80,
    label="Cluster 4"
)

plt.scatter(
    X.iloc[y_pred == 4, 0],
    X.iloc[y_pred == 4, 1],
    s=80,
    label="Cluster 5"
)


plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=300,
    c="black",
    marker="X",
    label="Centroids"
)

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()

plt.savefig("SCT_ML_2_RESULT.png", dpi=300, bbox_inches="tight")
plt.show()

