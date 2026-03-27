import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


data = pd.read_csv("all.csv")
y = data['DoH']

# Drop non-numeric columns
data = data.drop([
    'SourceIP',
    'DestinationIP',
    'SourcePort',
    'DestinationPort',
    'TimeStamp',
    'DoH'
], axis=1)

data = data.fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# PCA Loadings
loadings = pd.DataFrame(
    pca.components_,
    columns=data.columns,
    index=['PC1', 'PC2']
)

# Top features for PC1 and PC2
top_pc1 = loadings.loc['PC1'].abs().sort_values(ascending=False)
top_pc2 = loadings.loc['PC2'].abs().sort_values(ascending=False)

print("\nMost important feature for PC1:", top_pc1.idxmax())
print("Most important feature for PC2:", top_pc2.idxmax())

print("\nTop 5 Features contributing to PC1:")
print(top_pc1.head(5))

print("\nTop 5 Features contributing to PC2:")
print(top_pc2.head(5))

pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['DoH'] = y
pca_df.to_csv("doh_pca_output.csv", index=False)

plt.figure()
for label in pca_df['DoH'].unique():
    subset = pca_df[pca_df['DoH'] == label]
    plt.scatter(subset['PC1'], subset['PC2'], label=label)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - DoH Firefox Dataset")
plt.legend()
plt.show()