
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

iris = datasets.load_iris()
x = iris.data
y = iris.target

x

y

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

cov_matrix = np.cov(x_scaled.T)
print(cov_matrix)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print("eigenvalues:",eigenvalues)
print("eigenvectors:",eigenvectors)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
colors = ['red', 'green', 'blue']
labels = iris.target_names
for i in range(len(colors)):
  ax.scatter(x_scaled[y == i, 0], x_scaled[y == i, 1], x_scaled[y == i, 2], colors)
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Petal Length')
ax.set_title('3D Visualization of Iris Data Before PCA')
plt.legend()
plt.show()

U, S, Vt = np.linalg.svd(x_scaled, full_matrices=False)
print("Singular Values:", S)

pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)

plt.figure(figsize=(8, 6))
for i in range(len(colors)):
    plt.scatter(x_pca[y == i, 0], x_pca[y == i, 1], color=colors[i], label=labels[i])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA on Iris Dataset (Dimensionality Reduction)')
plt.legend()
plt.grid()
plt.show()

