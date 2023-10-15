import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin

# Завантаження набору даних Iris
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
# Створення об'єкта K-Means для кластеризації
kmeans = KMeans(n_clusters=y.max() + 1, init='k-means++', n_init=10, max_iter=300,
                tol=0.0001, verbose=0, random_state=None, copy_x=True)
# Проведення кластеризації з використанням K-Means
kmeans.fit(X)
# Прогнозування кластерів для даних
y_pred = kmeans.predict(X)
# Візуалізація результатів
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()


# Функція для пошуку кластерів вручну
def find_clusters(X, n_clusters, rseed=2):
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    while True:
        # Пошук найближчих центроїдів для кожної точки
        labels = pairwise_distances_argmin(X, centers)
        # Перерахунок центроїдів
        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(n_clusters)])
        if np.all(centers == new_centers):
            break
        centers = new_centers
    return centers, labels


# Виклик функції find_clusters для пошуку кластерів вручну та візуалізація результатів
centers, labels = find_clusters(X, 3)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.show()
# Інший варіант виклику find_clusters з різними початковими умовами
centers, labels = find_clusters(X, 3, rseed=0)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.show()
# Використання K-Means для кластеризації з різними початковими умовами
labels = KMeans(3, random_state=0).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.show()
