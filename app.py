from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
knn = KNeighborsClassifier(n_neighbors=1)

x = iris.data
y = iris.target

knn.fit(x, y)
species = knn.predict([[5.1,3.5,1.4,0.2]])[0]

print(iris.target_names[species])
