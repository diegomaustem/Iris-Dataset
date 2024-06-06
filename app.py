from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

iris = load_iris()
knn = KNeighborsClassifier(n_neighbors=1)

x = iris.data
y = iris.target

knn.fit(x, y)
species = knn.predict([[5.1,3.5,1.4,0.2]])[0]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

knn.fit(x_train, y_train)
predictions = knn.predict(x_test)

hits = metrics.accuracy_score(y_test, predictions)

print(hits)
