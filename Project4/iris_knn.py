from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


#-----creat dataset-----
iris = load_iris()
type(iris)
X = iris.data
Y = iris.target
# print(iris.data.shape)

#-----spilt data-----
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.4, random_state = 7)

# print("X_train.shape:", X_train.shape)
# print("X_test.shape :", X_test.shape)

scores = {}
score_list = []
k_range = (1, 20)
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    scores[k] = metrics.accuracy_score(Y_test, Y_pred)
    score_list.append(metrics.accuracy_score(Y_test, Y_pred))


import matplotlib
import matplotlib.pyplot as plt

plt.plot(k_range, score_list)
plt.xlabel('Value of k for KNN')
plt.ylabel('Test Accuracy')
plt.show()