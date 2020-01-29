#import dataset
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
Y = iris.target

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.5)

#we can use either decision or kneighbors classifier for the classification

#import Tree
#from sklearn import tree
#my_classi = tree.DecisionTreeClassifier()

from sklearn.neighbors import KNeighborsClassifier
my_classi = KNeighborsClassifier()


my_classi.fit(X_train,Y_train)

predications = my_classi.predict(X_test)
print(predications)

from sklearn.metrics import accuracy_score
print (accuracy_score(Y_test,predications)) 