from sklearn import svm
from sklearn import datasets
import pickle

iris=datasets.load_iris()
X,y=iris.data,iris.target

# '''
clf=svm.SVC()
clf.fit(X,y)

# method 1:pickle
# save
with open('clf.pkl','wb') as f:
    pickle.dump(clf,f)
# '''
# restore
with open('clf.pkl','rb') as f:
    clf2=pickle.load(f)
    print(clf2.predict(X[0:1]))
# '''
