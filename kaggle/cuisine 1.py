# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 10:18:37 2015

@author: razkevich
"""
from sklearn.linear_model import OrthogonalMatchingPursuit,RidgeClassifier
import pandas as pd
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC,NuSVC,LinearSVC
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression,Perceptron
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier,RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import TruncatedSVD,NMF,FactorAnalysis,PCA
from nltk.stem.snowball import RussianStemmer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from nltk import word_tokenize
from nltk.tokenize.api import StringTokenizer
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
import numpy
import matplotlib.pyplot as plt
import numpy as np
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
   
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
import itertools
import pandas as pd
igt=pd.read_json('d:\\test.json')
ig=igt.iloc[:,1:2]
data=pd.read_json('d:\\train.json')[:]
ingredients=data.iloc[:,2]
#ingredients=list([str(len(x)),'a'+str(sum(list(len(y) for y in x)))]+x for x in data.iloc[:,2])#data.iloc[:,2]


classes=data.iloc[:,0]

ingredients=list((" ".join(x)+' '+" ".join(list(y.replace(' ','_') for y in x))) for x in ingredients)
ig_t=list((" ".join(x)+' '+" ".join(list(y.replace(' ','_') for y in x))) for x in ig['ingredients'])
print 'data loaded'
svc=LinearSVC()#SVC(verbose=True,probability=False,kernel='linear')
svd=TruncatedSVD(n_components=1000)

classifier=Pipeline(steps=[('svd',svd), ('svc', svc)])

vectorizer = CountVectorizer(min_df=1,lowercase=True)
transformer=TfidfTransformer()

X=transformer.fit_transform(vectorizer.fit_transform(ingredients))
X_1=transformer.transform(vectorizer.transform(ingredients))
X_train, X_test, Y_train, Y_test = train_test_split(X.toarray(), classes, test_size=0.1)

print 'classifier initialized'
classifier.fit(X_train,Y_train)
print 'classifier fit'

Y_pred=classifier.predict(X_test)
print 'predicted'
#Y_proba=(max(x) for x in classifier.predict_proba(X_test))
unfiltered=pd.DataFrame(data=[Y_pred,Y_test]).T






plot_confusion_matrix(confusion_matrix(unfiltered.iloc[:,1],unfiltered.iloc[:,0]))
print(classification_report(pd.DataFrame(unfiltered.iloc[:,1]),unfiltered.iloc[:,0]))





