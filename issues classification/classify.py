# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
import numpy as np
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
   
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
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
import numpy
st=RussianStemmer()
libra=pd.read_excel('libra.xls')[['body','ticket_queue_id']].dropna()
libra.body=pd.Series(st.stem(x) for x in libra.body)

    
    
libra=libra.dropna()


classifier=SVC(probability=True,kernel='linear')
from nltk.stem.snowball import RussianStemmer
import nltk
st = RussianStemmer()
       

    
    
    
vectorizer = CountVectorizer(min_df=1,lowercase=True)
transformer=TfidfTransformer()

X=transformer.fit_transform(vectorizer.fit_transform(libra.body))

Y=libra.iloc[:,1]

X_train, X_test, Y_train, Y_test = train_test_split(X.toarray(), Y, test_size=0.15)




classifier.fit(X_train,Y_train)

Y_pred=classifier.predict(X_test)

Y_proba=(max(x) for x in classifier.predict_proba(X_test))

unfiltered=pd.DataFrame(data=[Y_pred,Y_proba,Y_test]).T
filtered=unfiltered[unfiltered.iloc[:,1]>0.8]

plot_confusion_matrix(confusion_matrix(filtered.iloc[:,2],filtered.iloc[:,0]))
print(classification_report(pd.DataFrame(filtered.iloc[:,2]),filtered.iloc[:,0]))