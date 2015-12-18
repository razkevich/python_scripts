# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 10:18:37 2015

@author: razkevich
"""
from sklearn.linear_model import OrthogonalMatchingPursuit,RidgeClassifier
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble  import VotingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from nltk import word_tokenize

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC,NuSVC,LinearSVC
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet
from sklearn.linear_model.stochastic_gradient import SGDClassifier
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
from nltk.corpus import stopwords
from sklearn.grid_search import GridSearchCV
from sklearn import metrics










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
data= pd.read_json('d:\\train.json')[:]

ingredients=data.iloc[:,2].tolist()


#ingredients=list([str(len(x)),'a'+str(sum(list(len(y) for y in x)))]+x for x in data.iloc[:,2])#data.iloc[:,2]


classes=data.iloc[:,0]
del data
ps=PorterStemmer()
ii=0
ad=[]

for x in range(0,len(ingredients)-1):
   x1=ingredients[x]
   print str(x) +" of "+str(len(ingredients))
   ad=[]
   for y in x1:
       yrep=y.replace(" ","x")
       ii=ii+1
       for z in x1:
           zrep=z.replace(" ","x")
           ad=ad+[yrep+"x"+zrep]
   ingredients[x]=x1+ad
   
   
for x in range(0,len(ig['ingredients'])-1):
   x1=ig['ingredients'][x]
   print str(x) +" of "+str(len(ig['ingredients']))
   ad=[]
   for y in x1:
       yrep=y.replace(" ","x")
       ii=ii+1
       for z in x1:
           zrep=z.replace(" ","x")
           ad=ad+[yrep+"x"+zrep]
   ig['ingredients'][x]=x1+ad

print "loops"
ingredients=list(((" ".join(x))+' '+(" ".join(list(y.replace(' ','_') for y in x)))) for x in ingredients)
ig_t=list(((" ".join(x))+' '+(" ".join(list(y.replace(' ','_') for y in x)))) for x in ig['ingredients'])

print "ingredients"

#ig_t=list( list(y.replace(' ','_')+" "+(" ".join(map(lambda x:x.definition(),wn.synsets(y))) if wn.synsets(y).count>0 else "") for y in x) for x in ig['ingredients'])



print 'data loaded'




#parameters = dict(loss=['hinge', 'log', 'modified_huber', 'perceptron','squared_hinge', 'squared_loss', 'squared_epsilon_insensitive'],l1_ratio = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],penalty=['l2', 'elasticnet'],alpha= np.logspace(0.0001, 10, 5),n_jobs=[3],n_iter =[5,20])

#parameters = dict(loss=['hinge', 'log', 'perceptron','squared_epsilon_insensitive'],l1_ratio = [0.3,0.5,0.7,0.9],penalty=['l2', 'elasticnet'])


parameters = dict(C=[1,2,4,3],loss=[ 'squared_hinge'],dual  = [True,False],penalty=['l2'],multi_class= ['ovr'],fit_intercept   = [True,False])


vectorizer = CountVectorizer(min_df=4,binary=True)
transformer=TfidfTransformer(sublinear_tf=True )

X=transformer.fit_transform(vectorizer.fit_transform(ingredients))
X_train, X_test, Y_train, Y_test = train_test_split(X, classes, test_size=0.1)

from sklearn import cross_validation

cv = cross_validation.StratifiedKFold(  Y_train, 10)
grid_search = GridSearchCV(  LinearSVC(), parameters,verbose=10,error_score=0)
#grid_search.fit(X_train, Y_train)

#best_parameters = grid_search.best_estimator_.get_params()


classifier = LinearSVC(loss='squared_hinge', C=1,fit_intercept=True,penalty='l2',multi_class='ovr',dual=True)





#X_1=transformer.transform(vectorizer.transform(ingredients))


print 'classifier initialized'
classifier.fit(X_train,Y_train)
print 'classifier fit'

Y_pred=classifier.predict(X_test)
print 'predicted'
#Y_proba=(max(x) for x in classifier.predict_proba(X_test))
unfiltered=pd.DataFrame(data=[Y_pred,Y_test]).T






plot_confusion_matrix(confusion_matrix(unfiltered.iloc[:,1],unfiltered.iloc[:,0]))
print(classification_report(pd.DataFrame(unfiltered.iloc[:,1]),unfiltered.iloc[:,0]))




 